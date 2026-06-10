from models.network_blocks import assemble_CNN_blocks, get_block_ops
import tensorflow as tf


# ===== KITTI RoITr-Lite modification start =====
def roitr_lite_ppf_encoding(features,
                            points,
                            neighbors,
                            out_dim=None,
                            geo_dim=16,
                            name='roitr_lite_ppf_encoding'):
    """
    Lightweight RoITr-style local geometric descriptor refinement for KITTI.

    This module is inspired by RoITr's rotation-invariant local geometry idea.
    It does NOT implement the full RoITr transformer matching pipeline. Instead,
    it computes compact PPF-like local geometric statistics and fuses them into
    the D3Feat descriptor branch.

    Memory note:
    KITTI has many points, so this module avoids dense layers on [N, K, C]
    neighbor feature tensors. It only uses compact [N, 12] geometric statistics.
    """

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        C = features.get_shape()[-1].value
        if out_dim is None:
            out_dim = C

        # Add one shadow point for invalid neighbor indices.
        shadow_point = tf.zeros_like(points[:1, :])
        points_with_shadow = tf.concat([points, shadow_point], axis=0)

        neighbor_points = tf.gather(points_with_shadow, neighbors)  # [N, K, 3]
        center_points = tf.expand_dims(points, axis=1)              # [N, 1, 3]

        relative_xyz = neighbor_points - center_points              # [N, K, 3]
        relative_dist = tf.norm(relative_xyz + 1e-9, axis=-1, keepdims=True)

        # PPF-like compact local geometry cues.
        # These features are cheap and avoid [N, K, C] dense operations.
        mean_xyz = tf.reduce_mean(relative_xyz, axis=1)              # [N, 3]
        max_xyz = tf.reduce_max(relative_xyz, axis=1)                # [N, 3]
        min_xyz = tf.reduce_min(relative_xyz, axis=1)                # [N, 3]
        mean_dist = tf.reduce_mean(relative_dist, axis=1)            # [N, 1]
        max_dist = tf.reduce_max(relative_dist, axis=1)              # [N, 1]
        std_dist = tf.sqrt(
            tf.reduce_mean(tf.square(relative_dist - tf.expand_dims(mean_dist, axis=1)), axis=1) + 1e-9
        )                                                           # [N, 1]

        local_geo = tf.concat(
            [mean_xyz, max_xyz, min_xyz, mean_dist, max_dist, std_dist],
            axis=-1
        )                                                           # [N, 12]

        local_geo = tf.layers.dense(local_geo,
                                    geo_dim,
                                    activation=tf.nn.relu,
                                    name='geo_fc1')
        local_geo = tf.layers.dense(local_geo,
                                    geo_dim,
                                    activation=tf.nn.relu,
                                    name='geo_fc2')

        fused = tf.concat([features, local_geo], axis=-1)
        refined = tf.layers.dense(fused,
                                  out_dim,
                                  activation=None,
                                  name='refine_fc')

        refined = features + refined
        refined = tf.nn.l2_normalize(refined, axis=1, epsilon=1e-10)

        return refined
# ===== KITTI RoITr-Lite modification end =====


def assemble_FCNN_blocks(inputs, config, dropout_prob):
    """
    D3Feat FCNN with KITTI RoITr-Lite descriptor refinement.
    The detector score branch remains the original D3Feat soft detection module.
    """

    # First get features from CNN
    F = assemble_CNN_blocks(inputs, config, dropout_prob)
    features = F[-1]

    # Current radius of convolution and feature dimension
    layer = config.num_layers - 1
    r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
    fdim = config.first_features_dim * 2 ** layer

    # Boolean of training
    training = dropout_prob < 0.99

    # Find first upsampling block
    start_i = 0
    for block_i, block in enumerate(config.architecture):
        if 'upsample' in block:
            start_i = block_i
            break

    # Loop over upsampling blocks
    block_in_layer = 0
    for block_i, block in enumerate(config.architecture[start_i:]):

        with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer)):

            block_ops = get_block_ops(block)

            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

        block_in_layer += 1

        if 'upsample' in block:
            layer -= 1
            r *= 0.5
            fdim = fdim // 2
            block_in_layer = 0
            features = tf.concat((features, F[layer]), axis=1)

    # Descriptor branch
    backup_features = tf.nn.l2_normalize(features, axis=1, epsilon=1e-10)

    # ===== KITTI RoITr-Lite modification start =====
    backup_features = roitr_lite_ppf_encoding(
        backup_features,
        inputs['points'][0],
        inputs['neighbors'][0],
        out_dim=backup_features.get_shape()[-1].value,
        geo_dim=16,
        name='kitti_roitr_lite_descriptor_refinement'
    )
    # ===== KITTI RoITr-Lite modification end =====

    # Soft Detection Module: original D3Feat score branch.
    neighbor = inputs['neighbors'][0]
    in_batches = inputs['in_batches']
    first_pcd_indices = in_batches[0]
    second_pcd_indices = in_batches[1]
    statcked_length = inputs['stack_lengths']
    first_pcd_length = statcked_length[0]
    second_pcd_length = statcked_length[1]

    # Add fake point for shadow neighbors.
    shadow_features = tf.zeros_like(features[:1, :])
    features = tf.concat([features, shadow_features], axis=0)
    shadow_neighbor = tf.ones_like(neighbor[:1, :]) * (first_pcd_length + second_pcd_length)
    neighbor = tf.concat([neighbor, shadow_neighbor], axis=0)

    # Normalize feature to avoid overflow.
    point_cloud_feature0 = tf.reduce_max(tf.gather(features, first_pcd_indices, axis=0))
    point_cloud_feature1 = tf.reduce_max(tf.gather(features, second_pcd_indices, axis=0))
    max_per_sample = tf.concat([
        tf.cast(tf.ones([first_pcd_length, 1]), tf.float32) * point_cloud_feature0,
        tf.cast(tf.ones([second_pcd_length + 1, 1]), tf.float32) * point_cloud_feature1],
        axis=0)
    features = tf.divide(features, max_per_sample + 1e-6)

    # Local max score.
    neighbor_features = tf.gather(features, neighbor, axis=0)
    neighbor_features_sum = tf.reduce_sum(neighbor_features, axis=-1)
    neighbor_num = tf.count_nonzero(neighbor_features_sum, axis=-1, keepdims=True)
    neighbor_num = tf.maximum(neighbor_num, 1)
    mean_features = tf.reduce_sum(neighbor_features, axis=1) / tf.cast(neighbor_num, tf.float32)
    local_max_score = tf.math.softplus(features - mean_features)

    # Depth-wise max score.
    depth_wise_max = tf.reduce_max(features, axis=1, keepdims=True)
    depth_wise_max_score = features / (1e-6 + depth_wise_max)

    # Original D3Feat score.
    all_score = local_max_score * depth_wise_max_score
    score = tf.reduce_max(all_score, axis=1, keepdims=True)

    return backup_features, score[:-1, :]
