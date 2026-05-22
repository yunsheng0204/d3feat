from models.network_blocks import assemble_CNN_blocks, get_block_ops
import tensorflow as tf


# ===== KITTI PARE-style modification start =====
def position_aware_local_encoding(features,
                                  points,
                                  neighbors,
                                  out_dim=None,
                                  name='pare_position_aware_local_encoding'):
    """
    Lightweight PARE-style position-aware local geometric encoding.

    This module is designed for the existing D3Feat TensorFlow 1.x code base.
    It keeps the original D3Feat detector-score branch unchanged, and only
    refines the descriptor branch using local relative geometry.

    Args:
        features:  [N, C] point features from the D3Feat decoder.
        points:    [N, 3] point coordinates at the original resolution.
        neighbors: [N, K] neighbor indices. Shadow neighbors are handled safely.
        out_dim:   output descriptor dimension. If None, use C.
        name:      TensorFlow variable scope name.

    Returns:
        refined descriptor features: [N, C]
    """

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        C = features.get_shape()[-1].value
        if out_dim is None:
            out_dim = C

        # Add one shadow point/feature so invalid neighbor indices can gather safely.
        shadow_feature = tf.zeros_like(features[:1, :])
        features_with_shadow = tf.concat([features, shadow_feature], axis=0)

        shadow_point = tf.zeros_like(points[:1, :])
        points_with_shadow = tf.concat([points, shadow_point], axis=0)

        neighbor_features = tf.gather(features_with_shadow, neighbors)  # [N, K, C]
        neighbor_points = tf.gather(points_with_shadow, neighbors)      # [N, K, 3]

        center_points = tf.expand_dims(points, axis=1)                  # [N, 1, 3]
        relative_xyz = neighbor_points - center_points                  # [N, K, 3]
        relative_dist = tf.norm(relative_xyz + 1e-9, axis=-1, keepdims=True)

        # Local position encoding: relative xyz + relative distance.
        local_geometry = tf.concat([relative_xyz, relative_dist], axis=-1)  # [N, K, 4]
        local_geometry = tf.layers.dense(local_geometry,
                                         out_dim,
                                         activation=tf.nn.relu,
                                         name='geo_fc1')
        local_geometry = tf.layers.dense(local_geometry,
                                         out_dim,
                                         activation=None,
                                         name='geo_fc2')

        # Fuse neighbor semantic feature and geometric encoding.
        local_context = tf.concat([neighbor_features, local_geometry], axis=-1)
        local_context = tf.layers.dense(local_context,
                                        out_dim,
                                        activation=tf.nn.relu,
                                        name='fusion_fc1')
        local_context = tf.reduce_max(local_context, axis=1)  # [N, C]

        refined = tf.layers.dense(tf.concat([features, local_context], axis=-1),
                                  out_dim,
                                  activation=None,
                                  name='refine_fc')

        # Residual connection + L2 normalization for descriptor stability.
        refined = features + refined
        refined = tf.nn.l2_normalize(refined, axis=1, epsilon=1e-10)

        return refined
# ===== KITTI PARE-style modification end =====


def assemble_FCNN_blocks(inputs, config, dropout_prob):
    """
    Definition of all the layers according to config.
    This KITTI PARE-style version keeps the D3Feat detection-score computation
    unchanged and only changes the output descriptor branch.
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

    # ===== KITTI PARE-style modification start =====
    backup_features = position_aware_local_encoding(
        backup_features,
        inputs['points'][0],
        inputs['neighbors'][0],
        out_dim=backup_features.get_shape()[-1].value,
        name='kitti_pare_descriptor_refinement'
    )
    # ===== KITTI PARE-style modification end =====

    # Soft Detection Module: keep original D3Feat score branch unchanged.
    neighbor = inputs['neighbors'][0]  # [n_points, n_neighbors]
    in_batches = inputs['in_batches']
    first_pcd_indices = in_batches[0]
    second_pcd_indices = in_batches[1]
    statcked_length = inputs['stack_lengths']
    first_pcd_length = statcked_length[0]
    second_pcd_length = statcked_length[1]

    # add a fake point in the last row for shadow neighbors
    shadow_features = tf.zeros_like(features[:1, :])
    features = tf.concat([features, shadow_features], axis=0)
    shadow_neighbor = tf.ones_like(neighbor[:1, :]) * (first_pcd_length + second_pcd_length)
    neighbor = tf.concat([neighbor, shadow_neighbor], axis=0)

    # normalize feature to avoid overflow
    point_cloud_feature0 = tf.reduce_max(tf.gather(features, first_pcd_indices, axis=0))
    point_cloud_feature1 = tf.reduce_max(tf.gather(features, second_pcd_indices, axis=0))
    max_per_sample = tf.concat([
        tf.cast(tf.ones([first_pcd_length, 1]), tf.float32) * point_cloud_feature0,
        tf.cast(tf.ones([second_pcd_length + 1, 1]), tf.float32) * point_cloud_feature1],
        axis=0)
    features = tf.divide(features, max_per_sample + 1e-6)

    # local max score
    neighbor_features = tf.gather(features, neighbor, axis=0)
    neighbor_features_sum = tf.reduce_sum(neighbor_features, axis=-1)
    neighbor_num = tf.count_nonzero(neighbor_features_sum, axis=-1, keepdims=True)
    neighbor_num = tf.maximum(neighbor_num, 1)
    mean_features = tf.reduce_sum(neighbor_features, axis=1) / tf.cast(neighbor_num, tf.float32)
    local_max_score = tf.math.softplus(features - mean_features)

    # depth-wise max score
    depth_wise_max = tf.reduce_max(features, axis=1, keepdims=True)
    depth_wise_max_score = features / (1e-6 + depth_wise_max)

    # Original D3Feat score
    all_score = local_max_score * depth_wise_max_score
    score = tf.reduce_max(all_score, axis=1, keepdims=True)

    return backup_features, score[:-1, :]
