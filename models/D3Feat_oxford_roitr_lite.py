from models.network_blocks import assemble_CNN_blocks, get_block_ops
import tensorflow as tf


def roitr_lite_ppf_encoding(features, points, neighbors, out_dim=None, geo_dim=16, name='roitr_lite_ppf_encoding'):
    """
    Lightweight RoITr-style descriptor refinement for the existing D3Feat pipeline.

    It uses compact PPF-like local geometric statistics instead of the full RoITr
    Transformer pipeline. This keeps the Oxford training/test/evaluation settings
    unchanged and avoids large [N, K, C] tensors.
    """
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        C = features.get_shape()[-1].value
        if out_dim is None:
            out_dim = C

        shadow_point = tf.zeros_like(points[:1, :])
        points_with_shadow = tf.concat([points, shadow_point], axis=0)

        neighbor_points = tf.gather(points_with_shadow, neighbors)  # [N, K, 3]
        center_points = tf.expand_dims(points, axis=1)              # [N, 1, 3]

        rel_xyz = neighbor_points - center_points                   # [N, K, 3]
        rel_dist = tf.norm(rel_xyz + 1e-9, axis=-1, keepdims=True)  # [N, K, 1]

        # Compact RoITr-style PPF/local geometry statistics.
        mean_xyz = tf.reduce_mean(rel_xyz, axis=1)                  # [N, 3]
        max_xyz = tf.reduce_max(rel_xyz, axis=1)                    # [N, 3]
        min_xyz = tf.reduce_min(rel_xyz, axis=1)                    # [N, 3]
        mean_dist = tf.reduce_mean(rel_dist, axis=1)                # [N, 1]
        max_dist = tf.reduce_max(rel_dist, axis=1)                  # [N, 1]
        std_dist = tf.sqrt(tf.reduce_mean(tf.square(rel_dist - tf.expand_dims(mean_dist, axis=1)), axis=1) + 1e-9)

        local_geo = tf.concat([mean_xyz, max_xyz, min_xyz, mean_dist, max_dist, std_dist], axis=-1)  # [N, 12]

        local_geo = tf.layers.dense(local_geo, geo_dim, activation=tf.nn.relu, name='geo_fc1')
        local_geo = tf.layers.dense(local_geo, geo_dim, activation=tf.nn.relu, name='geo_fc2')

        refined = tf.layers.dense(tf.concat([features, local_geo], axis=-1), out_dim, activation=None, name='refine_fc')
        refined = features + refined
        refined = tf.nn.l2_normalize(refined, axis=1, epsilon=1e-10)
        return refined


def assemble_FCNN_blocks(inputs, config, dropout_prob):
    """D3Feat FCNN with Oxford RoITr-Lite descriptor refinement."""

    F = assemble_CNN_blocks(inputs, config, dropout_prob)
    features = F[-1]

    layer = config.num_layers - 1
    r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
    fdim = config.first_features_dim * 2 ** layer
    training = dropout_prob < 0.99

    start_i = 0
    for block_i, block in enumerate(config.architecture):
        if 'upsample' in block:
            start_i = block_i
            break

    block_in_layer = 0
    for block_i, block in enumerate(config.architecture[start_i:]):
        with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer)):
            block_ops = get_block_ops(block)
            features = block_ops(layer, inputs, features, r, fdim, config, training)

        block_in_layer += 1

        if 'upsample' in block:
            layer -= 1
            r *= 0.5
            fdim = fdim // 2
            block_in_layer = 0
            features = tf.concat((features, F[layer]), axis=1)

    # Descriptor branch: RoITr-Lite refinement.
    backup_features = tf.nn.l2_normalize(features, axis=1, epsilon=1e-10)
    backup_features = roitr_lite_ppf_encoding(
        backup_features,
        inputs['points'][0],
        inputs['neighbors'][0],
        out_dim=backup_features.get_shape()[-1].value,
        geo_dim=16,
        name='oxford_roitr_lite_descriptor_refinement'
    )

    # Original D3Feat soft detection score branch.
    neighbor = inputs['neighbors'][0]
    in_batches = inputs['in_batches']
    first_pcd_indices = in_batches[0]
    second_pcd_indices = in_batches[1]
    statcked_length = inputs['stack_lengths']
    first_pcd_length = statcked_length[0]
    second_pcd_length = statcked_length[1]

    shadow_features = tf.zeros_like(features[:1, :])
    features = tf.concat([features, shadow_features], axis=0)
    shadow_neighbor = tf.ones_like(neighbor[:1, :]) * (first_pcd_length + second_pcd_length)
    neighbor = tf.concat([neighbor, shadow_neighbor], axis=0)

    point_cloud_feature0 = tf.reduce_max(tf.gather(features, first_pcd_indices, axis=0))
    point_cloud_feature1 = tf.reduce_max(tf.gather(features, second_pcd_indices, axis=0))
    max_per_sample = tf.concat([
        tf.cast(tf.ones([first_pcd_length, 1]), tf.float32) * point_cloud_feature0,
        tf.cast(tf.ones([second_pcd_length + 1, 1]), tf.float32) * point_cloud_feature1],
        axis=0)
    features = tf.divide(features, max_per_sample + 1e-6)

    neighbor_features = tf.gather(features, neighbor, axis=0)
    neighbor_features_sum = tf.reduce_sum(neighbor_features, axis=-1)
    neighbor_num = tf.count_nonzero(neighbor_features_sum, axis=-1, keepdims=True)
    neighbor_num = tf.maximum(neighbor_num, 1)
    mean_features = tf.reduce_sum(neighbor_features, axis=1) / tf.cast(neighbor_num, tf.float32)
    local_max_score = tf.math.softplus(features - mean_features)

    depth_wise_max = tf.reduce_max(features, axis=1, keepdims=True)
    depth_wise_max_score = features / (1e-6 + depth_wise_max)

    all_score = local_max_score * depth_wise_max_score
    score = tf.reduce_max(all_score, axis=1, keepdims=True)

    return backup_features, score[:-1, :]
