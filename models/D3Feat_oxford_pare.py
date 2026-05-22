from models.network_blocks import assemble_CNN_blocks, get_block_ops
import tensorflow as tf


def position_aware_local_encoding(features, points, neighbors, name='pare_position_aware_encoding'):
    """
    PARE-style position-aware local geometric encoding for D3Feat TensorFlow code base.

    This is NOT the official PARE-Net implementation.  It keeps the original D3Feat
    Oxford pipeline and adds a light-weight local position-aware descriptor refinement:
      1. gather local neighbor features and relative positions
      2. encode relative xyz and distance
      3. use the encoded geometric relation to aggregate local feature context
      4. fuse the context back into the descriptor with a residual connection

    features:  [N, C]
    points:    [N, 3]
    neighbors: [N, K], shadow neighbor index is handled safely
    return:    [N, C]
    """

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):

        C = features.get_shape()[-1].value
        if C is None:
            C = tf.shape(features)[-1]

        # Shadow point / feature for invalid neighbors.
        shadow_feature = tf.zeros_like(features[:1, :])
        features_with_shadow = tf.concat([features, shadow_feature], axis=0)

        shadow_point = tf.zeros_like(points[:1, :])
        points_with_shadow = tf.concat([points, shadow_point], axis=0)

        # Gather local neighborhoods.
        neigh_features = tf.gather(features_with_shadow, neighbors)  # [N, K, C]
        neigh_points = tf.gather(points_with_shadow, neighbors)      # [N, K, 3]
        center_points = tf.expand_dims(points, axis=1)               # [N, 1, 3]

        # Position-aware geometric relation: relative xyz + local distance.
        rel_pos = neigh_points - center_points                       # [N, K, 3]
        rel_dist = tf.norm(rel_pos + 1e-9, axis=-1, keepdims=True)    # [N, K, 1]
        geo_input = tf.concat([rel_pos, rel_dist], axis=-1)           # [N, K, 4]

        # Lightweight geometric embedding. Keep it simple for TF1.15 stability.
        geo_hidden = tf.layers.dense(geo_input, C, activation=tf.nn.relu, name='geo_fc1')
        geo_weight = tf.layers.dense(geo_hidden, C, activation=tf.nn.sigmoid, name='geo_fc2')

        # Position-aware aggregation.
        weighted_neigh = neigh_features * geo_weight

        # Mask shadow neighbors. In D3Feat/KPConv neighbors, shadow index usually points to N.
        num_valid_points = tf.shape(features)[0]
        valid_mask = tf.cast(tf.less(neighbors, num_valid_points), tf.float32)  # [N, K]
        valid_mask = tf.expand_dims(valid_mask, axis=-1)                        # [N, K, 1]
        weighted_neigh = weighted_neigh * valid_mask

        denom = tf.reduce_sum(valid_mask, axis=1)
        denom = tf.maximum(denom, 1.0)
        local_context = tf.reduce_sum(weighted_neigh, axis=1) / denom           # [N, C]

        # Residual descriptor refinement + projection.
        fused = tf.concat([features, local_context], axis=1)
        refined = tf.layers.dense(fused, C, activation=None, name='fuse_fc')
        refined = tf.nn.l2_normalize(features + refined, axis=1, epsilon=1e-10)

        return refined


def assemble_FCNN_blocks(inputs, config, dropout_prob):
    """
    D3Feat FCNN with a PARE-style position-aware descriptor refinement.
    Oxford settings, dataset, loss, detector score, trainer, tester are unchanged.
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
            features = block_ops(layer, inputs, features, r, fdim, config, training)

        block_in_layer += 1

        if 'upsample' in block:
            layer -= 1
            r *= 0.5
            fdim = fdim // 2
            block_in_layer = 0
            features = tf.concat((features, F[layer]), axis=1)

    # Descriptor branch: original D3Feat descriptor + PARE-style position-aware local encoding.
    backup_features = tf.nn.l2_normalize(features, axis=1, epsilon=1e-10)
    backup_features = position_aware_local_encoding(
        backup_features,
        inputs['points'][0],
        inputs['neighbors'][0],
        name='pare_position_aware_encoding'
    )

    # Soft Detection Module: keep original D3Feat score for fair comparison.
    neighbor = inputs['neighbors'][0]
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

    # Normalize feature values per sample to avoid overflow.
    point_cloud_feature0 = tf.reduce_max(tf.gather(features, first_pcd_indices, axis=0))
    point_cloud_feature1 = tf.reduce_max(tf.gather(features, second_pcd_indices, axis=0))
    max_per_sample = tf.concat([
        tf.cast(tf.ones([first_pcd_length, 1]), tf.float32) * point_cloud_feature0,
        tf.cast(tf.ones([second_pcd_length + 1, 1]), tf.float32) * point_cloud_feature1],
        axis=0)
    features = tf.divide(features, max_per_sample + 1e-6)

    # local max score (saliency score)
    neighbor_features = tf.gather(features, neighbor, axis=0)
    neighbor_features_sum = tf.reduce_sum(neighbor_features, axis=-1)
    neighbor_num = tf.count_nonzero(neighbor_features_sum, axis=-1, keepdims=True)
    neighbor_num = tf.maximum(neighbor_num, 1)
    mean_features = tf.reduce_sum(neighbor_features, axis=1) / tf.cast(neighbor_num, tf.float32)
    local_max_score = tf.math.softplus(features - mean_features)

    # depth-wise max score
    depth_wise_max = tf.reduce_max(features, axis=1, keepdims=True)
    depth_wise_max_score = features / (1e-6 + depth_wise_max)

    # Original D3Feat score, without Level1 attention reweighting.
    all_score = local_max_score * depth_wise_max_score
    score = tf.reduce_max(all_score, axis=1, keepdims=True)

    return backup_features, score[:-1, :]
