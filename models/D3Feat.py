from models.network_blocks import assemble_CNN_blocks, get_block_ops
import tensorflow as tf


def assemble_FCNN_blocks(inputs, config, dropout_prob):
    """
    Definition of all the layers according to config
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, upsamples, features, batches, labels]
    :param config:
    :param dropout_prob:
    :return:
    """

    # First get features from CNN
    F = assemble_CNN_blocks(inputs, config, dropout_prob)
    features = F[-1]

    # Current radius of convolution and feature dimension
    layer = config.num_layers - 1
    r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
    fdim = config.first_features_dim * 2 ** layer  # if you use resnet, fdim is actually 2 times that

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

            # Get the function for this layer
            block_ops = get_block_ops(block)

            # Apply the layer function defining tf ops
            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

        # Index of block in this layer
        block_in_layer += 1

        # Detect change to a subsampled layer
        if 'upsample' in block:
            # Update radius and feature dimension for next layer
            layer -= 1
            r *= 0.5
            fdim = fdim // 2
            block_in_layer = 0

            # Concatenate with CNN feature map
            features = tf.concat((features, F[layer]), axis=1)

    ### edit by yunsheng add attention backbone LEVEL 2
    # original feature for descriptor
    backup_features = tf.nn.l2_normalize(features, axis=1, epsilon=1e-10)


    ### edit by yunsheng add attention backbone LEVEL 2
    with tf.variable_scope('self_attention'):
        neighbor_idx = inputs['neighbors'][0]   # [N, K]

        # add shadow point for safe gather
        shadow_features_attn = tf.zeros_like(features[:1, :])
        features_for_attn = tf.concat([features, shadow_features_attn], axis=0)

        # gather neighbor features
        neighbor_feat = tf.gather(features_for_attn, neighbor_idx, axis=0)  # [N, K, C]

        # central feature
        central_feat = tf.expand_dims(features, axis=1)  # [N, 1, C]

        C = features.get_shape().as_list()[1]
        d = max(C // 4, 8)

        # Q K 降維，V 保持原通道
        Q = tf.layers.dense(central_feat, d, name='q')   # [N, 1, d]
        K = tf.layers.dense(neighbor_feat, d, name='k')  # [N, K, d]
        V = tf.layers.dense(neighbor_feat, C, name='v')  # [N, K, C]

        # attention score
        attn = tf.reduce_sum(Q * K, axis=-1)             # [N, K]
        attn = attn / tf.sqrt(tf.cast(d, tf.float32))
        attn = tf.nn.softmax(attn, axis=-1)

        # weighted sum
        attn = tf.expand_dims(attn, axis=-1)             # [N, K, 1]
        new_feat = tf.reduce_sum(attn * V, axis=1)       # [N, C]



    # Soft Detection Module
    neighbor = inputs['neighbors'][0]  # [n_points, n_neighbors]
    in_batches = inputs['in_batches']
    first_pcd_indices = in_batches[0]
    second_pcd_indices = in_batches[1]
    statcked_length = inputs['stack_lengths']
    first_pcd_length = statcked_length[0]
    second_pcd_length = statcked_length[1]

    # use attn_features for detection branch
    det_features = features

    # add a fake point in the last row for shadow neighbors
    shadow_features = tf.zeros_like(det_features[:1, :])
    det_features = tf.concat([det_features, shadow_features], axis=0)

    shadow_neighbor = tf.ones_like(neighbor[:1, :]) * (first_pcd_length + second_pcd_length)
    neighbor = tf.concat([neighbor, shadow_neighbor], axis=0)

    # normalize the feature to avoid overflow
    point_cloud_feature0 = tf.reduce_max(tf.gather(det_features, first_pcd_indices, axis=0))
    point_cloud_feature1 = tf.reduce_max(tf.gather(det_features, second_pcd_indices, axis=0))
    max_per_sample = tf.concat([
        tf.cast(tf.ones([first_pcd_length, 1]), tf.float32) * point_cloud_feature0,
        tf.cast(tf.ones([second_pcd_length + 1, 1]), tf.float32) * point_cloud_feature1],
        axis=0)
    det_features = tf.divide(det_features, max_per_sample + 1e-6)

    # local max score
    neighbor_features = tf.gather(det_features, neighbor, axis=0)
    neighbor_features_sum = tf.reduce_sum(neighbor_features, axis=-1)
    neighbor_num = tf.count_nonzero(neighbor_features_sum, axis=-1, keepdims=True)
    neighbor_num = tf.maximum(neighbor_num, 1)
    mean_features = tf.reduce_sum(neighbor_features, axis=1) / tf.cast(neighbor_num, tf.float32)
    local_max_score = tf.math.softplus(det_features - mean_features)

    # depth-wise max score
    depth_wise_max = tf.reduce_max(det_features, axis=1, keepdims=True)
    depth_wise_max_score = det_features / (1e-6 + depth_wise_max)


    ### edit by yunsheng
    # all_score = local_max_score * depth_wise_max_score
    # # use the max score among channel to be the score of a single point. 
    # score = tf.reduce_max(all_score, axis=1, keepdims=True)  # [n_points, 1]

    # # hard selection (used during test)
    # # local_max = tf.reduce_max(neighbor_features, axis=1)
    # # is_local_max = tf.equal(features, local_max)
    # # is_local_max = tf.Print(is_local_max, [tf.reduce_sum(tf.cast(is_local_max, tf.int32))], message='num of local max')
    # # detected = tf.reduce_max(tf.cast(is_local_max, tf.float32), axis=1, keepdims=True)
    # # score = score * detected

    # return backup_features, score[:-1, :]

    ### edit by yunsheng
    ### edit by yunsheng
    ### edit by yunsheng

    all_score = local_max_score * depth_wise_max_score
    score = tf.reduce_max(all_score, axis=1, keepdims=True)  # [n_points+1, 1]

    # remove shadow point
    score = score[:-1, :]   # [n_points, 1]

    with tf.variable_scope('context_head'):
        context_weight = tf.layers.dense(new_feat, 1, name='context_fc')
        context_weight = tf.nn.sigmoid(context_weight)

    # attention branch
    with tf.variable_scope('attention_head'):
        attn_score = tf.layers.dense(backup_features, 1, name='attn_fc')
        attn_score = tf.nn.sigmoid(attn_score)

    # reweight keypoint score
    score = score * attn_score * context_weight

    return backup_features, score

    ### edit by yunsheng
