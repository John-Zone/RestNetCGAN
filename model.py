import tensorflow as tf
from ops import *
import collections
import scipy.io as scio
import numpy as np


# noise应是shape为[batch,length]的tensor,输出[batch,length,1]
def generator_spe(embedded_vector, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator_spe')

    def residual_block1d(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            net = conv1d(inputs, 5, output_channel, stride, use_bias=False, scope='conv1d_1')
            net = batchnorm(net, FLAGS.is_training)
            net = prelu_tf(net)
            net = conv1d(net, 5, output_channel, stride, use_bias=False, scope='conv1d_2')
            net = batchnorm(net, FLAGS.is_training)
            net = net + inputs
        return net

    # network start
    with tf.variable_scope('generator_spe'):

        net = denselayer(embedded_vector, FLAGS.channel_num_spectral)
        net = tf.expand_dims(net, axis=2)

        net = conv1d(net, 5, 32, 1, use_bias=False, scope='conv1d1')
        stage = net

        for i in range(1, FLAGS.num_resblock + 1, 1):
            name_scope = 'resblock1d_%d' % (i)
            net = residual_block1d(net, 32, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv1d(net, 5, 32, 1, use_bias=False, scope='output_res')
            net = net + stage

        net = conv1d(net, 7, 1, 1, use_bias=False, scope='conv2d_final')

    return net


def generator_spa(embedded_vector, reuse=False, FLAGS=None):
    # Check the flag
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator_spe')

    def residual_block2d(inputs, output_channel, stride, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            net = conv2d(inputs, 3, output_channel, stride, use_bias=False, scope='conv2d_1')
            net = batchnorm(net, FLAGS.is_training)
            net = prelu_tf(net)
            net = conv2d(net, 3, output_channel, stride, use_bias=False, scope='conv2d_2')
            net = batchnorm(net, FLAGS.is_training)
            net = net + inputs
        return net

    with tf.variable_scope('generator_spa'):
        net = tf.reshape(embedded_vector, [FLAGS.batch_size, 5, 5, -1])  # 5*5
        # o =s(i−1)+k−2p
        net = deconv2d(net, 32, 5, 1, True, 'deconv1')  # 9*9
        net = batchnorm(net, FLAGS.is_training)
        net = lrelu(net, 0.2)
        net = conv2d(net, 3, 32, 1, True, 'conv2d1')  # 9*9
        net = batchnorm(net, FLAGS.is_training)
        net = lrelu(net, 0.2)
        net = deconv2d(net, 32, 3, 1, True, 'deconv2')  # 11*11
        net = batchnorm(net, FLAGS.is_training)
        net = lrelu(net, 0.2)

        stage = net

        for i in range(1, FLAGS.num_resblock + 1, 1):
            name_scope = 'resblock2d_%d' % (i)
            net = residual_block2d(net, 32, 1, name_scope)

        net = net + stage
        net = conv2d(net, 5, FLAGS.channel_num_spatial, 1, True, 'res_output')

    return net


# Define the discriminator block

# input_spe是[batch,length,1]的tensor,input_spa是[batch,d,w,h,c]的tensor
def discriminator(input_spe, input_spa, FLAGS):
    def dis_conv1d_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            net = conv1d(inputs, kernel_size, output_channel, stride, scope='dis_conv1d')
            net = batchnorm(net, FLAGS.is_training)
            net = lrelu(net, 0.2)
        return net

    def dis_conv2d_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            net = conv2d(inputs, kernel_size, output_channel, stride, scope='dis_conv2d')
            net = batchnorm(net, FLAGS.is_training)
            net = lrelu(net, 0.2)
        return net

    with tf.variable_scope('spe_dis', reuse=tf.AUTO_REUSE):
        spe_map = dis_conv1d_block(input_spe, 8, 3, 1, 'dis_conv1d1')
        spe_map = dis_conv1d_block(spe_map, 8, 3, 1, 'dis_conv1d2')
        spe_map = dis_conv1d_block(spe_map, 16, 3, 1, 'dis_conv1d3')
        spe_map = dis_conv1d_block(spe_map, 16, 3, 1, 'dis_conv1d4')
        spe_map = dis_conv1d_block(spe_map, 32, 3, 1, 'dis_conv1d5')
        spe_map = dis_conv1d_block(spe_map, 32, 3, 1, 'dis_conv1d6')
        spe_map = dis_conv1d_block(spe_map, 32, 3, 1, 'dis_conv1d7')
        spe_map = slim.flatten(spe_map)

    with tf.variable_scope('spa_dis', reuse=tf.AUTO_REUSE):
        spa_map = dis_conv2d_block(input_spa, 8, 3, 1, 'dis_conv2d1')
        spa_map = dis_conv2d_block(spa_map, 8, 3, 1, 'dis_conv2d2')
        spa_map = dis_conv2d_block(spa_map, 16, 3, 1, 'dis_conv2d3')
        spa_map = dis_conv2d_block(spa_map, 16, 3, 1, 'dis_conv2d4')
        spa_map = dis_conv2d_block(spa_map, 32, 3, 1, 'dis_conv2d5')
        spa_map = dis_conv2d_block(spa_map, 32, 3, 1, 'dis_conv2d6')
        spa_map = dis_conv2d_block(spa_map, 32, 3, 1, 'dis_conv2d7')
        spa_map = slim.flatten(spa_map)

    feature = tf.concat([spa_map, spe_map], axis=1)

    with tf.variable_scope('dense_layer_1', reuse=tf.AUTO_REUSE):
        class_pro = denselayer(feature, FLAGS.class_num)

    with tf.variable_scope('dense_layer_2', reuse=tf.AUTO_REUSE):
        real_pro = denselayer(feature, 1)
        real_pro = tf.sigmoid(real_pro)

    return class_pro, real_pro


def model(embedded_z1, embedded_z2, target_spe, target_spa, label, FLAGS):
    # Define the container of the parameter
    Network = collections.namedtuple('Network', 'discrim_real_output_c, discrim_fake_output_c, discrim_loss,gen_loss, \
     gen_grads_and_vars, gen_output_spa,gen_output_spe, train, global_step,discrim_grads_and_vars,train_only_dis \
    learning_rate_dis, learning_rate_gen,clip_discrim_train')

    # Build the two generator part
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        gen_output_spe = generator_spe(embedded_z1, reuse=False, FLAGS=FLAGS)
        gen_output_spa = generator_spa(embedded_z2, reuse=False, FLAGS=FLAGS)

    # Build the fake discriminator
    with tf.name_scope('discriminator_real'):
        with tf.variable_scope('discriminator'):
            discrim_real_output_c, discrim_real_output_s = discriminator(target_spe, target_spa, FLAGS)

    with tf.name_scope('discriminator_fake'):
        with tf.variable_scope('discriminator'):
            discrim_fake_output_c, discrim_fake_output_s = discriminator(gen_output_spe, gen_output_spa, FLAGS)

    # with tf.variable_scope('spectral_loss', reuse=tf.AUTO_REUSE):
    #     gen_dis = tf.sqrt(tf.reduce_sum(gen_output_spe * gen_output_spe, axis=1))
    #     target_dis = tf.sqrt(tf.reduce_sum(target_spe * target_spe, axis=1))
    #     temp = tf.reduce_sum(gen_output_spe * target_spe, axis=1)
    #     spectral_loss = tf.reduce_mean(tf.math.acos(temp / (gen_dis * target_dis)))

    with tf.variable_scope('Ls'):
        Ls = -tf.reduce_mean(
            tf.log(discrim_real_output_s + FLAGS.EPS) + tf.log(1 - discrim_fake_output_s + FLAGS.EPS))

    with tf.variable_scope('Lc'):
        Lc = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(label, FLAGS.class_num),
                                                                    logits=discrim_real_output_c) +
                            tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(label, FLAGS.class_num),
                                                                    logits=discrim_fake_output_c))

    with tf.variable_scope('discrim_loss'):
        discrim_loss = Ls + Lc

    with tf.variable_scope('adversial_loss'):
        adversial_loss = tf.reduce_mean(-tf.log(discrim_fake_output_s + FLAGS.EPS))
    ##改进一下loss
    with tf.variable_scope('gen_loss'):
        gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(label, FLAGS.class_num),
                                                                          logits=discrim_fake_output_c)) + adversial_loss

    # Define the learning rate and global step
    with tf.variable_scope('get_learning_rate_and_global_step', reuse=tf.AUTO_REUSE):
        global_step = tf.contrib.framework.get_or_create_global_step()
        learning_rate_dis = tf.train.exponential_decay(FLAGS.learning_rate_dis, global_step, FLAGS.decay_step,
                                                       FLAGS.decay_rate, staircase=FLAGS.stair)
        learning_rate_gen = tf.train.exponential_decay(FLAGS.learning_rate_gen, global_step, FLAGS.decay_step,
                                                       FLAGS.decay_rate, staircase=FLAGS.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('dicriminator_train', reuse=tf.AUTO_REUSE):
        discrim_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
        discrim_optimizer = tf.train.AdamOptimizer(learning_rate_dis, beta1=FLAGS.beta)
        discrim_grads_and_vars = discrim_optimizer.compute_gradients(discrim_loss, discrim_tvars)
        discrim_train = discrim_optimizer.apply_gradients(discrim_grads_and_vars)

    clip_discrim_train = [var.assign(tf.clip_by_value(var,-0.01,0.01)) for var in discrim_tvars]

    with tf.variable_scope('generator_train', reuse=tf.AUTO_REUSE):
        # Need to wait discriminator to perform train step
        with tf.control_dependencies([discrim_train] + tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            gen_tvars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='embedding') + tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
            gen_optimizer = tf.train.AdamOptimizer(learning_rate_gen, beta1=FLAGS.beta)
            gen_grads_and_vars = gen_optimizer.compute_gradients(gen_loss, gen_tvars)
            gen_train = gen_optimizer.apply_gradients(gen_grads_and_vars)

    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([discrim_loss, adversial_loss])

    return Network(
        discrim_real_output_c=discrim_real_output_c,
        discrim_fake_output_c=discrim_fake_output_c,
        discrim_loss=exp_averager.average(discrim_loss),
        gen_loss=gen_loss,
        discrim_grads_and_vars=discrim_grads_and_vars,
        # spectral_loss=spectral_loss,
        gen_grads_and_vars=gen_grads_and_vars,
        gen_output_spa=gen_output_spa,
        gen_output_spe=gen_output_spe,
        train_only_dis=tf.group(incr_global_step, update_loss),
        train=tf.group(incr_global_step, gen_train, update_loss),
        global_step=global_step,
        learning_rate_dis=learning_rate_dis,
        learning_rate_gen=learning_rate_gen,
        clip_discrim_train = clip_discrim_train
    )


# load training data
def load_data(FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for data loader')
        # Need to Modify the path of training data
    data = scio.loadmat(FLAGS.input_data_dir)
    label = data['train_label']
    spe = data['train_spe']
    spa = data['train_spa']

    label = label.astype(np.int64)
    spe = spe.astype(np.float32)
    spa = spa.astype(np.float32)

    label = tf.squeeze(label)
    spe = tf.squeeze(spe)
    spe = tf.expand_dims(spe, dim=2)

    output = tf.train.slice_input_producer([label, spe, spa], num_epochs=None, shuffle=True)
    input_label, input_spe, input_spa = tf.train.shuffle_batch(
        [output[0], output[1], output[2]], batch_size=FLAGS.batch_size, capacity=500,
        min_after_dequeue=300, num_threads=4)

    return input_label, input_spe, input_spa


def load_test(FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for data loader')
        # Need to Modify the path of training data
    data = scio.loadmat(FLAGS.input_data_dir)
    label = data['test_label']
    spe = data['test_spe']
    spa = data['test_spa']

    label = label.astype(np.int64)
    spe = spe.astype(np.float32)
    spa = spa.astype(np.float32)

    label = tf.squeeze(label)
    spe = tf.squeeze(spe)
    spe = tf.expand_dims(spe, dim=2)

    return label, spe, spa
