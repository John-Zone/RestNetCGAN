from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import os
from model import *
from ops import *
import math
import time
import numpy as np
import scipy
import scipy.io as scio
from random import randint
import time

# 定义各类参数
Flags = tf.app.flags
# 数据描述参数
Flags.DEFINE_integer('channel_num_spectral', 220, 'The number of images')
Flags.DEFINE_integer('channel_num_spatial', 3, 'The number of images')
Flags.DEFINE_integer('num_resblock', 8, 'The number of resblock')
# Pines 16;
Flags.DEFINE_integer('class_num', 16, 'the number of the class of the classification')
Flags.DEFINE_integer('noise_size', 100, 'the size of the input noise')

# The system parameter
Flags.DEFINE_string('output_dir', './0426/', 'The output directory of the checkpoint')
Flags.DEFINE_string('summary_dir', './0426/log', 'The dirctory to output the summary')
Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
# 实现断点训练时是否要从已训练得模型中提取参数
Flags.DEFINE_boolean('pre_trained_model', False,
                     'If set True, the weight will be loaded but the global_step will still '
                     'be 0. If set False, you are going to continue the training. That is, '
                     'the global_step will be initiallized from the checkpoint, too')

# 模型是否是在训练阶段
Flags.DEFINE_boolean('is_training', True, 'Training => True, Testing => False')
Flags.DEFINE_integer('batch_size', 20, 'Batch size of the input batch')
Flags.DEFINE_string('input_data_dir', './data/train.mat', 'The directory of the input resolution input data')
Flags.DEFINE_integer('name_queue_capacity', 2048, 'The capacity of the filename queue (suggest large to ensure'
                                                  'enough random shuffle.')
Flags.DEFINE_integer('image_queue_capacity', 2048, 'The capacity of the image queue (suggest large to ensure'
                                                   'enough random shuffle')
Flags.DEFINE_integer('queue_thread', 10, 'The threads of the queue (More threads can speedup the training process.')
# Generator configuration
# The content loss parameter
Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
# The training parameters
Flags.DEFINE_float('learning_rate_dis', 0.002, 'The learning rate for the discriminator')
Flags.DEFINE_float('learning_rate_gen', 0.01, 'The learning rate for the generator')
Flags.DEFINE_integer('decay_step', 10000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
# 学习率的阶梯下降
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 100000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 20, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 1000, 'The frequency of saving images')
Flags.DEFINE_integer('k_to_train_dis', 10, 'The frequency of train discriminator')
Flags.DEFINE_string('port', 'value', 'Nothing')

FLAGS = Flags.FLAGS

# Print the configuration of the model
print_configuration_op(FLAGS)

if FLAGS.mode == 'train':
    # 准备数据
    input_label, input_spe, input_spa = load_data(FLAGS)

    # 定义过程
    z1 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.noise_size], name='z1')
    z2 = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, FLAGS.noise_size], name='z2')

    with tf.variable_scope('embedding', reuse=False):
        embeding_matrix = tf.Variable(tf.random_normal([FLAGS.class_num, FLAGS.noise_size], stddev=0.35),
                                      trainable=True)
        onehot = tf.one_hot(input_label, FLAGS.class_num)
        label_embeding = tf.matmul(onehot, embeding_matrix)
        embedded_z1 = tf.multiply(z1, label_embeding)
        embedded_z2 = tf.multiply(z2, label_embeding)

    Net = model(embedded_z1, embedded_z2, input_spe, input_spa, input_label, FLAGS)
    # Add scalar summary
    tf.summary.scalar('discriminator_loss', Net.discrim_loss)
    # tf.summary.scalar('spectral_loss', Net.spectral_loss)
    tf.summary.scalar('learning_rate_dis', Net.learning_rate_dis)
    tf.summary.scalar('learning_rate_gen', Net.learning_rate_gen)
    tf.summary.scalar('gen_loss', Net.gen_loss)

    print('Finish building the network!!!')
    # Define the saver and weight initiallizer
    saver = tf.train.Saver(max_to_keep=10)

    # Start the session
    config = tf.ConfigProto(log_device_placement=True)
    config.gpu_options.allow_growth = True

    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    weight_initiallizer = tf.train.Saver(var_list)

    # Use superviser to coordinate all queue and summary writer
    sv = tf.train.Supervisor(logdir=FLAGS.summary_dir, save_summaries_secs=0, saver=None)
    with sv.managed_session(config=config) as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess, coord)

        # Here if we restore the weight from the _3DSRResnet the var_list2 do not need to contain the discriminator weights
        # On contrary, if you initial your weight from other _3DSRGAN checkpoint, var_list2 need to contain discriminator
        # weights.

        if FLAGS.pre_trained_model:
            if FLAGS.checkpoint == None:
                raise ValueError('No checkpoint given')
            else:
                weight_initiallizer.restore(sess, FLAGS.checkpoint)
                print('Loading the model from the checkpoint...')
        else:
            print('New Training!!!')

        # Performing the training
        if FLAGS.max_epoch is None:
            if FLAGS.max_iter is None:
                raise ValueError('one of max_epoch or max_iter should be provided')
            else:
                max_iter = FLAGS.max_iter

        print('Optimization starts!!!')

        start = time.time()

        for step in range(max_iter):

            fetches = {
                "train": Net.train,
                "global_step": sv.global_step}

            if ((step + 1) % FLAGS.display_freq) == 0:
                fetches["discrim_loss"] = Net.discrim_loss
                fetches["gen_loss"] = Net.gen_loss
                # fetches["spectral_loss"] = Net.spectral_loss
                fetches["learning_rate_dis"] = Net.learning_rate_dis
                fetches["learning_rate_gen"] = Net.learning_rate_gen
                fetches["global_step"] = Net.global_step

            if ((step + 1) % FLAGS.summary_freq) == 0:
                fetches["summary"] = sv.summary_op

            z1v = np.random.normal(size=(FLAGS.batch_size, FLAGS.noise_size)).astype(np.float32)
            z2v = np.random.normal(size=(FLAGS.batch_size, FLAGS.noise_size)).astype(np.float32)

            results = sess.run(fetches, feed_dict={z1: z1v, z2: z2v})
            # discriminator的截断参数
            sess.run(Net.clip_discrim_train)

            if ((step + 1) % FLAGS.summary_freq) == 0:
                print('Recording summary!!')
                sv.summary_writer.add_summary(results['summary'], results['global_step'])

            if ((step + 1) % FLAGS.display_freq) == 0:
                print("global_step", results["global_step"])
                print("gen_loss", results['gen_loss'])
                print("discrim_loss", results["discrim_loss"])
                # print("spectral_loss", results["spectral_loss"])
                print("learning_rate_dis", results['learning_rate_dis'])
                print("learning_rate_gen", results['learning_rate_gen'])

            if ((step + 1) % FLAGS.save_freq) == 0:
                print('Save the checkpoint')
                saver.save(sess, os.path.join(FLAGS.output_dir, 'model'), global_step=sv.global_step)

        print('Optimization done!!!!!!!!!!!!')

elif FLAGS.mode == 'test':
    # Check the checkpoint
    if FLAGS.checkpoint is None:
        raise ValueError('The checkpoint file is needed to performing the test.')

    input_label, input_spe, input_spa = load_test(FLAGS)
    FLAGS.is_training = False
    with tf.variable_scope('discriminator'):
        class_pro, real_pro = discriminator(input_spe, input_spa, FLAGS)

    print('Finish building the network')
    # Define the weight initiallizer (In inference time, we only need to restore the weight of the generator)
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')
    weight_initiallizer = tf.train.Saver(var_list)

    # Define the initialization operation
    init_op = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        # Load the pretrained model
        print('Loading weights from the pre-trained model')
        weight_initiallizer.restore(sess, FLAGS.checkpoint)
        class_pro = sess.run(class_pro)
        input_label = tf.one_hot(input_label, depth=FLAGS.class_num)
        correct_prediction = tf.equal(tf.argmax(class_pro, 1), tf.argmax(input_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", sess.run(accuracy))
