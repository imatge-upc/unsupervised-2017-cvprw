import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from os import listdir

import sys
import time
import tools.ops

import numpy as np

import tensorflow as tf
import scipy.misc as sm

from models.mfb_dis_net import *
from tools.utilities import *
from tools.ops import *


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs.')	# ~13 min per epoch
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs.')
flags.DEFINE_integer('seq_length', 16, 'Length of each video clip.')
flags.DEFINE_integer('height', 128, 'Height of video frame.')
flags.DEFINE_integer('width', 128, 'Width of video frame.')
flags.DEFINE_integer('channel', 3, 'Number of channels for each frame.')
flags.DEFINE_integer('num_sample', 4451, 'Number of samples in this dataset.')
flags.DEFINE_integer('num_class', 24, 'Number of classes to classify.')

FLAGS = flags.FLAGS

prefix          = 'DIS_ae_mfb_baseline_finetune=1.0'
model_save_dir  = './ckpt/' + prefix
loss_save_dir   = './loss'
test_list_path  = './dataset/testlist.txt'
dataset_path    = './dataset/UCF-101-tf-records'
test_model_name = model_save_dir + '/mfb_baseline_ucf24.model-7000'

use_pretrained_model = True
save_predictions     = True


def run_testing():

	# Create model directory
	if not os.path.exists(model_save_dir):
		os.makedirs(model_save_dir)
	model_filename = "./mfb_dis_ucf24.model"

	tower_grads, tower_ac  = [], []
	tower_losses, tower_ac_losses, tower_wd_losses = [], [], []

	global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                trainable=False
                )
	starter_learning_rate = 1e-4
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           1000000, 0.8, staircase=True)
	#opt = tf.train.AdamOptimizer(learning_rate)

	# Create a session for running Ops on the Graph.
	config = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=config)
	coord = tf.train.Coordinator()
	threads = None

	test_list_file = open(test_list_path, 'r')
	test_list = test_list_file.read().splitlines()
	for i, line in enumerate(test_list):
		test_list[i] = os.path.join(dataset_path, test_list[i])

	assert(len(test_list) % FLAGS.num_gpus == 0)
	num_for_each_gpu = len(test_list) // FLAGS.num_gpus

	clips_list, labels_list, texts_list = [], [], []
	with sess.as_default():
		for i in range(FLAGS.num_gpus):
			clips, labels, texts = input_pipeline_dis(test_list[i*num_for_each_gpu:(i+1)*num_for_each_gpu], \
												  FLAGS.batch_size, num_epochs=FLAGS.num_epochs, is_training=False)
			clips_list.append(clips)
			labels_list.append(labels)
			texts_list.append(texts)

	mfb_list = []
	with tf.variable_scope('vars') as var_scope:
		for gpu_index in range(FLAGS.num_gpus):
			with tf.device('/gpu:%d' % (gpu_index)):
				with tf.name_scope('%s_%d' % ('tower', gpu_index)) as scope:
				
					# construct model
					mfb = mfb_dis_net(clips_list[gpu_index], labels_list[gpu_index], FLAGS.num_class, FLAGS.height, \
									  FLAGS.width, FLAGS.seq_length, FLAGS.channel, FLAGS.batch_size, is_training=False)
					mfb_list.append(mfb)
					loss, ac_loss, wd_loss = tower_loss(scope, mfb, use_pretrained_model)

					var_scope.reuse_variables()

					#tower_grads.append(grads)
					tower_losses.append(loss)
					tower_ac_losses.append(ac_loss)
					tower_wd_losses.append(wd_loss)
					tower_ac.append(mfb.ac)


	# concatenate the losses of all towers
	loss_op    = tf.reduce_mean(tower_losses)
	ac_loss_op = tf.reduce_mean(tower_ac_losses)
	wd_loss_op = tf.reduce_mean(tower_wd_losses)
	ac_op      = tf.reduce_mean(tower_ac)

	tf.summary.scalar('loss', loss_op)
	tf.summary.scalar('ac_loss', ac_loss_op)
	tf.summary.scalar('ac', ac_op)
	tf.summary.scalar('wd_loss', wd_loss_op)

	# saver for saving checkpoints
	saver = tf.train.Saver()
	init = tf.initialize_all_variables()

	sess.run(init)
	if not os.path.exists(model_save_dir):
		os.makedirs(model_save_dir)
	if use_pretrained_model:
		print('[*] Loading checkpoint ...')
		saver.restore(sess, test_model_name)

	# Create loss output folder
	if not os.path.exists(loss_save_dir):
		os.makedirs(loss_save_dir)
	loss_file = open(os.path.join(loss_save_dir, prefix+'_test.txt'), 'a+')

	loss_file.write('\n' + test_model_name + '\n')
	loss_file.flush()

	total_steps = (FLAGS.num_sample / (FLAGS.num_gpus * FLAGS.batch_size)) * FLAGS.num_epochs

	# start queue runner
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	ac_list, loss_list = [], []
	step = 0
	try:
		with sess.as_default():
			print('\n\n\n*********** start testing ***********\n\n\n')
			step = global_step.eval()
			print('[step = %d]'%step)
			while not coord.should_stop():
				# Run training steps or whatever
				ac, ac_loss = sess.run([ac_op, ac_loss_op])
				ac_list.append(ac)
				loss_list.append(ac_loss)
				line = 'ac=%.3f, loss=%.8f' %(ac*100, ac_loss)
				loss_file.write(line + '\n')
				loss_file.flush()
				print(line)


	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

	# Wait for threads to finish.
	coord.join(threads)
	sess.close()

	mean_ac   = np.mean(np.asarray(ac_list))
	mean_loss = np.mean(np.asarray(loss_list))

	line = '[step=%d] mean_ac=%.3f, mean_loss=%.8f' %(step, mean_ac*100, mean_loss)
	print(line)
	loss_file.write(line + '\n')



def main(_):
	run_testing()


if __name__ == '__main__':
	tf.app.run()
