import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from os import listdir

import sys
import time
import argparse
import tools.ops

import numpy as np

import tensorflow as tf
import scipy.misc as sm

from models.mfb_net_cross import *
from tools.utilities import *
from tools.ops import *

parser = argparse.ArgumentParser()
parser.add_argument('-lr', dest='lr', type=float, default='1e-4', help='original learning rate')
args = parser.parse_args()

flags = tf.app.flags
flags.DEFINE_float('lr', args.lr, 'Original learning rate.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs.')	# ~13 min per epoch
flags.DEFINE_integer('num_gpus', 4, 'Number of GPUs.')
flags.DEFINE_integer('seq_length', 16, 'Length of each video clip.')
flags.DEFINE_integer('height', 128, 'Height of video frame.')
flags.DEFINE_integer('width', 128, 'Width of video frame.')
flags.DEFINE_integer('channel', 3, 'Number of channels for each frame.')
flags.DEFINE_integer('num_sample', 1240, 'Number of samples in this dataset.')
flags.DEFINE_float('wd', 0.001, 'Weight decay rate.')

FLAGS = flags.FLAGS

prefix          = 'mfb_cross'
model_save_dir  = './ckpt/' + prefix
loss_save_dir   = './loss'
val_list_path   = './dataset/vallist.txt'
dataset_path    = './dataset/UCF-101-tf-records'

use_pretrained_model = True
save_predictions     = True


def run_validation():

	# Create model directory
	if not os.path.exists(model_save_dir):
		os.makedirs(model_save_dir)
	model_filename = "./mfb_baseline_ucf24.model"

	tower_ffg_losses, tower_fbg_losses, tower_lfg_losses, tower_feat_losses = [], [], [], []
	tower_ffg_m_losses, tower_fbg_m_losses, tower_lfg_m_losses = [], [], []

	global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                trainable=False
                )
	starter_learning_rate = 1e-4
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000000, 0.5, staircase=True)
	opt = tf.train.AdamOptimizer(learning_rate)

	# Create a session for running Ops on the Graph.
	config = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=config)
	coord = tf.train.Coordinator()
	threads = None

	val_list_file = open(val_list_path, 'r')
	val_list = val_list_file.read().splitlines()
	for i, line in enumerate(val_list):
		val_list[i] = os.path.join(dataset_path, val_list[i])

	assert(len(val_list) % FLAGS.num_gpus == 0)
	num_for_each_gpu = len(val_list) // FLAGS.num_gpus

	clips_list, img_masks_list, loss_masks_list = [], [], []
	with sess.as_default():
		for i in range(FLAGS.num_gpus):
			clips, img_masks, loss_masks = input_pipeline(val_list[i*num_for_each_gpu:(i+1)*num_for_each_gpu], \
											FLAGS.batch_size, read_threads=1, num_epochs=FLAGS.num_epochs, is_training=False)
			clips_list.append(clips)
			img_masks_list.append(img_masks)
			loss_masks_list.append(loss_masks)

	mfb_list = []
	with tf.variable_scope('vars') as var_scope:
		for gpu_index in range(FLAGS.num_gpus):
			with tf.device('/gpu:%d' % (gpu_index)):
				with tf.name_scope('%s_%d' % ('tower', gpu_index)) as scope:
				
					# construct model
					mfb = mfb_net(clips_list[gpu_index], FLAGS.height, FLAGS.width, FLAGS.seq_length, \
									FLAGS.channel, FLAGS.batch_size, is_training=False)
					mfb_list.append(mfb)
					_, first_fg_loss, first_bg_loss, last_fg_loss, feat_loss, _ = \
						tower_loss(scope, mfb, clips_list[gpu_index], img_masks_list[gpu_index], loss_masks_list[gpu_index])

					var_scope.reuse_variables()

					tower_ffg_losses.append(first_fg_loss)
					tower_fbg_losses.append(first_bg_loss)
					tower_lfg_losses.append(last_fg_loss)
					tower_feat_losses.append(feat_loss)


	# concatenate the losses of all towers
	ffg_loss_op  = tf.reduce_mean(tower_ffg_losses)
	fbg_loss_op  = tf.reduce_mean(tower_fbg_losses)
	lfg_loss_op  = tf.reduce_mean(tower_lfg_losses)
	feat_loss_op = tf.reduce_mean(tower_feat_losses)

	# saver for saving checkpoints
	saver = tf.train.Saver()
	init = tf.initialize_all_variables()

	sess.run(init)
	if not os.path.exists(model_save_dir):
		os.makedirs(model_save_dir)
	if use_pretrained_model:
		print('[*] Loading checkpoint ...')
		model = tf.train.latest_checkpoint(model_save_dir)
		if model is not None:
			saver.restore(sess, model)
			print('[*] Loading success: %s!'%model)
		else:
			print('[*] Loading failed ...')

	# Create loss output folder
	if not os.path.exists(loss_save_dir):
		os.makedirs(loss_save_dir)
	loss_file = open(os.path.join(loss_save_dir, prefix+'_val.txt'), 'a+')

	total_steps = (FLAGS.num_sample / (FLAGS.num_gpus * FLAGS.batch_size)) * FLAGS.num_epochs

	# start queue runner
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	ffg_loss_list, fbg_loss_list, lfg_loss_list, feat_loss_list = [], [], [], []
	try:
		with sess.as_default():
			print('\n\n\n*********** start validating ***********\n\n\n')
			step = global_step.eval()
			print('[step = %d]'%step)
			while not coord.should_stop():
				# Run inference steps
				ffg_loss, fbg_loss, lfg_loss, feat_loss = \
						sess.run([ffg_loss_op, fbg_loss_op, lfg_loss_op, feat_loss_op])
				ffg_loss_list.append(ffg_loss)
				fbg_loss_list.append(fbg_loss)
				lfg_loss_list.append(lfg_loss)
				feat_loss_list.append(feat_loss)
				print('ffg_loss=%.8f, fbg_loss=%.8f, lfg_loss=%.8f, feat_loss=%.8f' \
						%(ffg_loss, fbg_loss, lfg_loss, feat_loss))

	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

	# Wait for threads to finish.
	coord.join(threads)
	sess.close()

	mean_ffg = np.mean(np.asarray(ffg_loss_list))
	mean_fbg = np.mean(np.asarray(fbg_loss_list))
	mean_lfg = np.mean(np.asarray(lfg_loss_list))
	mean_feat = np.mean(np.asarray(feat_loss_list))

	line = '[step=%d] ffg_loss=%.8f, fbg_loss=%.8f, lfg_loss=%.8f, feat_loss=%.8f' \
			%(step, mean_ffg, mean_fbg, mean_lfg, mean_feat)
	print(line)
	loss_file.write(line + '\n')



def main(_):
	run_validation()


if __name__ == '__main__':
	tf.app.run()