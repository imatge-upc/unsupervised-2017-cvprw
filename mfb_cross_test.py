import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from os import listdir

import sys
import time
import tools.ops

import numpy as np

import tensorflow as tf
import scipy.misc as sm

from models.mfb_net_cross import *
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

FLAGS = flags.FLAGS

prefix          = 'mfb_cross'
model_save_dir  = './ckpt/' + prefix
loss_save_dir   = './loss'
test_save_dir   = './test/' + prefix
test_list_path  = './dataset/testlist.txt'
dataset_path    = './dataset/UCF-101-tf-records'

use_pretrained_model = True
save_predictions     = True


def run_testing():

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

	test_list_file = open(test_list_path, 'r')
	test_list = test_list_file.read().splitlines()
	for i, line in enumerate(test_list):
		test_list[i] = os.path.join(dataset_path, test_list[i])

	assert(len(test_list) % FLAGS.num_gpus == 0)
	num_for_each_gpu = len(test_list) // FLAGS.num_gpus

	clips_list, img_masks_list, loss_masks_list = [], [], []
	with sess.as_default():
		for i in range(FLAGS.num_gpus):
			clips, img_masks, loss_masks = input_pipeline(test_list[i*num_for_each_gpu:(i+1)*num_for_each_gpu], \
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
	loss_file = open(os.path.join(loss_save_dir, prefix+'_test.txt'), 'a+')

	# Create test output folder
	if not os.path.exists(test_save_dir):
		os.makedirs(test_save_dir)

	# start queue runner
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	ffg_loss_list, fbg_loss_list, lfg_loss_list, feat_loss_list = [], [], [], []
	try:
		with sess.as_default():
			print('\n\n\n*********** start testing ***********\n\n\n')
			step = global_step.eval()
			print('[step = %d]'%step)
			cnt = 0
			while not coord.should_stop():
				# Run training steps or whatever
				mfb = mfb_list[0]
				ffg, fbg, lfg, gt_ffg, gt_fbg, gt_lfg, ffg_loss, fbg_loss, lfg_loss, feat_loss = sess.run([
					mfb.first_fg_rec, mfb.first_bg_rec, mfb.last_fg_rec, \
					mfb.gt_ffg, mfb.gt_fbg, mfb.gt_lfg, ffg_loss_op, fbg_loss_op, lfg_loss_op, feat_loss_op])

				ffg, fbg, lfg, gt_ffg, gt_fbg, gt_lfg = \
					ffg[0], fbg[0], lfg[0], gt_ffg[0], gt_fbg[0], gt_lfg[0]

				ffg, fbg, lfg = (ffg+1)/2*255.0, (fbg+1)/2*255.0, (lfg+1)/2*255.0
				gt_ffg, gt_fbg, gt_lfg = (gt_ffg+1)/2*255.0, (gt_fbg+1)/2*255.0, (gt_lfg+1)/2*255.0

				img = gen_pred_img(ffg, fbg, lfg)
				gt  = gen_pred_img(gt_ffg, gt_fbg, gt_lfg)
				save_img = np.concatenate((img, gt))
				sm.imsave(os.path.join(test_save_dir, '%05d.jpg'%cnt), save_img)

				ffg_loss_list.append(ffg_loss)
				fbg_loss_list.append(fbg_loss)
				lfg_loss_list.append(lfg_loss)
				feat_loss_list.append(feat_loss)

				line = '%05d: ffg_loss=%.8f, fbg_loss=%.8f, lfg_loss=%.8f, feat_loss=%.8f' \
						%(cnt, ffg_loss, fbg_loss, lfg_loss, feat_loss)
				loss_file.write(line + '\n')
				loss_file.flush()
				print(line)
				cnt += 1

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
	run_testing()


if __name__ == '__main__':
	tf.app.run()