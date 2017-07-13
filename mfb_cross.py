import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from os import listdir

import sys
import time
import argparse
import tools.ops
import subprocess

import numpy as np
import tensorflow as tf
import scipy.misc as sm

from models.mfb_net_cross import *
from tools.utilities import *
from tools.ops import *

parser = argparse.ArgumentParser()
parser.add_argument('-lr', dest='lr', type=float, default='1e-4', help='original learning rate')
parser.add_argument('-batch_size', dest='batch_size', type=int, default='10', help='batch_size')
args = parser.parse_args()

flags = tf.app.flags
flags.DEFINE_float('lr', args.lr, 'Original learning rate.')
flags.DEFINE_integer('batch_size', args.batch_size, 'Batch size.')
flags.DEFINE_integer('num_epochs', 500, 'Number of epochs.')
flags.DEFINE_integer('num_gpus', 4, 'Number of GPUs.')
flags.DEFINE_integer('seq_length', 16, 'Length of each video clip.')
flags.DEFINE_integer('height', 128, 'Height of video frame.')
flags.DEFINE_integer('width', 128, 'Width of video frame.')
flags.DEFINE_integer('channel', 3, 'Number of channels for each frame.')
flags.DEFINE_integer('num_sample', 10060, 'Number of samples in this dataset.')
flags.DEFINE_float('wd', 0.001, 'Weight decay rate.')

FLAGS = flags.FLAGS

prefix          = 'mfb_cross'
model_save_dir  = './ckpt/' + prefix
logs_save_dir   = './logs/' + prefix
pred_save_dir   = './output/' + prefix
loss_save_dir   = './loss'
train_list_path = './dataset/trainlist.txt'
dataset_path    = './dataset/UCF-101-tf-records'
evaluation_job  = './jobs/mfb_cross_val'

use_pretrained_model = True
save_predictions     = True


def run_training():

	# Create model directory
	if not os.path.exists(model_save_dir):
		os.makedirs(model_save_dir)
	model_filename = "./mfb_baseline_ucf24.model"

	# Consturct computational graph
	tower_grads  = []
	tower_losses, tower_ffg_losses, tower_fbg_losses, tower_lfg_losses, tower_feat_losses, tower_wd_losses = [], [], [], [], [], []
	tower_ffg_m_losses, tower_fbg_m_losses, tower_lfg_m_losses = [], [], []

	global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                trainable=False
                )
	starter_learning_rate = FLAGS.lr
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.5, staircase=True)
	opt = tf.train.AdamOptimizer(learning_rate)

	# Create a session for running Ops on the Graph.
	config = tf.ConfigProto(allow_soft_placement=True)
	sess = tf.Session(config=config)
	coord = tf.train.Coordinator()
	threads = None

	train_list_file = open(train_list_path, 'r')
	train_list = train_list_file.read().splitlines()
	for i, line in enumerate(train_list):
		train_list[i] = os.path.join(dataset_path, train_list[i])

	assert(len(train_list) % FLAGS.num_gpus == 0)
	num_for_each_gpu = len(train_list) // FLAGS.num_gpus		

	clips_list, img_masks_list, loss_masks_list = [], [], []
	with sess.as_default():
		for i in range(FLAGS.num_gpus):
			clips, img_masks, loss_masks = input_pipeline(train_list[i*num_for_each_gpu:(i+1)*num_for_each_gpu], \
														  FLAGS.batch_size, read_threads=4, num_epochs=FLAGS.num_epochs, is_training=True)
			clips_list.append(clips)
			img_masks_list.append(img_masks)
			loss_masks_list.append(loss_masks)

	mfb_list = []
	with tf.variable_scope('vars') as var_scope:
		for gpu_index in range(FLAGS.num_gpus):
			with tf.device('/gpu:%d' % (gpu_index)):
				with tf.name_scope('%s_%d' % ('tower', gpu_index)) as scope:
				
					# construct model
					mfb = mfb_net(clips_list[gpu_index], FLAGS.height, FLAGS.width, FLAGS.seq_length, FLAGS.channel, FLAGS.batch_size)
					mfb_list.append(mfb)
					loss, first_fg_loss, first_bg_loss, last_fg_loss, feat_loss, wd_loss = \
						tower_loss(scope, mfb, clips_list[gpu_index], img_masks_list[gpu_index], loss_masks_list[gpu_index])

					var_scope.reuse_variables()

					vars_to_optimize = tf.trainable_variables()
					grads = opt.compute_gradients(loss, var_list=vars_to_optimize)

					tower_grads.append(grads)
					tower_losses.append(loss)
					tower_ffg_losses.append(first_fg_loss)
					tower_fbg_losses.append(first_bg_loss)
					tower_lfg_losses.append(last_fg_loss)
					tower_feat_losses.append(feat_loss)
					tower_wd_losses.append(wd_loss)


	# concatenate the losses of all towers
	loss_op      = tf.reduce_mean(tower_losses)
	ffg_loss_op  = tf.reduce_mean(tower_ffg_losses)
	fbg_loss_op  = tf.reduce_mean(tower_fbg_losses)
	lfg_loss_op  = tf.reduce_mean(tower_lfg_losses)
	feat_loss_op = tf.reduce_mean(tower_feat_losses)
	wd_loss_op   = tf.reduce_mean(tower_wd_losses)

	tf.summary.scalar('loss', loss_op)
	tf.summary.scalar('ffg_loss', ffg_loss_op)
	tf.summary.scalar('fbg_loss', fbg_loss_op)
	tf.summary.scalar('lfg_loss', lfg_loss_op)
	tf.summary.scalar('feat_loss', feat_loss_op)
	tf.summary.scalar('wd_loss', wd_loss_op)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	grads = average_gradients(tower_grads)
	with tf.control_dependencies(update_ops):
		train_op = opt.apply_gradients(grads, global_step=global_step)

	# saver for saving checkpoints
	saver = tf.train.Saver(max_to_keep=10)
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

	# Create summary writer
	merged = tf.summary.merge_all()
	if not os.path.exists(logs_save_dir):
		os.makedirs(logs_save_dir)
	sum_writer = tf.summary.FileWriter(logs_save_dir, sess.graph)

	# Create prediction output folder
	if not os.path.exists(pred_save_dir):
		os.makedirs(pred_save_dir)

	# Create loss output folder
	if not os.path.exists(loss_save_dir):
		os.makedirs(loss_save_dir)
	loss_file = open(os.path.join(loss_save_dir, prefix+'.txt'), 'w')

	total_steps = (FLAGS.num_sample / (FLAGS.num_gpus * FLAGS.batch_size)) * FLAGS.num_epochs

	# start queue runner
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	try:
		with sess.as_default():
			print('\n\n\n*********** start training ***********\n\n\n')
			while not coord.should_stop():
				# Run training steps
				start_time = time.time()
				sess.run(train_op)
				duration = time.time() - start_time
				step = global_step.eval()

				if step == 1 or step % 10 == 0: # evaluate loss
					loss, ffg_loss, fbg_loss, lfg_loss, feat_loss, wd_loss, lr = \
						sess.run([loss_op, ffg_loss_op, fbg_loss_op, lfg_loss_op, feat_loss_op, wd_loss_op, learning_rate])
					line = 'step %d/%d, loss=%.8f, ffg=%.8f, fbg=%.8f, lfg=%.8f, feat=%.8f, lwd=%.8f, dur=%.3f, lr=%.8f' \
							%(step, total_steps, loss, ffg_loss, fbg_loss, lfg_loss, feat_loss, wd_loss, duration, lr)
					print(line)
					loss_file.write(line + '\n')
					loss_file.flush()

				if step == 1 or step % 10 == 0: # save summary
					summary = summary_str = sess.run(merged)
					sum_writer.add_summary(summary, step)

				if step % 100 == 0 and save_predictions: # save current predictions
					mfb = mfb_list[0] # only visualize prediction in first tower
					ffg, fbg, lfg, gt_ffg, gt_fbg, gt_lfg = sess.run([
						mfb.first_fg_rec[0], mfb.first_bg_rec[0], mfb.last_fg_rec[0], \
						mfb.gt_ffg[0], mfb.gt_fbg[0], mfb.gt_lfg[0]])
					ffg, fbg, lfg = (ffg+1)/2*255.0, (fbg+1)/2*255.0, (lfg+1)/2*255.0
					gt_ffg, gt_fbg, gt_lfg = (gt_ffg+1)/2*255.0, (gt_fbg+1)/2*255.0, (gt_lfg+1)/2*255.0
					img = gen_pred_img(ffg, fbg, lfg)
					gt  = gen_pred_img(gt_ffg, gt_fbg, gt_lfg)
					save_img = np.concatenate((img, gt))
					sm.imsave(os.path.join(pred_save_dir, '%07d.jpg'%step), save_img)

				if step % 500 == 0: # save checkpoint
					saver.save(sess, os.path.join(model_save_dir, model_filename), global_step=global_step)

				if step % 500 == 0:
					pass
					# launch a new script for validation (please modify it for your own script)
					#subprocess.check_output(['python', evaluation_job])

	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

	# Wait for threads to finish.
	coord.join(threads)
	sess.close()



def main(_):
	run_training()


if __name__ == '__main__':
	tf.app.run()