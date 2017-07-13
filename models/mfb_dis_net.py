
import tensorflow as tf

from tools.ops import *


FLAGS = tf.app.flags.FLAGS

class mfb_dis_net(object):

    def __init__(self, clips, labels, class_num=24, height=128, width=128, seq_length=16, c_dim=3, \
                 batch_size=32, keep_prob=1.0, is_training=True, encoder_gradient_ratio=1.0, use_pretrained_encoder=False):

        self.seq        = clips
        self.labels     = labels
        self.class_num  = class_num
        self.batch_size = batch_size
        self.height     = height
        self.width      = width
        self.seq_length = seq_length
        self.c_dim      = c_dim
        self.dropout    = keep_prob
        self.encoder_gradient_ratio = encoder_gradient_ratio
        self.use_pretrained_encoder = use_pretrained_encoder

        self.seq_shape  = [seq_length, height, width, c_dim]

        self.batch_norm_params = {
            'is_training': is_training,
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'center': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS
        }

        pred_logits  = self.build_model()
        self.ac_loss = tf.reduce_mean(\
                        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred_logits))

        prob = tf.nn.softmax(pred_logits)
        pred = tf.one_hot(tf.nn.top_k(prob).indices, self.class_num)
        pred = tf.squeeze(pred, axis=1)
        pred = tf.cast(pred, tf.bool)
        labels  = tf.cast(labels, tf.bool)
        self.ac = tf.reduce_sum(tf.cast(tf.logical_and(labels, pred), tf.float32)) / self.batch_size


    def build_model(self):

        c3d_feat = self.mapping_layer(self.c3d(self.seq))

        if self.use_pretrained_encoder and self.encoder_gradient_ratio == 0.0:
            c3d_feat = tf.stop_gradient(c3d_feat)

        with tf.variable_scope('classifier'):
            dense1 = tf.reshape(c3d_feat, [self.batch_size, -1])

            dense1 = fc(dense1, self.class_num, name='fc1')
            pred   = tf.nn.dropout(dense1, self.dropout)

        return pred


    def bn(self, x):
        return tf.contrib.layers.batch_norm(x, **self.batch_norm_params)


    def mapping_layer(self, input_, name='mapping'):
        with tf.variable_scope(name):
            feat = relu(self.bn(conv3d(input_, self.map_dim, k_t=self.map_length, k_h=2, k_w=2, d_h=2, d_w=2, padding='VALID', name='mapping1')))
            feat = tf.reshape(feat, [self.batch_size, self.map_height//2, self.map_width//2, self.map_dim])

        return feat


    def c3d(self, input_, _dropout=1.0, name='c3d'):

        with tf.variable_scope(name):

            # Convolution Layer
            conv1 = relu(self.bn(conv3d(input_, 64, name='conv1')))
            pool1 = max_pool3d(conv1, k=1, name='pool1')

            # Convolution Layer
            conv2 = relu(self.bn(conv3d(pool1, 128, name='conv2')))
            pool2 = max_pool3d(conv2, k=2, name='pool2')

            # Convolution Layer
            conv3 = relu(self.bn(conv3d(pool2, 256, name='conv3a')))
            conv3 = relu(self.bn(conv3d(conv3, 256, name='conv3b')))
            pool3 = max_pool3d(conv3, k=2, name='pool3')

            # Convolution Layer
            conv4 = relu(self.bn(conv3d(pool3, 512, name='conv4a')))
            conv4 = relu(self.bn(conv3d(conv4, 512, name='conv4b')))
            pool4 = max_pool3d(conv4, k=2, name='pool4')

            # Convolution Layer
            conv5 = relu(self.bn(conv3d(pool4, 512, name='conv5a')))
            conv5 = relu(self.bn(conv3d(conv5, 512, name='conv5b')))
            #pool5 = max_pool3d(conv5, k=2, name='pool5')

            conv5_shape     = conv5.get_shape().as_list()
            self.map_length = conv5_shape[1]
            self.map_height = conv5_shape[2]
            self.map_width  = conv5_shape[3]
            self.map_dim    = conv5_shape[4]

            feature = conv5

        return feature


def tower_loss(name_scope, mfb, use_pretrained_encoder, encoder_gradient_ratio=1.0):
    # get reconstruction and ground truth
    ac_loss = mfb.ac_loss

    weight_decay_loss_list = tf.get_collection('losses', name_scope)
    if use_pretrained_encoder:
        if encoder_gradient_ratio == 0.0:
            weight_decay_loss_list = [var for var in weight_decay_loss_list \
                                      if 'c3d' not in var.name and 'mapping' not in var.name]

    weight_decay_loss = 0.0
    if len(weight_decay_loss_list) > 0:
        weight_decay_loss = tf.add_n(weight_decay_loss_list)
        
    total_loss = weight_decay_loss * 100 + ac_loss

    return total_loss, ac_loss, weight_decay_loss