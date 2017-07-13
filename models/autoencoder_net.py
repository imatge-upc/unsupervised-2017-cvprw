
import tensorflow as tf

from tools.ops import *

class autoencoder_net(object):

    def __init__(self, input_, height=128, width=128, seq_length=16, c_dim=3, batch_size=32, is_training=True):

        self.seq        = input_
        self.batch_size = batch_size
        self.height     = height
        self.width      = width
        self.seq_length = seq_length
        self.c_dim      = c_dim

        self.seq_shape  = [seq_length, height, width, c_dim]

        self.batch_norm_params = {
            'is_training': is_training,
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'center': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS
        }

        self.build_model()


    def build_model(self):

        c3d_feat = self.mapping_layer(self.c3d(self.seq))

        self.rec_vid = self.decoder(c3d_feat)


    def bn(self, x):
        return tf.contrib.layers.batch_norm(x, **self.batch_norm_params)


    def mapping_layer(self, input_, name='mapping'):
        with tf.variable_scope(name):
            feat = relu(self.bn(conv3d(input_, self.map_dim, k_t=self.map_length, k_h=2, k_w=2, d_h=2, d_w=2, padding='VALID', name='mapping1')))

        return feat


    def decoder(self, input_, name='decoder'):
        # mirror decoder of c3d
        with tf.variable_scope(name):

            deconv6a = relu(self.bn(deconv3d(input_,
                            output_shape=[self.batch_size,self.map_length,self.map_height,self.map_width,self.map_dim],
                                k_t=2,k_h=2,k_w=2,d_t=2,d_h=2,d_w=2,padding='SAME',name='deconv6a')))

            deconv5b = relu(self.bn(deconv3d(deconv6a,
                            output_shape=[self.batch_size,self.map_length,self.map_height,self.map_width,self.map_dim],
                                k_t=3,k_h=3,k_w=3,d_t=1,d_h=1,d_w=1,padding='SAME',name='deconv5b')))
            deconv5a = relu(self.bn(deconv3d(deconv5b,
                            output_shape=[self.batch_size,self.map_length,self.map_height,self.map_width,self.map_dim],
                                k_t=3,k_h=3,k_w=3,d_t=1,d_h=1,d_w=1,padding='SAME',name='deconv5a')))
            unpool4  = relu(self.bn(deconv3d(deconv5a,
                            output_shape=[self.batch_size,self.map_length*2,self.map_height*2,self.map_width*2,self.map_dim],
                                k_t=3,k_h=3,k_w=3,d_t=2,d_h=2,d_w=2,padding='SAME',name='unpool4')))                                                                       

            deconv4b = relu(self.bn(deconv3d(unpool4,
                            output_shape=[self.batch_size,self.map_length*2,self.map_height*2,self.map_width*2,self.map_dim],
                                k_t=3,k_h=3,k_w=3,d_t=1,d_h=1,d_w=1,padding='SAME',name='deconv4b')))
            deconv4a = relu(self.bn(deconv3d(deconv4b,
                            output_shape=[self.batch_size,self.map_length*2,self.map_height*2,self.map_width*2,self.map_dim//2],
                                k_t=3,k_h=3,k_w=3,d_t=1,d_h=1,d_w=1,padding='SAME',name='deconv4a')))
            unpool3  = relu(self.bn(deconv3d(deconv4a,
                            output_shape=[self.batch_size,self.map_length*4,self.map_height*4,self.map_width*4,self.map_dim//2],
                                k_t=3,k_h=3,k_w=3,d_t=2,d_h=2,d_w=2,padding='SAME',name='unpool3')))

            deconv3b = relu(self.bn(deconv3d(unpool3,
                            output_shape=[self.batch_size,self.map_length*4,self.map_height*4,self.map_width*4,self.map_dim//2],
                                k_t=3,k_h=3,k_w=3,d_t=1,d_h=1,d_w=1,padding='SAME',name='deconv3b')))
            deconv3a = relu(self.bn(deconv3d(deconv3b,
                            output_shape=[self.batch_size,self.map_length*4,self.map_height*4,self.map_width*4,self.map_dim//4],
                                k_t=3,k_h=3,k_w=3,d_t=1,d_h=1,d_w=1,padding='SAME',name='deconv3a')))
            unpool2  = relu(self.bn(deconv3d(deconv3a,
                            output_shape=[self.batch_size,self.map_length*8,self.map_height*8,self.map_width*8,self.map_dim//4],
                                k_t=3,k_h=3,k_w=3,d_t=2,d_h=2,d_w=2,padding='SAME',name='unpool2')))    

            deconv2a = relu(self.bn(deconv3d(unpool2,
                            output_shape=[self.batch_size,self.map_length*8,self.map_height*8,self.map_width*8,self.map_dim//8],
                                k_t=3,k_h=3,k_w=3,d_t=1,d_h=1,d_w=1,padding='SAME',name='deconv2a')))         
            unpool1  = relu(self.bn(deconv3d(deconv2a,
                            output_shape=[self.batch_size,self.map_length*8,self.map_height*16,self.map_width*16,self.map_dim//8],
                                k_t=3,k_h=3,k_w=3,d_t=1,d_h=2,d_w=2,padding='SAME',name='unpool1')))

            deconv1a = deconv3d(unpool1,
                            output_shape=[self.batch_size,self.map_length*8,self.map_height*16,self.map_width*16,self.c_dim],
                                k_t=3,k_h=3,k_w=3,d_t=1,d_h=1,d_w=1,padding='SAME',name='deconv1a')

            vid = tf.tanh(deconv1a)
            
        return vid


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

            conv5_shape     = conv5.get_shape().as_list()
            self.map_length = conv5_shape[1]
            self.map_height = conv5_shape[2]
            self.map_width  = conv5_shape[3]
            self.map_dim    = conv5_shape[4]

            feature = conv5

        return feature


def tower_loss(name_scope, autoencoder, clips):
    # calculate reconstruction loss
    rec_loss = tf.reduce_mean(tf.abs(clips-autoencoder.rec_vid))

    weight_decay_loss_list = tf.get_collection('losses', name_scope)
    weight_decay_loss = 0.0
    if len(weight_decay_loss_list) > 0:
        weight_decay_loss = tf.add_n(weight_decay_loss_list)

    tf.add_to_collection('losses', rec_loss)
    losses = tf.get_collection('losses', name_scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='total_loss')

    return total_loss, rec_loss, weight_decay_loss