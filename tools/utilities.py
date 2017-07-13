import numpy as np
import tensorflow as tf
import scipy.misc as sm


FLAGS = tf.app.flags.FLAGS


def gen_pred_img(ffg, fbg, lfg):
    border = 2
    shape = ffg.shape # [h, w, c]
    image = np.ones([shape[0]+2*border, shape[1]*3+4*border, shape[2]]) * 255
    image[border:-border,border:shape[1]+border] = ffg
    image[border:-border,shape[1]+border*2:2*shape[1]+border*2] = fbg
    image[border:-border,2*shape[1]+3*border:-border] = lfg

    return image


def gen_pred_vid(vid):
    shape = vid.shape
    vid_img = np.zeros((shape[1], shape[0]*shape[2], shape[3]))

    for i in range(shape[0]):
        vid_img[:,i*shape[2]:(i+1)*shape[2]] = vid[i]

    return vid_img


def decode_frames(frame_list, h, w, l):
    clip = []
    for i in range(l):
        frame = frame_list[i]
        image = tf.cast(tf.image.decode_jpeg(frame), tf.float32)
        image.set_shape((h, w, 3))
        clip.append(image)

    return tf.stack(clip)


def generate_mask(img_mask_list, h, w, l):
    img_masks, loss_masks = [], []

    for i in range(l):
        # generate image mask
        img_mask = img_mask_list[i]
        img_mask = tf.cast(tf.image.decode_png(img_mask), tf.float32)
        img_mask = tf.reshape(img_mask, (h, w))
        img_masks.append(img_mask)

        # generate loss mask
        s_total   = h * w
        s_mask    = tf.reduce_sum(img_mask)
        def f1(): return img_mask*((s_total-s_mask)/s_mask-1)+1
        def f2(): return tf.zeros_like(img_mask)
        def f3(): return tf.ones_like(img_mask)
        loss_mask = tf.case([(tf.equal(s_mask, 0), f2), \
                             (tf.less(s_mask, s_total/2), f1)],
                             default=f3)

        loss_masks.append(loss_mask)

    return tf.stack(img_masks), tf.stack(loss_masks)
    

def read_my_file_format(filename_queue, is_training):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_features = {
        "height": tf.FixedLenFeature([], dtype=tf.int64),
        "width": tf.FixedLenFeature([], dtype=tf.int64),
        "sequence_length": tf.FixedLenFeature([], dtype=tf.int64),
        "text": tf.FixedLenFeature([], dtype=tf.string),
        "label": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "masks": tf.FixedLenSequenceFeature([], dtype=tf.string)
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    # start queue runner so it won't stuck
    tf.train.start_queue_runners(sess=tf.get_default_session())

    height = FLAGS.height
    width  = FLAGS.width
    sequence_length = 32

    clip = decode_frames(sequence_parsed['frames'], height, width, sequence_length)
    img_mask, loss_mask = generate_mask(sequence_parsed['masks'], \
                                        height, width, sequence_length)

    if is_training:
        # randomly sample clips of 16 frames
        idx = tf.squeeze(tf.random_uniform([1], 0, sequence_length-FLAGS.seq_length+1, dtype=tf.int32))
    else:
        # sample the middle clip
        idx = 8
    clip = clip[idx:idx+FLAGS.seq_length] / 255.0 * 2 - 1
    img_mask  = img_mask[idx:idx+FLAGS.seq_length]
    loss_mask = loss_mask[idx:idx+FLAGS.seq_length]

    if is_training:
        # randomly temporally flip data
        reverse   = tf.squeeze(tf.random_uniform([1], 0, 2, dtype=tf.int32))
        clip      = tf.cond(tf.equal(reverse,0), lambda: clip, lambda: clip[::-1])
        img_mask  = tf.cond(tf.equal(reverse,0), lambda: img_mask, lambda: img_mask[::-1])
        loss_mask = tf.cond(tf.equal(reverse,0), lambda: loss_mask, lambda: loss_mask[::-1])
        clip.set_shape([FLAGS.seq_length, height, width, 3])
        img_mask.set_shape([FLAGS.seq_length, height, width])
        loss_mask.set_shape([FLAGS.seq_length, height, width])

        # randomly horizontally flip data
        flip      = tf.squeeze(tf.random_uniform([1], 0, 2, dtype=tf.int32))
        img_list, img_mask_list, loss_mask_list  = tf.unstack(clip), tf.unstack(img_mask), tf.unstack(loss_mask)
        flip_clip, flip_img_mask, flip_loss_mask = [], [], []
        for i in range(FLAGS.seq_length):
            flip_clip.append(tf.cond(tf.equal(flip, 0), lambda: img_list[i], lambda: tf.image.flip_left_right(img_list[i])))
            flip_img_mask.append(tf.cond(tf.equal(flip, 0), lambda: img_mask_list[i], \
                                    lambda: tf.squeeze(tf.image.flip_left_right(tf.expand_dims(img_mask_list[i],-1)),-1)))
            flip_loss_mask.append(tf.cond(tf.equal(flip, 0), lambda: loss_mask_list[i], \
                                    lambda: tf.squeeze(tf.image.flip_left_right(tf.expand_dims(loss_mask_list[i],-1)),-1)))
        clip = tf.stack(flip_clip)
        img_mask = tf.stack(flip_img_mask)
        loss_mask = tf.stack(flip_loss_mask)

    clip.set_shape([FLAGS.seq_length, height, width, 3])
    img_mask.set_shape([FLAGS.seq_length, height, width])
    loss_mask.set_shape([FLAGS.seq_length, height, width])

    return clip, img_mask, loss_mask


def input_pipeline(filenames, batch_size, read_threads=4, num_epochs=None, is_training=True):
    filename_queue = tf.train.string_input_producer(
                        filenames, num_epochs=FLAGS.num_epochs, shuffle=is_training)
    # initialize local variables if num_epochs is not None or it'll raise uninitialized problem
    tf.get_default_session().run(tf.local_variables_initializer())

    example_list = [read_my_file_format(filename_queue, is_training) \
                        for _ in range(read_threads)]

    min_after_dequeue = 300 if is_training else 10
    capacity = min_after_dequeue + 3 * batch_size
    clip_batch, img_mask_batch, loss_mask_batch = tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return clip_batch, img_mask_batch, loss_mask_batch



def read_my_file_format_dis(filename_queue, is_training):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    context_features = {
        "height": tf.FixedLenFeature([], dtype=tf.int64),
        "width": tf.FixedLenFeature([], dtype=tf.int64),
        "sequence_length": tf.FixedLenFeature([], dtype=tf.int64),
        "text": tf.FixedLenFeature([], dtype=tf.string),
        "label": tf.FixedLenFeature([], dtype=tf.int64)
    }
    sequence_features = {
        "frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
        "masks": tf.FixedLenSequenceFeature([], dtype=tf.string)
    }
    context_parsed, sequence_parsed = tf.parse_single_sequence_example(
        serialized=serialized_example,
        context_features=context_features,
        sequence_features=sequence_features
    )

    height = 128#context_parsed['height'].eval()
    width  = 128#context_parsed['width'].eval()
    sequence_length = 32#context_parsed['sequence_length'].eval()

    clip  = decode_frames(sequence_parsed['frames'], height, width, sequence_length)

    # generate one hot vector
    label = context_parsed['label']
    label = tf.one_hot(label-1, FLAGS.num_class)
    text  = context_parsed['text']

    # randomly sample clips of 16 frames
    if is_training:
        idx = tf.squeeze(tf.random_uniform([1], 0, sequence_length-FLAGS.seq_length+1, dtype=tf.int32))
    else:
        idx = 8
    clip = clip[idx:idx+FLAGS.seq_length] / 255.0 * 2 - 1

    if is_training:
        # randomly reverse data
        reverse   = tf.squeeze(tf.random_uniform([1], 0, 2, dtype=tf.int32))
        clip      = tf.cond(tf.equal(reverse,0), lambda: clip, lambda: clip[::-1])

        # randomly horizontally flip data
        flip      = tf.squeeze(tf.random_uniform([1], 0, 2, dtype=tf.int32))
        clip      = tf.cond(tf.equal(flip,0), lambda: clip, lambda: \
                            tf.map_fn(lambda img: tf.image.flip_left_right(img), clip))

    clip.set_shape([FLAGS.seq_length, height, width, 3])

    return clip, label, text


def input_pipeline_dis(filenames, batch_size, read_threads=4, num_epochs=None, is_training=True):
    filename_queue = tf.train.string_input_producer(
                        filenames, num_epochs=FLAGS.num_epochs, shuffle=is_training)
    # initialize local variables if num_epochs is not None or it'll raise uninitialized problem
    tf.get_default_session().run(tf.local_variables_initializer())

    example_list = [read_my_file_format_dis(filename_queue, is_training) \
                        for _ in range(read_threads)]

    min_after_dequeue = 300 if is_training else 10
    capacity = min_after_dequeue + 3 * batch_size
    clip_batch, label_batch, text_batch = tf.train.shuffle_batch_join(
        example_list, batch_size=batch_size, capacity=capacity,
        min_after_dequeue=min_after_dequeue)

    return clip_batch, label_batch, text_batch


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, v in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def average_gradients_dis(tower_grads, encoder_gradient_ratio):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g, v in grad_and_vars:
            if 'c3d' in v.name or 'mapping' in v.name:
                g = g * encoder_gradient_ratio
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        if len(grads) == 0:
            continue
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads