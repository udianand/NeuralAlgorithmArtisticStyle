import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize, imsave

#Function to define Weight variable
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#Function to define Bias variable
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#Function to perform convolution
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1, 1, 1, 1], padding='SAME')
#Pooling Function
def max_pool(x):
    return tf.nn.max_pool(x,strides=[1, 2, 2, 1], ksize=[1, 2, 2, 1] , padding='SAME')
#function for convolution layer
def conv_layer(input,shape):
    W=weight_variable(shape)
    b=bias_variable([shape[3]])
    return tf.nn.relu(tf.nn.conv2d(x,W,strides=[1, 1, 1, 1], padding='SAME')+b)


class vgg16:

    def convlayers(self):
        self.parameters = []

        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='img_mean')
            images = self.imgs_update-mean

        # conv1_1
        with tf.name_scope('conv1_1') as scope:
            shape1=[3, 3, 3, 64]
            W = weight_variable(shape1)
            b = bias_variable([shape1[3]])
            conv = tf.nn.relu(tf.nn.conv2d(images, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv1_1 = conv
            self.parameters += [W, b]

        # conv1_2
        with tf.name_scope('conv1_2') as scope:
            shape2=[3, 3, 64, 64]
            W = weight_variable(shape2)
            b = bias_variable([shape2[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv1_1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv1_2 = conv
            self.parameters += [W, b]

        # pool1
        self.pool1 = max_pool(self.conv1_2)

        # conv2_1
        with tf.name_scope('conv2_1') as scope:
            shape3=[3, 3, 64, 128]
            W = weight_variable(shape3)
            b = bias_variable([shape3[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.pool1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv2_1 = conv
            self.parameters += [W, b]

        # conv2_2
        with tf.name_scope('conv2_2') as scope:
            shape4=[3, 3, 128, 128]
            W = weight_variable(shape4)
            b = bias_variable([shape4[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv2_1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv2_2 =conv
            self.parameters += [W, b]

        # pool2
        self.pool2 = max_pool(self.conv2_2)
        
        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            shape5=[3, 3, 128, 256]
            W = weight_variable(shape5)
            b = bias_variable([shape5[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.pool2, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv3_1 = conv
            self.parameters += [W, b]

        # conv3_2
        with tf.name_scope('conv3_2') as scope:
            shape6=[3, 3, 256, 256]
            W = weight_variable(shape6)
            b = bias_variable([shape6[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv3_1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv3_2 = conv
            self.parameters += [W, b]

        # conv3_3
        with tf.name_scope('conv3_3') as scope:
            shape7=[3, 3, 256, 256]
            W = weight_variable(shape7)
            b = bias_variable([shape7[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv3_2, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv3_3 =conv
            self.parameters += [W, b]

        # pool3
        self.pool3 = max_pool(self.conv3_3)
        
        # conv4_1
        with tf.name_scope('conv4_1') as scope:
            shape8=[3, 3, 256, 512]
            W = weight_variable(shape8)
            b = bias_variable([shape8[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.pool3, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv4_1 = conv
            self.parameters += [W, b]

        # conv4_2
        with tf.name_scope('conv4_2') as scope:
            shape9=[3, 3, 512, 512]
            W = weight_variable(shape9)
            b = bias_variable([shape9[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv4_1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv4_2 = conv
            self.parameters += [W, b]

        # conv4_3
        with tf.name_scope('conv4_3') as scope:
            shape10=[3, 3, 512, 512]
            W = weight_variable(shape10)
            b = bias_variable([shape10[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv4_2 , W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv4_3 =conv
            self.parameters += [W, b]

        # pool4
        self.pool4 = max_pool(self.conv4_3)
        
        # conv5_1
        with tf.name_scope('conv5_1') as scope:
            shape11=[3, 3, 512, 512]
            W = weight_variable(shape11)
            b = bias_variable([shape11[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.pool4 , W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv5_1 = conv
            self.parameters += [W, b]

        # conv5_2
        with tf.name_scope('conv5_2') as scope:
            shape12=[3, 3, 512, 512]
            W = weight_variable(shape12)
            b = bias_variable([shape12[3]])
            conv = tf.nn.relu(tf.nn.conv2d(self.conv5_1, W, [1, 1, 1, 1], padding='SAME')+b)
            self.conv5_2 = conv
            self.parameters += [W, b]

        # conv5_3
        with tf.name_scope('conv5_3') as scope:
            shape13=[3, 3, 512, 512]
            W = weight_variable(shape13)
            b = bias_variable([shape13[3]])
            conv = tf.nn.conv2d(self.conv5_2, W, [1, 1, 1, 1], padding='SAME')+b
            self.conv5_3 = tf.nn.relu(conv, name=scope)
            self.parameters += [W, b]

        # pool5
        self.pool5 = max_pool(self.conv5_3)
        
        
    def load_weights(self, weight_file, sess):
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if i <= 25:
                sess.run(self.parameters[i].assign(weights[k]))

    def __init__(self, imgs, weights=None, sess=None):
        self.imgs = imgs
        self.imgs_update = tf.Variable(tf.constant(0.0, shape=[1,224,224,3], dtype=tf.float32))
        self.convlayers()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)

    def transfer_style(self, content_features, style_features):
        content_loss = tf.reduce_sum(tf.square(self.conv5_2 - content_features))

        M_5 = style_features[4].shape[1] * style_features[4].shape[2]
        N_5 = style_features[4].shape[3]
        gram_s_5 = gram_matrix(style_features[4], M_5, N_5)
        gram_5 = gram_matrix(self.conv5_1, M_5, N_5)
        result_5 = (1.0 / (4 * N_5**2 * M_5**2)) * tf.reduce_sum(tf.pow(gram_s_5 - gram_5, 2))

        M_4 = style_features[3].shape[1] * style_features[3].shape[2]
        N_4 = style_features[3].shape[3]
        gram_s_4 = gram_matrix(style_features[3], M_4, N_4)
        gram_4 = gram_matrix(self.conv4_1, M_4, N_4)
        result_4 = (1.0 / (4 * N_4**2 * M_4**2)) * tf.reduce_sum(tf.pow(gram_s_4 - gram_4, 2))

        M_3 = style_features[2].shape[1] * style_features[2].shape[2]
        N_3 = style_features[2].shape[3]
        gram_s_3 = gram_matrix(style_features[2], M_3, N_3)
        gram_3 = gram_matrix(self.conv3_1, M_3, N_3)
        result_3 = (1.0 / (4 * N_3**2 * M_3**2)) * tf.reduce_sum(tf.pow(gram_s_3 - gram_3, 2))

        M_2 = style_features[1].shape[1] * style_features[1].shape[2]
        N_2 = style_features[1].shape[3]
        gram_s_2 = gram_matrix(style_features[1], M_2, N_2)
        gram_2 = gram_matrix(self.conv2_1, M_2, N_2)
        result_2 = (1.0 / (4 * N_2**2 * M_2**2)) * tf.reduce_sum(tf.pow(gram_s_2 - gram_2, 2))

        M_1 = style_features[0].shape[1] * style_features[0].shape[2]
        N_1 = style_features[0].shape[3]
        gram_s_1 = gram_matrix(style_features[0], M_1, N_1)
        gram_1 = gram_matrix(self.conv1_1, M_1, N_1)
        result_1 = (1.0 / (4 * N_1**2 * M_1**2)) * tf.reduce_sum(tf.pow(gram_s_1 - gram_1, 2))

        self.loss = 0.000004*content_loss + 0.0001*(result_1+result_2+result_3+result_4)
        # self.train_step = tf.gradients(self.loss, self.imgs)
        self.temp = set(tf.all_variables())
        self.optim = tf.train.AdamOptimizer(1)
        self.train_step = self.optim.minimize(self.loss, var_list=[self.imgs_update])


def gram_matrix(A,M,N):
    v = tf.reshape(A, (M, N))
    return tf.matmul(tf.transpose(v), v)


if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    content_img = imread('input.jpg')
    content_img = imresize(content_img, (224, 224))

    style_img = imread('style.jpg')
    style_img = imresize(style_img, (224, 224))

    s_assign = vgg.imgs_update.assign(np.asarray([style_img]).astype(float))
    sess.run(s_assign)
    style_features = [0 for i in range(5)]
    style_features = sess.run([vgg.conv1_1,vgg.conv2_1,vgg.conv3_1,vgg.conv4_1,vgg.conv5_1], feed_dict={vgg.imgs: [style_img]})

    c_assign = vgg.imgs_update.assign(np.asarray([content_img]).astype(float))
    sess.run(c_assign)
    content_features = sess.run(vgg.conv5_2, feed_dict={vgg.imgs: [content_img]})

    result_img = np.zeros((1,224,224,3)).tolist()
    # r_assign = vgg.imgs_update.assign(np.asarray(result_img).astype(float))
    # sess.run(r_assign)
    vgg.transfer_style(content_features, style_features)

    sess.run(tf.initialize_variables(set(tf.all_variables()) - vgg.temp))

    for i in range(1000):
        loss = sess.run(vgg.loss, feed_dict={vgg.imgs: result_img})
        print("iteration",i,"loss",loss)
        update = sess.run(vgg.train_step, feed_dict={vgg.imgs: result_img})

    result_img = sess.run(vgg.imgs_update, feed_dict={vgg.imgs: result_img})

    # import skimage.io as io
    x = np.asarray(result_img[0]).astype(np.uint8)
    # io.imshow(x)
    # io.show()

    imsave('output.jpg', x)