
import tensorflow as tf


class VGG16(object):

    def __init__(self, x, keep_prob, trainable, name):
        '''
        :param x: x.shape = [batch, 128, 512, channel]
        :param keep_prob:
        :param trainable:
        :param name:
        '''
        self.trainable = trainable
        self.name = name

        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # layer 1: conv3-64
            self.layer1_output = self.conv_layer(x, 3, 3, 64, False, True, 'conv1_1')
            # layer 2: conv3-64
            self.layer2_output = self.conv_layer(self.layer1_output, 3, 64, 64, False, True, 'conv1_2')
            # layer3: max pooling
            self.layer3_output = self.maxpool_layer(self.layer2_output, 'layer3_maxpool2x2') # shape = [64, 256]

            # layer 4: conv3-128
            self.layer4_output = self.conv_layer(self.layer3_output, 3, 64, 128, False, True, 'conv2_1')
            # layer 5: conv3-128
            self.layer5_output = self.conv_layer(self.layer4_output, 3, 128, 128, False, True, 'conv2_2')
            # layer 6: max pooling
            self.layer6_output = self.maxpool_layer(self.layer5_output, 'layer6_maxpool2x2') # shape = [32, 128]

            # layer 7: conv3-256
            self.layer7_output = self.conv_layer(self.layer6_output, 3, 128, 256, False, True, 'conv3_1')
            # layer 8: conv3-256
            self.layer8_output = self.conv_layer(self.layer7_output, 3, 256, 256, False, True, 'conv3_2')
            # layer 9: conv3-256
            self.layer9_output = self.conv_layer(self.layer8_output, 3, 256, 256, False, True, 'conv3_3')
            # layer 10: max pooling
            self.layer10_output = self.maxpool_layer(self.layer9_output, 'layer10_maxpool2x2') # shape = [16, 64]

            # layer 11: conv3-512
            self.layer11_output = self.conv_layer(self.layer10_output, 3, 256, 512, trainable, True, 'conv4_1')
            self.layer11_output = tf.nn.dropout(self.layer11_output, keep_prob, name='conv4_1_dropout')
            # layer 12: conv3-512
            self.layer12_output = self.conv_layer(self.layer11_output, 3, 512, 512, trainable, True, 'conv4_2')
            self.layer12_output = tf.nn.dropout(self.layer12_output, keep_prob, name='conv4_2_dropout')
            # layer 13: conv3-512
            self.layer13_output = self.conv_layer(self.layer12_output, 3, 512, 512, trainable, True, 'conv4_3')
            self.layer13_output = tf.nn.dropout(self.layer13_output, keep_prob, name='conv4_3_dropout') # shape = [16, 64]
            # layer 14: max pooling
            self.layer14_output = self.maxpool_layer(self.layer13_output, 'layer14_maxpool2x2')

            # # layer 15: conv3-512
            # self.layer15_output = self.conv_layer(self.layer14_output, 3, 512, 512, trainable, True, 'conv5_1')
            # self.layer15_output = tf.nn.dropout(self.layer15_output, keep_prob, name='conv5_1_dropout')
            # # layer 16: conv3-512
            # self.layer16_output = self.conv_layer(self.layer15_output, 3, 512, 512, trainable, True, 'conv5_2')
            # self.layer16_output = tf.nn.dropout(self.layer16_output, keep_prob, name='conv5_2_dropout')
            # # layer 17: conv3-512
            # self.layer17_output = self.conv_layer(self.layer16_output, 3, 512, 512, trainable, True, 'conv5_3')   # shape = [8, 32]
            # self.layer17_output = tf.nn.dropout(self.layer17_output, keep_prob, name='conv5_3_dropout')


    def conv2d(self, x, W, strides=[1,1,1,1]):
        # w_pad = tf.pad
        return tf.nn.conv2d(x, W, strides,
                            padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME')

    ############################ layers ###############################
    def conv_layer(self, x, kernel_dim, input_dim, output_dim, trainable, activated,
                   name='layer_conv', activation_function=tf.nn.relu):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE): # reuse=tf.AUTO_REUSE
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            if activated:
                out = activation_function(self.conv2d(x, weight) + bias)
            else:
                out = self.conv2d(x, weight) + bias

            return out


    def conv_layer2(self, x, kernel_dim, strides, input_dim, output_dim, trainable, activated,
                   name='layer_conv', activation_function=tf.nn.relu):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE): # reuse=tf.AUTO_REUSE
            weight = tf.get_variable(name='weights', shape=[kernel_dim, kernel_dim, input_dim, output_dim],
                                     trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())
            bias = tf.get_variable(name='biases', shape=[output_dim],
                                   trainable=trainable, initializer=tf.contrib.layers.xavier_initializer())

            if activated:
                out = activation_function(self.conv2d(x, weight, strides) + bias)
            else:
                out = self.conv2d(x, weight, strides) + bias

            return out



    def maxpool_layer(self, x, name):
        with tf.name_scope(name):
            maxpool = self.max_pool_2x2(x)
            return maxpool

    def conv1(self, x, scope_name, dimension):
        '''
        :param x: shape = [batch, height, widh, 512]
        :param scope_name:
        :param dimension:
        :return:
        '''
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            layer15_output = self.conv_layer(x, 3, 512, 256, self.trainable, True, 'conv_dim_reduct_1')

            layer16_output = self.conv_layer(layer15_output, 3, 256, 64, self.trainable, True, 'conv5_dim_reduct_2')

            layer17_output = self.conv_layer(layer16_output, 3, 64, dimension, self.trainable, False, 'conv5_dim_reduct_3')  # shape = [16, 64]
            # shape = [batch, height, width, dimension]
        return layer17_output

    def conv2(self, x, scope_name, dimension=16):
        '''
        :param x: shape = [batch, height, widh, 512]
        :param scope_name:
        :param dimension:
        :return:
        '''
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            layer15_output = self.conv_layer2(x, 3, [1, 2, 1, 1], 512, 256, self.trainable, True, 'conv_dim_reduct_1')
            #shape = [batch, height/2, widh, 256]
            layer16_output = self.conv_layer2(layer15_output, 3, [1, 2, 1, 1], 256, 64, self.trainable, True, 'conv5_dim_reduct_2')
            # shape = [batch, height/4, widh, 64]
            layer17_output = self.conv_layer(layer16_output, 3, 64, dimension, self.trainable, False, 'conv5_dim_reduct_3')  # shape = [7, 39]
            # shape = [batch, height/4, widh, dimension]
        return layer17_output

    def conv3(self, x, scope_name, dimension=16):
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            layer15_output = self.conv_layer2(x, 3, [1, 2, 1, 1], 512, 256, self.trainable, True, 'conv_dim_reduct_1')

            layer16_output = self.conv_layer2(layer15_output, 3, [1, 2, 1, 1], 256, 64, self.trainable, True, 'conv5_dim_reduct_2')

            layer17_output = self.conv_layer2(layer16_output, 3, [1, 2, 1, 1], 64, dimension, self.trainable, False, 'conv5_dim_reduct_3')  # shape = [7, 39]
            # shape = [batch, height/8, widh, dimension]
        return layer17_output

        # the convolutional part of VGG16-D
    # def VGG16_conv(self, x, keep_prob, trainable, name):
    #         print('VGG16: trainable =', trainable)
    #
    #         with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
    #             # layer 1: conv3-64
    #             layer1_output = self.conv_layer(x, 3, 3, 64, trainable, True, 'conv1_1')
    #             # layer 2: conv3-64
    #             layer2_output = self.conv_layer(layer1_output, 3, 64, 64, trainable, True, 'conv1_2')
    #             # layer3: max pooling
    #             layer3_output = self.maxpool_layer(layer2_output, 'layer3_maxpool2x2')
    #
    #             # layer 4: conv3-128
    #             layer4_output = self.conv_layer(layer3_output, 3, 64, 128, trainable, True, 'conv2_1')
    #             # layer 5: conv3-128
    #             layer5_output = self.conv_layer(layer4_output, 3, 128, 128, trainable, True, 'conv2_2')
    #             # layer 6: max pooling
    #             layer6_output = self.maxpool_layer(layer5_output, 'layer6_maxpool2x2')
    #
    #             # layer 7: conv3-256
    #             layer7_output = self.conv_layer(layer6_output, 3, 128, 256, trainable, True, 'conv3_1')
    #             # layer 8: conv3-256
    #             layer8_output = self.conv_layer(layer7_output, 3, 256, 256, trainable, True, 'conv3_2')
    #             # layer 9: conv3-256
    #             layer9_output = self.conv_layer(layer8_output, 3, 256, 256, trainable, True, 'conv3_3')
    #             # layer 10: max pooling
    #             layer10_output = self.maxpool_layer(layer9_output, 'layer10_maxpool2x2')
    #
    #             # layer 11: conv3-512
    #             layer11_output = self.conv_layer(layer10_output, 3, 256, 512, trainable, True, 'conv4_1')
    #             layer11_output = tf.nn.dropout(layer11_output, keep_prob, name='conv4_1_dropout')
    #             # layer 12: conv3-512
    #             layer12_output = self.conv_layer(layer11_output, 3, 512, 512, trainable, True, 'conv4_2')
    #             layer12_output = tf.nn.dropout(layer12_output, keep_prob, name='conv4_2_dropout')
    #             # layer 13: conv3-512
    #             layer13_output = self.conv_layer(layer12_output, 3, 512, 512, trainable, True, 'conv4_3')
    #             layer13_output = tf.nn.dropout(layer13_output, keep_prob, name='conv4_3_dropout')
    #             # layer 14: max pooling
    #             layer14_output = self.maxpool_layer(layer13_output, 'layer14_maxpool2x2')
    #
    #             # layer 15: conv3-512
    #             layer15_output = self.conv_layer(layer14_output, 3, 512, 512, trainable, True, 'conv5_1')
    #             layer15_output = tf.nn.dropout(layer15_output, keep_prob, name='conv5_1_dropout')
    #             # layer 16: conv3-512
    #             layer16_output = self.conv_layer(layer15_output, 3, 512, 512, trainable, True, 'conv5_2')
    #             layer16_output = tf.nn.dropout(layer16_output, keep_prob, name='conv5_2_dropout')
    #             # layer 17: conv3-512
    #             layer17_output = self.conv_layer(layer16_output, 3, 512, 512, trainable, True, 'conv5_3')
    #             layer17_output = tf.nn.dropout(layer17_output, keep_prob, name='conv5_3_dropout')
    #
    #             return layer17_output


