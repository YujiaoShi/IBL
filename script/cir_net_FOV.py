# import tensorflow as tf

from VGG import VGG16
from VGG_cir import VGG16_cir

# from utils import *
import tensorflow as tf
from polar_transformer import polar_transformer, projective_transformer



def tf_shape(x, rank):
    static_shape = x.get_shape().with_rank(rank).as_list()
    dynamic_shape = tf.unstack(tf.shape(x), rank)
    return [s if s is not None else d for s,d in zip(static_shape, dynamic_shape)]


def corr(sat_matrix, grd_matrix):

    s_h, s_w, s_c = sat_matrix.get_shape().as_list()[1:]
    g_h, g_w, g_c = grd_matrix.get_shape().as_list()[1:]

    assert s_h == g_h, s_c == g_c

    def warp_pad_columns(x, n):
        out = tf.concat([x, x[:, :, :n, :]], axis=2)
        return out

    n = g_w - 1
    x = warp_pad_columns(sat_matrix, n)
    f = tf.transpose(grd_matrix, [1, 2, 3, 0])
    out = tf.nn.conv2d(x, f,  strides=[1, 1, 1, 1], padding='VALID')
    h, w = out.get_shape().as_list()[1:-1]
    assert h==1, w==s_w

    out = tf.squeeze(out)  # shape = [batch_sat, w, batch_grd]
    orien = tf.argmax(out, axis=1)  # shape = [batch_sat, batch_grd]

    return out, tf.cast(orien, tf.int32)


def crop_sat(sat_matrix, orien, grd_width):
    batch_sat, batch_grd = tf_shape(orien, 2)
    h, w, channel = sat_matrix.get_shape().as_list()[1:]
    sat_matrix = tf.expand_dims(sat_matrix, 1) # shape=[batch_sat, 1, h, w, channel]
    sat_matrix = tf.tile(sat_matrix, [1, batch_grd, 1, 1, 1])
    sat_matrix = tf.transpose(sat_matrix, [0, 1, 3, 2, 4])  # shape = [batch_sat, batch_grd, w, h, channel]

    orien = tf.expand_dims(orien, -1) # shape = [batch_sat, batch_grd, 1]

    i = tf.range(batch_sat)
    j = tf.range(batch_grd)
    k = tf.range(w)
    x, y, z = tf.meshgrid(i, j, k, indexing='ij')

    z_index = tf.mod(z + orien, w)
    x1 = tf.reshape(x, [-1])
    y1 = tf.reshape(y, [-1])
    z1 = tf.reshape(z_index, [-1])
    index = tf.stack([x1, y1, z1], axis=1)

    sat = tf.reshape(tf.gather_nd(sat_matrix, index), [batch_sat, batch_grd, w, h, channel])

    index1 = tf.range(grd_width)
    sat_crop_matrix = tf.transpose(tf.gather(tf.transpose(sat, [2, 0, 1, 3, 4]), index1), [1, 2, 3, 0, 4])
    # shape = [batch_sat, batch_grd, h, grd_width, channel]
    assert sat_crop_matrix.get_shape().as_list()[3] == grd_width

    return sat_crop_matrix


def corr_crop_distance(sat_vgg, grd_vgg):
    corr_out, corr_orien = corr(sat_vgg, grd_vgg)
    sat_cropped = crop_sat(sat_vgg, corr_orien, grd_vgg.get_shape().as_list()[2])
    # shape = [batch_sat, batch_grd, h, grd_width, channel]

    sat_matrix = tf.nn.l2_normalize(sat_cropped, axis=[2, 3, 4])

    distance = 2 - 2 * tf.transpose(tf.reduce_sum(sat_matrix * tf.expand_dims(grd_vgg, axis=0), axis=[2, 3, 4]))
    # shape = [batch_grd, batch_sat]

    return sat_matrix, distance, corr_orien


def threebranch(x_grd, x_polar, x_a2g, keep_prob, trainable, dimension=8):
    ############## VGG module #################
    # ==================================================grd=========================================
    batch, height, width, channel = tf_shape(x_grd, 4)
    # grd
    vgg_grd1 = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_layer13_1 = vgg_grd1.layer13_output
    grd_vgg1 = vgg_grd1.conv2(grd_layer13_1, 'grd', dimension)
    grd_local1 = tf.nn.l2_normalize(grd_vgg1, axis=[1, 2, 3])    # shape = [batch, 4, 64, 8]

    # grd bottom half
    grd_bottom = tf.concat([tf.zeros([batch, int(height / 2), width, channel]), x_grd[:, int(height / 2):, :, :]],
                           axis=1)
    vgg_grd2 = VGG16(grd_bottom, keep_prob, trainable, 'VGG_a2g')
    grd_layer13_2 = vgg_grd2.layer13_output
    grd_vgg2 = vgg_grd2.conv2(grd_layer13_2, 'grd_bottom', dimension)
    grd_local2 = tf.nn.l2_normalize(grd_vgg2, axis=[1, 2, 3])    # shape = [batch, 4, 64, 8]

    # grd global feature
    grd_global = tf.concat([grd_local1, grd_local2], axis=-1)  # shape = [batch, 4, 64, 16]
    grd_global = tf.nn.l2_normalize(grd_global, axis=[1, 2, 3])

    # ==================================================sat=========================================
    # polar
    vgg_polar = VGG16_cir(x_polar, keep_prob, trainable, 'VGG_polar')
    polar_layer13 = vgg_polar.layer13_output
    polar_vgg = vgg_polar.conv2(polar_layer13, 'polar', dimension)
    polar_local = tf.nn.l2_normalize(polar_vgg, axis=[1, 2, 3])    # shape = [batch, 4, 64, 8]

    # a2g bottom half
    vgg_a2g = VGG16_cir(x_a2g, keep_prob, trainable, 'VGG_a2g')
    a2g_layer13 = vgg_a2g.layer13_output
    a2g_vgg = vgg_a2g.conv2(a2g_layer13, 'grd_bottom', dimension)
    a2g_local = tf.nn.l2_normalize(a2g_vgg, axis=[1, 2, 3])    # shape = [batch, 2, 64, 8]

    # sat global feature
    sat_global = tf.concat([polar_local, a2g_local], axis=-1)    # shape = [batch, 4, 64, 16]
    sat_global = tf.nn.l2_normalize(sat_global, axis=[1, 2, 3])

    return grd_global, sat_global, grd_local1, grd_local2, polar_local, a2g_local



def model_threebranch(x_grd, x_polar, x_a2g, keep_prob, trainable, dimension=8):

    grd_global, sat_global, grd_local1, grd_local2, polar_local, a2g_local = \
        threebranch(x_grd, x_polar, x_a2g, keep_prob, trainable, dimension)

    grd_width = grd_global.get_shape().as_list()[2]

    # global
    corr_out, corr_orien = corr(sat_global, grd_global)
    sat_cropped = crop_sat(sat_global, corr_orien, grd_width)
    sat_matrix = tf.nn.l2_normalize(sat_cropped, axis=[2, 3, 4])
    distance_global = 2 - 2 * tf.transpose(tf.reduce_sum(sat_matrix * tf.expand_dims(grd_global, axis=0), axis=[2, 3, 4]))

    # local1
    corr_out1, corr_orien1 = corr(polar_local, grd_local1)

    # local2
    corr_out2, corr_orien2 = corr(a2g_local, grd_local2)

    sat_cropped1 = crop_sat(polar_local, corr_orien1, grd_width)
    sat_cropped2 = crop_sat(a2g_local, corr_orien2, grd_width)

    sat_matrix1 = tf.nn.l2_normalize(sat_cropped1, axis=[2, 3, 4])
    distance_local1 = 2 - 2 * tf.transpose(
        tf.reduce_sum(sat_matrix1 * tf.expand_dims(grd_local1, axis=0), axis=[2, 3, 4]))

    sat_matrix2 = tf.nn.l2_normalize(sat_cropped2, axis=[2, 3, 4])
    distance_local2 = 2 - 2 * tf.transpose(
        tf.reduce_sum(sat_matrix2 * tf.expand_dims(grd_local2, axis=0), axis=[2, 3, 4]))

    return sat_global, grd_global, distance_global, distance_local1, distance_local2, corr_orien, corr_orien1, corr_orien2



def model_twobranch(x_grd, x_sat, keep_prob, trainable):
    vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_layer13 = vgg_grd.layer13_output
    grd_vgg = vgg_grd.conv2(grd_layer13, 'grd', dimension=16)
    grd_vgg = tf.nn.l2_normalize(grd_vgg, axis=[1, 2, 3])

    vgg_sat = VGG16_cir(x_sat, keep_prob, trainable, 'VGG_sat')
    sat_layer13 = vgg_sat.layer13_output
    sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=16)

    sat_matrix, distance, pred_orien = corr_crop_distance(sat_vgg, grd_vgg)

    return sat_vgg, grd_vgg, distance, pred_orien



def model_twobranch_polar(x_grd, x_sat, keep_prob, trainable):
    vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_layer13 = vgg_grd.layer13_output
    grd_vgg = vgg_grd.conv2(grd_layer13, 'grd', dimension=16)
    grd_vgg = tf.nn.l2_normalize(grd_vgg, axis=[1, 2, 3])

    x_polar = polar_transformer(x_sat, 128, 512)
    vgg_sat = VGG16_cir(x_polar, keep_prob, trainable, 'VGG_sat')
    sat_layer13 = vgg_sat.layer13_output
    sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=16)

    sat_matrix, distance, pred_orien = corr_crop_distance(sat_vgg, grd_vgg)

    return sat_vgg, grd_vgg, distance, pred_orien


def model_twobranch_proj(x_grd, x_sat, keep_prob, trainable):
    vgg_grd = VGG16(x_grd, keep_prob, trainable, 'VGG_grd')
    grd_layer13 = vgg_grd.layer13_output
    grd_vgg = vgg_grd.conv2(grd_layer13, 'grd', dimension=16)
    grd_vgg = tf.nn.l2_normalize(grd_vgg, axis=[1, 2, 3])

    x_geo = projective_transformer(x_sat, 128, 512)
    vgg_sat = VGG16_cir(x_geo, keep_prob, trainable, 'VGG_sat')
    sat_layer13 = vgg_sat.layer13_output
    sat_vgg = vgg_sat.conv2(sat_layer13, 'sat', dimension=16)

    sat_matrix, distance, pred_orien = corr_crop_distance(sat_vgg, grd_vgg)

    return sat_vgg, grd_vgg, distance, pred_orien

