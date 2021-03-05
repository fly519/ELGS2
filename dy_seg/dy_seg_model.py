"""
    Compared with model_baseline, do not use correlation output for skip link
    Compared to model_baseline_fixed, added return values to test whether nsample is set reasonably.
"""

import tensorflow as tf
import numpy as np
import math
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/grouping'))

from tf_sampling import farthest_point_sample, gather_point
#from tf_grouping import query_ball_point, query_ball_point_var_rad, group_point, knn_point
from tf_grouping import query_ball_point,  group_point, knn_point

import tf_util
from net_utils_GAT import *


def gating_process(inputs,num_output_channels,scope,stride=1,padding='VALID',bn_decay=None,is_training=None):

  with tf.variable_scope(scope) as sc:    
    num_in_channels = inputs.get_shape()[-1].value  
    kernel_shape = [1, num_in_channels, num_output_channels]

    with tf.device("/cpu:0"):       
        kernel = tf.get_variable('weights', kernel_shape, initializer= tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
        biases = tf.get_variable('biases', [num_output_channels], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
  
    df='NHWC'
    outputs = tf.nn.conv1d(inputs, kernel, stride=stride, padding=padding, data_format=df)
    outputs = tf.nn.bias_add(outputs, biases, data_format=df)
    

    outputs =tf.contrib.layers.batch_norm(outputs, 
                                      center=True, scale=True,
                                      is_training=is_training, decay=bn_decay,updates_collections=None,
                                      scope='bn',
                                      data_format=df)
    outputs = tf.nn.relu(outputs)

    return outputs

def placeholder_inputs(batch_size, num_point, num_frames):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * num_frames, 3 + 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point * num_frames))
    labelweights_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * num_frames))
    masks_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point * num_frames))
    return pointclouds_pl, labels_pl, labelweights_pl, masks_pl

def get_model(point_cloud, num_frames, is_training, bn_decay=None):
    """ Semantic segmentation PointNet, input is BxNx3, output Bxnum_class """
    end_points = {}
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value // num_frames

    l0_xyz = point_cloud[:, :, 0:3]
    #l0_time = tf.concat([tf.ones([batch_size, num_point, 1]) * i for i in range(num_frames)], \
    #        axis=-2)
    #l0_points = tf.concat([point_cloud[:, :, 3:], l0_time], axis=-1)

    l0_points = point_cloud[:, :, 3:]

    #######Contextual representation

    farthest_distance=0.6
    num_neighbors=4
    new_xyz = l0_xyz # (batch_size, npoint, 3)
    idx, pts_cnt = query_ball_point(farthest_distance, num_neighbors, l0_xyz, new_xyz)

    neighbor_xyz = group_point(l0_xyz, idx) 
    neighbor_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,num_neighbors,1]) 
    
    neighbor_points = group_point(l0_points, idx) 
    neighbor_representation = tf.concat([neighbor_xyz, neighbor_points], axis=-1) 
    neighbor_representation=tf.reshape(neighbor_representation, (batch_size, point_cloud.get_shape()[1].value, -1)) 

    num_channel=neighbor_representation.get_shape()[2].value
    points= tf_util.conv1d(point_cloud, num_channel, 1, padding='VALID', bn=True, is_training=is_training, scope='points_fc', bn_decay=bn_decay)
    
    neighbor_representation_gp= gating_process(neighbor_representation, num_channel, padding='VALID',  is_training=is_training, scope='neighbor_representation_gp', bn_decay=bn_decay)
    points_gp= gating_process(points, num_channel,  padding='VALID',  is_training=is_training, scope='points_gp', bn_decay=bn_decay)

    l0_points_CR=tf.concat([neighbor_representation_gp*points, points_gp*neighbor_representation], axis=-1)

    ########## Positional Representation

    idx, pts_cnt = query_ball_point(0.6, 32, l0_xyz, l0_xyz)
    neighbor_xyz = group_point(l0_xyz, idx)
    # neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
    xyz_tile = tf.tile(tf.expand_dims(l0_xyz, axis=2), [1, 1, tf.shape(idx)[-1], 1])
    relative_xyz = xyz_tile - neighbor_xyz
    #relative_xyz =neighbor_xyz

    relative_dis = tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True)
    encoded_position= tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
    encoded_position = tf_util.conv2d(encoded_position, num_channel*2, [1, 1],
                                padding='VALID', stride=[1, 1],
                                bn=True, is_training=is_training,
                                scope='conv011' , bn_decay=bn_decay
                                )

    positional_representation= tf.reduce_mean(encoded_position, axis=[2], keep_dims=True, name='avgpool')
    l0_points = tf_util.conv2d(tf.concat([positional_representation, tf.expand_dims(l0_points_CR, 2)], axis=-1),num_channel*2, [1,1],
                                padding='VALID', stride=[1,1],
                                bn=True, is_training=is_training,
                                scope='attp', bn_decay=bn_decay)
    l0_points= tf.squeeze(l0_points, [2])

    l1_xyz, l1_points, l1_indices = pointnet_sa_module_withgab(l0_xyz, l0_points, npoint=2048, radius=1.0, nsample=32, mlp=[32,32,64], mlp2=None, group_all=False,knn=False,  is_training=is_training, bn_decay=bn_decay, scope='layer1',gab=True)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=512, radius=2.0, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False,knn=False,  is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=128, radius=4.0, nsample=32, mlp=[128,128,256], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer3')
    l4_xyz, l4_points, l4_indices = pointnet_sa_module(l3_xyz, l3_points, npoint=64, radius=8.0, nsample=32, mlp=[256,256,512], mlp2=None, group_all=False, knn=False, is_training=is_training, bn_decay=bn_decay, scope='layer4')

    # Feature Propagation layers
    l3_points = pointnet_fp_module(l3_xyz, l4_xyz, l3_points, l4_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer2')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer3')
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, l0_points, l1_points, [128,128], is_training, bn_decay, scope='fa_layer4')

    ##### debug
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)

    ########## Channel-wise Attention
    input = net
    output_f = tf.transpose(input, [0, 2, 1])
    energy = tf.matmul(output_f, input)

    D = tf.reduce_max(energy, -1)
    D = tf.expand_dims(D, -1)

    energy_new = tf.tile(D, multiples=[1, 1, energy.shape[2]]) - energy
    attention = tf.nn.softmax(energy_new, axis=-1)

    output_CA = tf.matmul(input, attention)

    gamma2 = tf_util._variable_with_weight_decay('weightsgamma2m',
                                                 shape=[1],
                                                 use_xavier=True,
                                                 stddev=1e-3,
                                                 wd=0.0)
    output_CA = output_CA * gamma2 + input
    output_CA=tf_util.conv1d(output_CA, 2, 1, padding='VALID', activation_fn=None, scope='cpm')
    end_points['feats'] =output_CA

    ########## Squeeze-and-Excitation

    ex1 = tf.reduce_mean(input, axis=[1], keep_dims=True, name='avgpool1')
    print(ex1 .get_shape())
    ex1 = tf_util.conv1d(ex1,64, 1, padding='VALID',  scope='ex1')
    print(ex1 .get_shape())
    ex1 = tf_util.conv1d(ex1,128, 1, padding='VALID',  scope='ex2')
    print(ex1 .get_shape())
    output=input*ex1

    net = tf_util.dropout(output, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 12, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points

def get_loss(pred, label, mask, end_points, label_weights):
    """ pred: BxNx3,
        label: BxN,
        mask: BxN
    """
    classify_loss = tf.losses.sparse_softmax_cross_entropy( labels=label, \
                                                            logits=pred, \
                                                            weights=label_weights, \
                                                            reduction=tf.losses.Reduction.NONE)  
    classify_loss = tf.reduce_sum(classify_loss * mask) / (tf.reduce_sum(mask) + 1)

    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    tf.add_to_collection('losses1', classify_loss)

    return classify_loss

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024*2,6))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
