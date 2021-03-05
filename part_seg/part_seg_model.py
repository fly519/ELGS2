import os
import sys
BASE_DIR = os.path.dirname(__file__)
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 6))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    cls_labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl, cls_labels_pl

NUM_CATEGORIES = 16


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


def get_model(point_cloud, cls_label, is_training, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
    l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,3])

    farthest_distance=0.15
    num_neighbors=4

    #######Contextual representation
    new_xyz = l0_xyz # (batch_size, npoint, 3)
    idx, pts_cnt = query_ball_point(farthest_distance, num_neighbors, l0_xyz, new_xyz)

    neighbor_xyz = group_point(l0_xyz, idx) 
    neighbor_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,num_neighbors,1]) 
    
    neighbor_points = group_point(l0_points, idx) 
    neighbor_representation = tf.concat([neighbor_xyz, neighbor_points], axis=-1) 
    neighbor_representation=tf.reshape(neighbor_representation, (batch_size, num_point, -1)) 

    num_channel=neighbor_representation.get_shape()[2].value
    points= tf_util.conv1d(point_cloud, num_channel, 1, padding='VALID', bn=True, is_training=is_training, scope='points_fc', bn_decay=bn_decay)
    
    neighbor_representation_gp= gating_process(neighbor_representation, num_channel, padding='VALID',  is_training=is_training, scope='neighbor_representation_gp', bn_decay=bn_decay)
    points_gp= gating_process(points, num_channel,  padding='VALID',  is_training=is_training, scope='points_gp', bn_decay=bn_decay)

    l0_points_CR=tf.concat([neighbor_representation_gp*points, points_gp*neighbor_representation], axis=-1)
    l0_points=l0_points_CR

    ########## Positional Representation

    #num_channel=K
    idx, pts_cnt = query_ball_point(0.2, 16, l0_xyz, l0_xyz)
    neighbor_xyz = group_point(l0_xyz, idx)
    # neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)
    xyz_tile = tf.tile(tf.expand_dims(l0_xyz, axis=2), [1, 1, tf.shape(idx)[-1], 1])
    relative_xyz = xyz_tile - neighbor_xyz
    #relative_xyz =neighbor_xyz

    relative_dis = tf.reduce_sum(tf.square(relative_xyz), axis=-1, keepdims=True)
    encoded_position= tf.concat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], axis=-1)
    encoded_position = tf_util.conv2d(encoded_position, 16, [1, 1],
                                      padding='VALID', stride=[1, 1],
                                      bn=True, is_training=is_training,
                                      scope='conv011', bn_decay=bn_decay
                                      )

    encoded_neighbours = group_point(l0_points, idx)
    positional_representation = tf.concat([encoded_neighbours, encoded_position], axis=-1)
    positional_representation= tf.reduce_mean(positional_representation, axis=[2], keep_dims=True, name='avgpool')

    points = tf_util.conv2d(tf.concat([positional_representation, tf.expand_dims(l0_points_CR, 2)], axis=-1),num_channel*2, [1,1],
                                padding='VALID', stride=[1,1],
                                bn=True, is_training=is_training,
                                scope='attp', bn_decay=bn_decay)
    points= tf.squeeze(points, [2])
    end_points['points'] = points
    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.1,0.2,0.4], [32,64,128], [[32,32,64], [64,64,128], [64,96,128]], is_training, bn_decay, scope='layer1')
    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 128, [0.4,0.8], [64,128], [[128,128,256],[128,196,256]], is_training, bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Feature propagation layers
    l2_points = pointnet_fp_module(l2_xyz, l3_xyz, l2_points, l3_points, [256,256], is_training, bn_decay, scope='fa_layer1')
    l1_points = pointnet_fp_module(l1_xyz, l2_xyz, l1_points, l2_points, [256,128], is_training, bn_decay, scope='fa_layer2')

    cls_label_one_hot = tf.one_hot(cls_label, depth=NUM_CATEGORIES, on_value=1.0, off_value=0.0)
    cls_label_one_hot = tf.reshape(cls_label_one_hot, [batch_size, 1, NUM_CATEGORIES])
    cls_label_one_hot = tf.tile(cls_label_one_hot, [1,num_point,1])
    l0_points = pointnet_fp_module(l0_xyz, l1_xyz, tf.concat([cls_label_one_hot, l0_xyz, l0_points],axis=-1), l1_points, [128,128], is_training, bn_decay, scope='fp_layer3')

    # FC layers
    net = tf_util.conv1d(l0_points, 128, 1, padding='VALID', bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)


    ########## Channel-wise Attention
    input = net
    output_a = tf_util.conv2d(tf.expand_dims(input, 1),128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv_output_a', bn_decay=bn_decay) 

    output_b= tf_util.conv2d(tf.expand_dims(input, 1),128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv_output_b', bn_decay=bn_decay)

    output_b = tf.transpose(output_b, [0,1,3,2])
    output_a = tf.squeeze(output_a,[1])
    output_b = tf.squeeze(output_b,[1])
    energy=tf.matmul(output_b,output_a)

    D=tf.reduce_max(energy, -1)
    D=tf.expand_dims(D, -1)   


    energy_new=tf.tile(D, multiples=[1, 1,energy.shape[2]])-energy
    attention=tf.nn.softmax(energy_new,axis=-1)


    output_d= tf_util.conv2d(tf.expand_dims(input, 1),128, [1,1],
                            padding='VALID', stride=[1,1],
                            bn=True, is_training=is_training,
                            scope='conv_output_d', bn_decay=bn_decay)
    output_d= tf.squeeze(output_d,[1])
    output_CA=tf.matmul(output_d,attention)

    gamma2 = tf_util._variable_with_weight_decay('weightsgamma2m',
                                shape=[1],
                                use_xavier=True,
                                stddev=1e-3,
                                wd=0.0)
    output_CA=output_CA*gamma2+input

    ########## Squeeze-and-Excitation

    ex1 = tf.reduce_mean(input, axis=[1], keep_dims=True, name='avgpool1')
    print(ex1 .get_shape())
    ex1 = tf_util.conv1d(ex1,64, 1, padding='VALID', bn=True, is_training=is_training,  scope='ex1', bn_decay=bn_decay)
    print(ex1 .get_shape())
    ex1 = tf_util.conv1d(ex1,128, 1, padding='VALID', bn=True, is_training=is_training, scope='ex2', bn_decay=bn_decay)
    print(ex1 .get_shape())

    output2=input*ex1
    output=tf.concat([output_CA, output2], axis=-1)
    end_points['feats'] = output
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.conv1d(net, 50, 1, padding='VALID', activation_fn=None, scope='fc2')

    return net, end_points


def get_loss(pred, label):
    """ pred: BxNxC,
        label: BxN, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss



if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,2048,6))
        cls_labels = tf.zeros((32),dtype=tf.int32)
        output, ep = get_model(inputs, cls_labels, tf.constant(True))
        print(output)
