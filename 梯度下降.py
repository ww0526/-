# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 14:49:11 2019

@author: 75164
"""

import tensorflow as tf

w = tf.Variable(2,dtype=tf.float32)
x = tf.Variable(3,dtype=tf.float32)
loss = w*x*x
optimizer = tf.train.GradientDescentOptimizer(0.1)
grads_and_vars = optimizer.compute_gradients(loss,[w,x])
grads = tf.gradients(loss,[w,x])
# 修正梯度
for i,(gradient,var) in enumerate(grads_and_vars):
    if gradient is not None:
        grads_and_vars[i] = (tf.clip_by_norm(gradient,5),var)
train_op = optimizer.apply_gradients(grads_and_vars)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5):
        sess.run(train_op)
        print(sess.run(grads_and_vars))
        # 梯度修正前[(9.0, 2.0), (12.0, 3.0)]；梯度修正后 ，[(5.0, 2.0), (5.0, 3.0)]
        print(sess.run(grads))  #[9.0, 12.0]，
    
