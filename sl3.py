import tensorflow as tf
import numpy as np
import pdb
EPS = 1e-10


def SL3(v):
    E1 = tf.constant(np.array([[[1.,0,0],[0,-1,0],[0,0,0]]]), dtype=tf.float32)
    E2 = tf.constant(np.array([[[0.,0,0],[0,-1,0],[0,0,1]]]), dtype=tf.float32)
    E3 = tf.constant(np.array([[[0.,-1,0],[1,0,0],[0,0,0]]]), dtype=tf.float32)
    E4 = tf.constant(np.array([[[0.,1,0],[1,0,0],[0,0,0]]]), dtype=tf.float32)
    E5 = tf.constant(np.array([[[0.,0,1],[0,0,0],[0,0,0]]]), dtype=tf.float32)
    E6 = tf.constant(np.array([[[0.,0,0],[0,0,1],[0,0,0]]]), dtype=tf.float32)
    E7 = tf.constant(np.array([[[0.,0,0],[0,0,0],[1,0,0]]]), dtype=tf.float32)
    E8 = tf.constant(np.array([[[0.,0,0],[0,0,0],[0,1,0]]]), dtype=tf.float32)

    v1, v2, v3, v4, v5, v6, v7, v8 = tf.split(v, 8, axis=1)
    # pdb.set_trace()
    v1 = tf.expand_dims(v1,1)
    v2 = tf.expand_dims(v2,1)
    v3 = tf.expand_dims(v3,1)
    v4 = tf.expand_dims(v4,1)
    v5 = tf.expand_dims(v5,1)
    v6 = tf.expand_dims(v6,1)
    v7 = tf.expand_dims(v7,1)
    v8 = tf.expand_dims(v8,1)

    E = v1*E1+v2*E2+v3*E3+v4*E4+v5*E5+v6*E6+v7*E7+v8*E8
    E = tf.linalg.expm(E)
    E = E / tf.expand_dims(tf.expand_dims(tf.math.pow(tf.linalg.det(E), 1/3), axis=1), axis=2)

    return E
