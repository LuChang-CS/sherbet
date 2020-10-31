import os
import random
import _pickle as pickle

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import numpy as np

from utils import lr_decay


seed = 6669
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def flatten_hierarchy(subclass_dims, subclass_maps):
    h2f_map = dict()
    for level, subclass_dim in enumerate(subclass_dims):
        for code in range(1, subclass_dim + 1):
            h2f_map[(code, level)] = len(h2f_map) + 1
    pc_map = dict()
    pc_map[0] = [h2f_map[(code, 0)] for code in range(1, subclass_dims[0] + 1)]
    for level, subclass_dim in enumerate(subclass_dims[:-1]):
        for code in range(1, subclass_dim + 1):
            pc_map[h2f_map[(code, level)]] = [h2f_map[(c + 1, level + 1)] for c in subclass_maps[level][code - 1]]
    pc = []
    for i in range(len(pc_map)):
        pc.append(pc_map[i])
    cp = np.zeros((len(h2f_map) + 1, ), dtype=int)
    for parent, children in pc_map.items():
        for c in children:
            cp[c] = parent
    return h2f_map, pc, cp


def build_adjacent(node_num, pc):
    result = np.zeros((node_num + 1, node_num + 1), dtype=np.float64)
    for parent, children in enumerate(pc):
        for c in children:
            result[parent][c] = 1
            result[c][parent] = 1
    return result


class HierarchicalEmbedding(Layer):
    def __init__(self, pc, cp, node_num, embedding_size=128):
        super().__init__(dtype=tf.float64)
        self.pc = pc
        self.cp = cp
        self.node_num_with_children = len(pc)
        self.node_num_without_parent = 1
        self.s = self.add_weight(shape=(node_num + 1, embedding_size),
                                 initializer=tf.keras.initializers.GlorotUniform())  # global
        self.t = self.add_weight(shape=(node_num + 1, embedding_size),
                                 initializer=tf.keras.initializers.GlorotUniform())  # local
        self.lambda_ = self.add_weight(shape=(node_num + 1, 1),
                                       initializer=tf.keras.initializers.GlorotUniform())

    def call(self, inputs, **kwargs):
        lambda_ = self.lambda_
        e_prime = self.s * lambda_ + self.t * (1 - lambda_)
        s_left = self.s[:self.node_num_without_parent]
        s_right = tf.nn.embedding_lookup(e_prime, self.cp[self.node_num_without_parent:])
        s = tf.concat([s_left, s_right], axis=0)
        t_left = tf.stack([tf.reduce_mean(tf.nn.embedding_lookup(self.t, tf.cast(self.pc[i], dtype=tf.int32)), axis=0)
                           for i in range(self.node_num_with_children)], axis=0)
        t_right = self.t[self.node_num_with_children:]
        t = tf.concat([t_left, t_right], axis=0)
        e = s * lambda_ + t * (1 - lambda_)
        return e


class HyperbolicDecoder(Model):
    def __init__(self, pc, cp, adj, adj_mask, embedding_size=128):
        super().__init__(dtype=tf.float64)
        self.embeddings = HierarchicalEmbedding(pc, cp, node_num, embedding_size)
        self.adj = adj
        self.mask = adj_mask
        self.eps = 1e-10
        self.max_norm = 1 - self.eps

    def distance(self, u, v):
        sq_u_norm = tf.clip_by_value(
            tf.reduce_sum(u * u, axis=-1),
            clip_value_min=0,
            clip_value_max=self.max_norm
        )
        sq_v_norm = tf.clip_by_value(
            tf.reduce_sum(v * v, axis=-1),
            clip_value_min=0,
            clip_value_max=self.max_norm
        )
        sq_dist = tf.reduce_sum((u - v) ** 2, axis=-1)
        x = 1 + (sq_dist / ((1 - sq_u_norm) * (1 - sq_v_norm))) * 2
        distance = 1 / (x + tf.sqrt(x ** 2 - 1))
        return distance

    def log_no_nan(self, x):
        mask = tf.cast(x == 0, x.dtype)
        return tf.math.log(x + mask)

    def rec_loss(self, nid, distance):
        a = distance
        b = tf.reduce_sum(a * tf.nn.embedding_lookup(self.mask, nid), axis=-1, keepdims=True)
        c = (a * tf.nn.embedding_lookup(self.adj, nid)) / (b + a)
        d = self.log_no_nan(c)
        loss = -tf.reduce_mean(tf.reduce_sum(d, axis=-1))
        return loss

    def call(self, nid, training=None, mask=None):
        embeddings = self.embeddings(None)
        u = tf.nn.embedding_lookup(embeddings, nid)
        v = tf.expand_dims(embeddings, axis=0)
        distance = self.distance(u, v)
        nid = tf.squeeze(nid)
        loss = self.rec_loss(nid, distance)
        self.add_loss(loss)
        return loss


if __name__ == '__main__':
    dataset = 'mimic3'  # 'mimic3' or 'eicu'
    dataset_path = os.path.join('data', dataset)
    standard_path = os.path.join(dataset_path, 'standard')
    auxiliary = pickle.load(open(os.path.join(standard_path, 'auxiliary.pkl'), 'rb'))
    code_levels, subclass_maps = auxiliary['code_levels_pretrain'], auxiliary['subclass_maps_pretrain']
    subclass_dims = np.max(code_levels, axis=0)

    h2f_map, pc, cp = flatten_hierarchy(subclass_dims, subclass_maps)
    node_num = len(h2f_map)
    adj = build_adjacent(node_num, pc)

    adj_mask = 1 - adj - np.eye(len(adj))
    with tf.device('/GPU:0'):
        embedding_size = 128
        epochs = 500
        batch_size = 256
        learning_rate = 1e-2
        # split_val = [(20, 1e-3), (27, 1e-8), (80, 1e-5), (100, 1e-8)]
        # split_val = [(20, 1e-3), (30, 1e-4), (40, 1e-5), (50, 1e-6), (60, 1e-7), (70, 1e-8)]
        # split_val = [(50, 1e-3), (100, 1e-4), (150, 1e-5), (200, 1e-6)]
        split_val = [(100, 1e-3), (200, 1e-4), (300, 1e-5), (400, 1e-6)]

        lr_schedule_fn = lr_decay(total_epoch=epochs, init_lr=learning_rate, split_val=split_val)
        lr_scheduler = LearningRateScheduler(lr_schedule_fn)

        optimizer = Adam(learning_rate=learning_rate)
        decoder = HyperbolicDecoder(pc, cp, adj=adj, adj_mask=adj_mask, embedding_size=embedding_size)
        decoder.compile(optimizer=Adam(learning_rate=learning_rate), loss=None)
        decoder.fit(x=np.arange(node_num + 1).reshape((-1, 1)), epochs=epochs, batch_size=batch_size,
                    callbacks=[lr_scheduler])

        embeddings = decoder.embeddings(None).numpy()
        level = len(subclass_dims) - 1
        leaf_embeddings = np.zeros((subclass_dims[-1] + 1, 128), dtype=np.float64)
        for i in range(1, subclass_dims[-1] + 1):
            c = h2f_map[(i, level)]
            vec = embeddings[c]
            leaf_embeddings[i] = vec
        pickle.dump(leaf_embeddings, open('./saved/hyperbolic/%s_leaf_embeddings' % dataset, 'wb'))
