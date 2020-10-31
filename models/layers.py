import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras.initializers import Constant, GlorotUniform
from tensorflow.keras.losses import BinaryCrossentropy


class CodeEmbedding(Layer):
    def __init__(self, code_num, embedding_size, embedding_init=None, name='code_embedding'):
        super().__init__(name=name)
        if embedding_init is not None:
            self.code_embedding = self.add_weight(name=name + '_v', shape=(code_num + 1, embedding_size),
                                                  initializer=Constant(embedding_init))
        else:
            self.code_embedding = self.add_weight(name=name + '_v', shape=(code_num + 1, embedding_size),
                                                  initializer=GlorotUniform())

    def call(self, inputs=None):
        return self.code_embedding


class GraphConvolution(Layer):
    def __init__(self, adj, hiddens, dropout_rate=0., dropout_seed=6669, activation='relu', name='gcn'):
        super().__init__(name=name)
        self.adj = adj  # (n, n)
        self.denses = [Dense(dim, activation=activation, name='%s_layer_%d' % (name, i))
                       for i, dim in enumerate(hiddens)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate, seed=dropout_seed)

    def call(self, x, **kwargs):
        output = x  # (n, dim)
        for dense in self.denses:
            output = self.dropout(output)
            output = tf.matmul(self.adj, output)  # (n, dim)
            output = dense(output)
        return output


def masked_softmax(inputs, mask):
    inputs = inputs - tf.reduce_max(inputs, keepdims=True, axis=-1)
    exp = tf.exp(inputs) * mask
    result = tf.math.divide_no_nan(exp, tf.reduce_sum(exp, keepdims=True, axis=-1))
    return result


class Attention(Layer):
    def __init__(self, attention_size, name='attention'):
        super().__init__(name=name)
        self.attention_size = attention_size
        self.b_omega = self.add_weight(name=name + '_b', shape=(attention_size,), initializer=GlorotUniform())
        self.u_omega = self.add_weight(name=name + '_u', shape=(attention_size,), initializer=GlorotUniform())
        self.w_omega = None

    def build(self, input_shape):
        hidden_size = input_shape[-1]
        self.w_omega = self.add_weight(name=self.name + '_w', shape=(hidden_size, self.attention_size),
                                       initializer=GlorotUniform())

    def call(self, x, mask=None, **kwargs):
        # x: (**size, dim)
        v = tf.tanh(tf.matmul(x, self.w_omega) + self.b_omega)  # (**size, attention_size)
        vu = tf.tensordot(v, self.u_omega, axes=1)  # (**size)
        if mask is not None:
            vu *= mask
            alphas = masked_softmax(vu, mask)
        else:
            alphas = tf.nn.softmax(vu)  # (**size)
        output = tf.reduce_sum(x * tf.expand_dims(alphas, -1), axis=-2)  # (**size, dim)
        return output, alphas


class AttentionVisit(Attention):
    def __init__(self, attention_size, output_dim, name='attention_visit'):
        super().__init__(attention_size=attention_size, name=name)
        self.u_omega_o = self.add_weight(name=name + '_u', shape=(attention_size, output_dim),
                                         initializer=GlorotUniform())

    def call(self, x, mask=None, **kwargs):
        # x: (**size, dim)
        t = tf.math.l2_normalize(tf.matmul(x, self.w_omega) + self.b_omega, axis=-1)
        v = tf.tanh(t)  # (**size, attention_size)
        vu = tf.tensordot(v, self.u_omega, axes=1)  # (**size)
        vu_o = tf.tensordot(v, self.u_omega_o, axes=1)  # (**size, output_dim)
        if mask is not None:
            vu *= mask
            mask_o = tf.expand_dims(mask, axis=-1)
            vu_o *= mask_o
            alphas = masked_softmax(vu, mask)
            betas = masked_softmax(vu_o, mask_o)
        else:
            alphas = tf.nn.softmax(vu)  # (**size)
            betas = tf.nn.softmax(vu_o)  # (**size, output_dim)
        w = tf.expand_dims(alphas, axis=-1) * betas
        output = tf.reduce_sum(x * w, axis=-2)  # (**size, dim)
        return output, alphas, betas


class Encoder(Layer):
    def __init__(self, max_visit_num,
                 attention_size_code, attention_size_visit,
                 patient_size, patient_activation='relu',
                 name='encoder'):
        super().__init__(name=name)
        self.max_visit_num = max_visit_num
        self.attention_code = Attention(attention_size_code, name=name + '_attention_code')
        self.dense_v2p = Dense(patient_size, activation=patient_activation)
        self.attention_visit = AttentionVisit(attention_size_visit, patient_size, name=name + '_attention_visit')

    def call(self, code_embeddings, visit_codes, visit_lens, **kwargs):
        # x: (batch_size, max_seq_len, max_code_in_a_visit)
        embeddings = tf.nn.embedding_lookup(code_embeddings, visit_codes)  # x: (batch_size, max_seq_len, max_code_in_a_visit, code_dim)
        mask_code = tf.cast(visit_codes > 0, embeddings.dtype)  # (batch_size, max_seq_len, max_code_in_a_visit)
        embeddings = embeddings * tf.expand_dims(mask_code, axis=-1)
        visit_embeddings, code_alphas = self.attention_code(embeddings, mask_code)  # x: (batch_size, max_seq_len, code_dim)

        mask_visit = tf.sequence_mask(visit_lens, self.max_visit_num, dtype=visit_embeddings.dtype)
        patient_embeddings = self.dense_v2p(visit_embeddings)
        patient_embeddings = patient_embeddings * tf.expand_dims(mask_visit, axis=-1)
        patient_embedding, admission_alphas, betas = self.attention_visit(patient_embeddings, mask_visit)
        return patient_embedding, admission_alphas, betas


class Decoder(Layer):
    def __init__(self, output_dim, dropout_rate=0., dropout_seed=6669, activation=None, name='decoder'):
        super().__init__(name=name)
        self.dense = Dense(output_dim, activation=activation)
        self.dropout = tf.keras.layers.Dropout(dropout_rate, seed=dropout_seed)

    def call(self, patient_embedding, **kwargs):
        patient_embedding = self.dropout(patient_embedding)
        output = self.dense(patient_embedding)
        return output


class HierarchicalDecoder(Layer):
    def __init__(self, subclass_dims, subclass_maps, name='hierarchical_decoder'):
        super().__init__(name=name)
        self.total_num = tf.cast(tf.reduce_sum(subclass_dims), self.dtype)
        self.level_num = len(subclass_dims)
        self.subclass_dims = subclass_dims
        self.subclass_maps = subclass_maps  # len: level_num - 1, len(subclass_maps[i]): subclass_dims[i + 1]
        self.denses = [Dense(dim, activation=None) for dim in subclass_dims]
        self.bce = BinaryCrossentropy()

    def call(self, patient_embedding, y_trues, **kwargs):
        # patient_embedding: (batch_size, patient_size)
        t = self.denses[0](patient_embedding)  # (batch_size, level_1_dim)
        prob = tf.nn.sigmoid(t)  # (batch_size, level1_dim)
        prob_level = [prob]
        for dense, subclass_map in zip(self.denses[1:], self.subclass_maps):
            prob_last = tf.transpose(prob_level[-1])  # (level_i-1_dim, batch_size)
            t = tf.transpose(tf.nn.sigmoid(dense(patient_embedding)))  # (level_i_dim, batch_size)
            prob = tf.zeros_like(t, dtype=t.dtype)  # (level_i_dim, batch_size)
            for k, subclass in enumerate(subclass_map):
                index = tf.expand_dims(subclass, axis=-1)  # (subclass_num, 1)
                logits_subclass = tf.gather_nd(t, index)  # (subclass_num, batch_size)
                prob_subclass = tf.expand_dims(prob_last[k, :], axis=0) * logits_subclass  # (subclass_num, batch_size)
                prob = tf.tensor_scatter_nd_update(prob, index, prob_subclass)  # (level_i_dim, batch_size)
            prob_level.append(tf.transpose(prob))
        loss = 0.0
        for subclass_dim, y_pred, y_true in zip(self.subclass_dims, prob_level, y_trues):
            loss += self.bce(y_true, y_pred)  # * subclass_dim
        loss /= self.level_num
        loss *= self.total_num
        self.add_loss(loss)
        return prob_level[-1]
