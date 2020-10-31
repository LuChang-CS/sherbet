import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer

from models.layers import CodeEmbedding, GraphConvolution, Encoder, HierarchicalDecoder, Decoder


class SherbetFeature(Layer):
    def __init__(self, conf, hyper_params, name='sherbet_feature'):
        super().__init__(name=name)
        self.code_embedding = CodeEmbedding(code_num=conf['code_num'],
                                            embedding_size=hyper_params['code_embedding_size'],
                                            embedding_init=conf['code_embedding_init'])
        self.graph_convolution = GraphConvolution(adj=conf['adj'], hiddens=hyper_params['hiddens'],
                                                  dropout_rate=hyper_params['gnn_dropout_rate'])
        self.encoder = Encoder(max_visit_num=conf['max_visit_num'],
                               attention_size_code=hyper_params['attention_size_code'],
                               attention_size_visit=hyper_params['attention_size_visit'],
                               patient_size=hyper_params['patient_size'],
                               patient_activation=hyper_params['patient_activation'])

    def call(self, visit_codes, visit_lens, **kwargs):
        embeddings = self.code_embedding(None)
        embeddings = self.graph_convolution(embeddings)
        patient_embedding, admission_alphas, betas = self.encoder(embeddings, visit_codes, visit_lens)
        return patient_embedding, admission_alphas, betas


class Sherbet(Model):
    def __init__(self, feature_extractor, conf, hyper_params, name='sherbet'):
        super().__init__(name=name)
        self.feature_extractor = feature_extractor
        self.conf = conf
        if conf['use_hierarchical_decoder']:
            self.decoder = HierarchicalDecoder(subclass_dims=conf['subclass_dims'], subclass_maps=conf['subclass_maps'])
        else:
            self.decoder = Decoder(output_dim=conf['output_dim'], activation=conf['activation'],
                                   dropout_rate=hyper_params['decoder_dropout_rate'])

    def call(self, inputs, training=None, mask=None):
        visit_codes = inputs['visit_codes']  # (batch_size, max_seq_len, max_code_num_in_a_visit)
        visit_lens = tf.reshape(inputs['visit_lens'], (-1, ))  # (batch_size, )
        y_trues = inputs['y_trues'] if self.conf['use_hierarchical_decoder'] else None
        patient_embedding, admission_alphas, betas = self.feature_extractor(visit_codes, visit_lens)
        if self.conf['use_hierarchical_decoder']:
            output = self.decoder(patient_embedding, y_trues)
        else:
            output = self.decoder(patient_embedding)
        return output
