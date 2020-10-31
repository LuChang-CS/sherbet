import os
import random
import _pickle as pickle

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np

from models.model import Sherbet, SherbetFeature
from models.loss import medical_codes_loss
from metrics import EvaluateCodesCallBack, EvaluateHFCallBack
from utils import DataGenerator, lr_decay


seed = 6669
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


def load_data(dataset_path):
    encoded_path = os.path.join(dataset_path, 'encoded')
    standard_path = os.path.join(dataset_path, 'standard')
    code_maps = pickle.load(open(os.path.join(encoded_path, 'code_maps.pkl'), 'rb'))
    pretrain_codes_data = pickle.load(open(os.path.join(standard_path, 'pretrain_codes_dataset.pkl'), 'rb'))
    codes_dataset = pickle.load(open(os.path.join(standard_path, 'codes_dataset.pkl'), 'rb'))
    hf_dataset = pickle.load(open(os.path.join(standard_path, 'heart_failure.pkl'), 'rb'))
    auxiliary = pickle.load(open(os.path.join(standard_path, 'auxiliary.pkl'), 'rb'))
    return code_maps, pretrain_codes_data, codes_dataset, hf_dataset, auxiliary


if __name__ == '__main__':
    dataset = 'mimic3'  # 'mimic3' or 'eicu'
    dataset_path = os.path.join('data', dataset)
    code_maps, pretrain_codes_data, codes_dataset, hf_dataset, auxiliary = load_data(dataset_path)
    code_map, code_map_pretrain = code_maps['code_map'], code_maps['code_map_pretrain']
    (train_codes_data, valid_codes_data, test_codes_data) = (codes_dataset['train_codes_data'],
                                                             codes_dataset['valid_codes_data'],
                                                             codes_dataset['test_codes_data'])
    (train_hf_y, valid_hf_y, test_hf_y) = hf_dataset['train_hf_y'], hf_dataset['valid_hf_y'], hf_dataset['test_hf_y']

    (pretrain_codes_x, pretrain_codes_y, pretrain_y_h, pretrain_visit_lens) = pretrain_codes_data
    (train_codes_x, train_codes_y, train_y_h, train_visit_lens) = train_codes_data
    (valid_codes_x, valid_codes_y, valid_y_h, valid_visit_lens) = valid_codes_data
    (test_codes_x, test_codes_y, test_y_h, test_visit_lens) = test_codes_data
    (code_levels, code_levels_pretrain,
     subclass_maps, subclass_maps_pretrain,
     code_code_adj) = (auxiliary['code_levels'], auxiliary['code_levels_pretrain'],
                       auxiliary['subclass_maps'], auxiliary['subclass_maps_pretrain'],
                       auxiliary['code_code_adj'])

    op_conf = {
        'pretrain': False,
        'from_pretrain': True,
        'pretrain_path': './saved/hyperbolic/%s/sherbet_a/sherbet_pretrain' % dataset,
        'use_embedding_init': True,
        'use_hierarchical_decoder': True,
        'task': 'h',  # m: medical codes, h: heart failure
    }

    feature_model_conf = {
        'code_num': len(code_map_pretrain),
        'code_embedding_init': None,
        'adj': code_code_adj,
        'max_visit_num': train_codes_x.shape[1]
    }

    pretrain_model_conf = {
        'use_hierarchical_decoder': op_conf['use_hierarchical_decoder'],
        'subclass_dims': np.max(code_levels_pretrain, axis=0) if op_conf['use_hierarchical_decoder'] else None,
        'subclass_maps': subclass_maps_pretrain if op_conf['use_hierarchical_decoder'] else None,
        'output_dim': len(code_map_pretrain),
        'activation': None
    }

    task_conf = {
        'm': {
            'output_dim': len(code_map),
            'activation': None,
            'loss_fn': medical_codes_loss,
            'label': {
                'train': train_codes_y.astype(np.float32),
                'valid': valid_codes_y.astype(np.float32),
                'test': test_codes_y.astype(np.float32)
            },
            'evaluate_fn': EvaluateCodesCallBack
        },
        'h': {
            'output_dim': 1,
            'activation': 'sigmoid',
            'loss_fn': 'binary_crossentropy',
            'label': {
                'train': train_hf_y.astype(np.float32),
                'valid': valid_hf_y.astype(np.float32),
                'test': test_hf_y.astype(np.float32)
            },
            'evaluate_fn': EvaluateHFCallBack
        }
    }

    model_conf = {
        'use_hierarchical_decoder': False,
        'output_dim': task_conf[op_conf['task']]['output_dim'],
        'activation': task_conf[op_conf['task']]['activation']
    }

    hyper_params = {
        'code_embedding_size': 128,
        'hiddens': [64],
        'attention_size_code': 64,
        'attention_size_visit': 32,
        'patient_size': 64,
        'patient_activation': tf.keras.layers.LeakyReLU(),
        'pretrain_epoch': 1000,
        'pretrain_batch_size': 128,
        'epoch': 200,
        'batch_size': 32,
        'gnn_dropout_rate': 0.8,
        'decoder_dropout_rate': 0.17
    }

    if op_conf['use_embedding_init']:
        if op_conf['pretrain'] or (not op_conf['from_pretrain']):
            embedding_init = pickle.load(open('./saved/hyperbolic/%s_leaf_embeddings' % dataset, 'rb'))
            feature_model_conf['code_embedding_init'] = embedding_init
    sherbet_feature = SherbetFeature(feature_model_conf, hyper_params)

    if op_conf['pretrain']:
        pretrain_x = {
            'visit_codes': pretrain_codes_x,
            'visit_lens': pretrain_visit_lens
        }
        if op_conf['use_hierarchical_decoder']:
            pretrain_x['y_trues'] = pretrain_y_h
            pretrain_y = None
        else:
            pretrain_y = pretrain_codes_y.astype(np.float32)

        init_lr = 1e-2
        # split_val = [(20, 1e-3), (150, 1e-4), (500, 1e-5)]
        split_val = [(100, 1e-3)]
        lr_schedule_fn = lr_decay(total_epoch=hyper_params['epoch'], init_lr=init_lr, split_val=split_val)
        lr_scheduler = LearningRateScheduler(lr_schedule_fn)

        loss_fn = None if op_conf['use_hierarchical_decoder'] else medical_codes_loss
        sherbet_pretrain = Sherbet(sherbet_feature, pretrain_model_conf, hyper_params)
        sherbet_pretrain.compile(optimizer='rmsprop', loss=loss_fn)
        sherbet_pretrain.fit(x=pretrain_x, y=pretrain_y,
                           batch_size=hyper_params['pretrain_batch_size'], epochs=hyper_params['pretrain_epoch'],
                           callbacks=[lr_scheduler])

        sherbet_pretrain.save_weights(op_conf['pretrain_path'])
    else:
        if op_conf['from_pretrain']:
            sherbet_pretrain = Sherbet(sherbet_feature, pretrain_model_conf, hyper_params)
            sherbet_pretrain.load_weights(op_conf['pretrain_path'])

        x = {
            'visit_codes': train_codes_x,
            'visit_lens': train_visit_lens
        }
        valid_x = {
            'visit_codes': valid_codes_x,
            'visit_lens': valid_visit_lens
        }
        y = task_conf[op_conf['task']]['label']['train']
        valid_y = task_conf[op_conf['task']]['label']['valid']
        test_y = task_conf[op_conf['task']]['label']['test']

        # mimic3 m a, b, c
        # init_lr = 1e-2
        # split_val = [(20, 1e-3), (35, 1e-4), (100, 1e-5)]
        # mimic3 m d, e
        # init_lr = 1e-2
        # split_val = [(25, 1e-3), (40, 1e-4), (800, 1e-5)]
        # mimic3 h a, b, c
        init_lr = 1e-2
        split_val = [(25, 1e-3), (40, 1e-4), (45, 1e-5)]
        # split_val = [(10, 1e-3), (80, 1e-4), (100, 1e-5)]
        # mimic3 h d, e
        # init_lr = 1e-3
        # split_val = [(8, 1e-4), (10, 1e-5), (15, 1e-6)]
        # eicu m a, b, c
        # init_lr = 1e-2
        # split_val = [(50, 1e-3), (60, 1e-4), (100, 1e-5)]
        lr_schedule_fn = lr_decay(total_epoch=hyper_params['epoch'], init_lr=init_lr, split_val=split_val)

        test_codes_gen = DataGenerator([test_codes_x, test_visit_lens], shuffle=False, batch_size=128)

        loss_fn = task_conf[op_conf['task']]['loss_fn']
        lr_scheduler = LearningRateScheduler(lr_schedule_fn)
        test_callback = task_conf[op_conf['task']]['evaluate_fn'](test_codes_gen, test_y)

        sherbet = Sherbet(sherbet_feature, model_conf, hyper_params)
        sherbet.compile(optimizer='rmsprop', loss=loss_fn)
        history = sherbet.fit(x=x, y=y,
                            batch_size=hyper_params['batch_size'], epochs=hyper_params['epoch'],
                            callbacks=[lr_scheduler, test_callback])
        sherbet.summary()
