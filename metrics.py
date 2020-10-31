import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def f1(y_true_hot, y_pred, metrics='weighted'):
    result = np.zeros_like(y_true_hot)
    for i in range(len(result)):
        true_number = np.sum(y_true_hot[i] == 1)
        result[i][y_pred[i][:true_number]] = 1
    return f1_score(y_true=y_true_hot, y_pred=result, average=metrics)


def top_k_prec_recall(y_true_hot, y_pred, ks):
    a = np.zeros((len(ks), ))
    r = np.zeros((len(ks), ))
    for pred, true_hot in zip(y_pred, y_true_hot):
        true = np.where(true_hot == 1)[0].tolist()
        t = set(true)
        for i, k in enumerate(ks):
            p = set(pred[:k])
            it = p.intersection(t)
            a[i] += len(it) / k
            r[i] += len(it) / min(k, len(t))
    return a / len(y_true_hot), r / len(y_true_hot)


class EvaluateCallBack(Callback):
    def __init__(self, data_gen, y):
        super().__init__()
        self.data_gen = data_gen
        self.y = y

    def on_epoch_end(self, epoch, logs=None):
        raise NotImplementedError


class EvaluateCodesCallBack(EvaluateCallBack):
    def on_epoch_end(self, epoch, logs=None):
        step_size = len(self.data_gen)
        preds = []
        for i in range(step_size):
            batch_codes_x, batch_visit_lens = self.data_gen[i]
            output = self.model(inputs={
                'visit_codes': batch_codes_x,
                'visit_lens': batch_visit_lens
            }, training=False)
            # logits = tf.math.sigmoid(output)
            logits = output
            pred = tf.argsort(logits, axis=-1, direction='DESCENDING')
            preds.append(pred.numpy())
        preds = np.vstack(preds)
        f1_score = f1(self.y, preds)
        prec, recall = top_k_prec_recall(self.y, preds, ks=[10, 20, 30, 40])
        print('\t', 'f1_score:', f1_score, '\t', 'top_k_recall:', recall)


class EvaluateHFCallBack(EvaluateCallBack):
    def on_epoch_end(self, epoch, logs=None):
        step_size = len(self.data_gen)
        preds, outputs = [], []
        for i in range(step_size):
            batch_codes_x, batch_visit_lens = self.data_gen[i]
            output = self.model(inputs={
                'visit_codes': batch_codes_x,
                'visit_lens': batch_visit_lens
            }, training=False)
            outputs.append(tf.squeeze(output).numpy())
            pred = tf.squeeze(tf.cast(output > 0.5, tf.int32))
            preds.append(pred.numpy())
        outputs = np.concatenate(outputs)
        preds = np.concatenate(preds)
        auc = roc_auc_score(self.y, outputs)
        f1_score_ = f1_score(self.y, preds)
        print('\t', 'auc:', auc, '\t', 'f1_score:', f1_score_)
