import tensorflow as tf
from tensorflow.keras.metrics import Metric

class F1ScoreMetric(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1ScoreMetric, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_labels = tf.argmax(y_true, axis=1)
        y_pred_labels = tf.argmax(y_pred, axis=1)
        self.precision.update_state(y_true_labels, y_pred_labels, sample_weight)
        self.recall.update_state(y_true_labels, y_pred_labels, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return tf.cond(
            tf.math.equal(p + r, 0.0),
            lambda: 0.0,
            lambda: 2 * (p * r) / (p + r)
        )

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


class MatthewsCorrelationCoefficient(Metric):
    def __init__(self, name='mcc', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32))
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))
        self.tp.assign_add(tp)
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = tf.sqrt(
            (self.tp + self.fp) *
            (self.tp + self.fn) *
            (self.tn + self.fp) *
            (self.tn + self.fn)
        )
        return tf.where(denominator == 0, 0.0, numerator / denominator)

    def reset_states(self):
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)
