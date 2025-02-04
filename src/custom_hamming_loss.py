import tensorflow as tf

#NOTE: as stated in (https://github.com/tensorflow/addons), TensorFlow Addons (TFA) ended development and
# will not introduce any new feautures. The end of life of TFA was scheduled and carried out in May 2024.
# For these reasons, we wrote our own F1Score above and HammingLoss here below.

#Hamming Loss: measures the fraction of incorrectly predicted labels to the total number of labels
# in a multi-label classification problem. Therefore, if possible, we desire the Hamming Loss to be as close to
# zero as possible. If the Hamming Loss is 0, then it means that every label for every sample is correctly predicted.

class HammingLoss(tf.keras.metrics.Metric):
  def __init__(self, name='hamming_loss', **kwargs):
    super(HammingLoss, self).__init__(name=name, **kwargs)
    self.total = self.add_weight(name='total', initializer='zeros')
    self.count = self.add_weight(name='count', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_pred = tf.round(y_pred)
    values = tf.not_equal(y_true, y_pred)
    errors = tf.reduce_sum(tf.cast(values, tf.float32), axis=1)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, tf.float32)
      errors = tf.multiply(errors, sample_weight)
    self.total.assign_add(tf.reduce_sum(errors))
    self.count.assign_add(tf.cast(tf.size(errors), tf.float32))

  def result(self):
    return self.total / (self.count + tf.keras.backend.epsilon())

  def reset_states(self):
    self.total.assign(0)
    self.count.assign(0)