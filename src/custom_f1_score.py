import tensorflow as tf
from tensorflow.keras import backend as K

 
 #We have created our own F1Score class with the ability to choose a threshold, as well between micro, macro and weighted.

#Micro: It calculates the total true positives, false positives, and false negatives across all classes globally.
 # Classes with more frequent labels have a larger impact on the final score.
 # This is useful if the dataset is imbalanced, as it prioritizes the most frequent labels.

#Macro: Computes the F1Score for each label independently, and then averages them equally.
# Gives equal importance to each label, regardless of how frequently it occurs.
# This is useful when all labels are equally important, even if some labels are rare in some sense.

#Weighted: Computes the F1Score for each label independently, then averages them, weighted by the number of
# true instances for each label.
# It gives more importance to labels with more samples.
# It also balances the impact of label imbalance while still considering all labels.
# This is very useful when you want to account for label imbalance while evaluating the performance across all labels.

#NOTE: Because of the label imbalance in the dataset, it is better to use the weighted average if no sample weights are applied.
# Micro could still be used but it leans more towards prioritizing the most frequent labels. Therefore, after identifying the respective weights
# that will make the dataset "balanced" during training, it is beneficial to use the macro average F1Score instead. Since during training (after applying the sample weights)
# the dataset will be somewhat balanced, then using the macro average to give equal importance to each label is beneficial in this case as indeed now all labels are equally important during training.

class F1Score(tf.keras.metrics.Metric):
  def __init__(self, num_classes, threshold=0.5, average='micro', name='f1_score', **kwargs):
    super(F1Score, self).__init__(name=name, **kwargs)
    self.num_classes = num_classes
    self.threshold = threshold
    if average not in ['micro', 'macro', 'weighted']:
      raise ValueError("Invalid average type. Allowed values are: ['micro', 'macro', 'weighted']")
    self.average = average

    # Initialize state variables
    self.true_positives = self.add_weight(name='true_positives', shape=(self.num_classes,), initializer='zeros')
    self.false_positives = self.add_weight(name='false_positives', shape=(self.num_classes,), initializer='zeros')
    self.false_negatives = self.add_weight(name='false_negatives', shape=(self.num_classes,), initializer='zeros')
    self.weights = self.add_weight(name='weights', shape=(self.num_classes,), initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    # Convert predictions to binary based on the threshold
    y_pred = tf.cast(y_pred >= self.threshold, tf.float32)
    y_true = tf.cast(y_true, tf.float32)

    # Compute true positives, false positives, false negatives per class
    tp = tf.reduce_sum(y_true * y_pred, axis=0)
    fp = tf.reduce_sum((1 - y_true) * y_pred, axis=0)
    fn = tf.reduce_sum(y_true * (1 - y_pred), axis=0)

    # Update state variables
    self.true_positives.assign_add(tp)
    self.false_positives.assign_add(fp)
    self.false_negatives.assign_add(fn)
    self.weights.assign_add(tf.reduce_sum(y_true, axis=0))  # Sum of actual positives per class for weighted average

  def result(self):
    precision = self.true_positives / (self.true_positives + self.false_positives + K.epsilon())
    recall = self.true_positives / (self.true_positives + self.false_negatives + K.epsilon())
    f1 = 2 * precision * recall / (precision + recall + K.epsilon())

    if self.average == 'micro':
      tp_sum = tf.reduce_sum(self.true_positives)
      fp_sum = tf.reduce_sum(self.false_positives)
      fn_sum = tf.reduce_sum(self.false_negatives)
      precision_micro = tp_sum / (tp_sum + fp_sum + K.epsilon())
      recall_micro = tp_sum / (tp_sum + fn_sum + K.epsilon())
      f1_micro = 2 * precision_micro * recall_micro / (precision_micro + recall_micro + K.epsilon())
      return f1_micro
    elif self.average == 'macro':
      return tf.reduce_mean(f1)
    elif self.average == 'weighted':
      weights = self.weights / (tf.reduce_sum(self.weights) + K.epsilon())
      return tf.reduce_sum(f1 * weights)
    else:
      return f1  # If no averaging, return per-class f1 scores

  def reset_states(self):
    # Reset state variables for the next epoch
    for var in self.variables:
      var.assign(tf.zeros_like(var))