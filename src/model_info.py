import sample_weights as sw
import data_preparation as dp
import pprint

def dataset_info():
  info = {
    "Length of X_train": len(dp.X_train),
    "Length of X_val": len(dp.X_val),
    "Length of y_train": len(dp.y_train),
    "Length of y_val": len(dp.y_val),
    "Length of X": len(dp.X),
    "Length of y": len(dp.y),
    "Length of audio_label_dataset": len(dp.audio_label_dataset)
}
  return pprint.pprint(info)

def sample_weights_info():
  info = {
    "Length of sample_weights": len(sw.sample_weights),
    "Length of sample_weights_train": len(sw.sample_weights_train),
    "Length of sample_weights_val": len(sw.sample_weights_val),
    "Effective Number of Samples": sw.effective_samples,
    "Original Counts": sw.original_counts,
    "Balance Ratios": sw.balance_ratios
}

  return pprint.pprint(info)