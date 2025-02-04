import numpy as np
import data_preparation as dp
from sklearn.model_selection import train_test_split

#Since we have an imbalance in the dataset, we will apply different weights to each of the audio samples and
# their labels during training. In the dataset there are labels with fewer audio samples than others. For example,
# N (normal) has the least ammount of audio samples than any other label. What this means is that N's contribution to the model's training
# will have a fewer effect than the other labels which could be really bad since an imbalance dataset doesn't let the model generalize well to all possible areas.
# Therefore, applying different weights, that is, applying heavier weights to the underrepresented labels and balancing it with the other weights for the other labels,
# well make the model be trained on a more balanced dataset. That is, the model will pay attention more to the heavier weighted labels during training, which means those labels will contribute more than the others
# during the training phase while the less weighted label will still contribute but not as much that overshadows the others, therefore creating a balance across all areas which helps the model generalize better.

def compute_sample_weights(df, labels):
    label_counts = {label: df[label].sum() for label in labels}
    total_samples = len(df)
    label_weights = {
        label: total_samples / (len(labels) * count) if count > 0 else 0
        for label, count in label_counts.items()
    }
    sample_weights = []
    for _, row in df.iterrows():
        weight = sum(label_weights[label] for label in labels if row[label] == 1)
        sample_weights.append(weight)
    return np.array(sample_weights)


labels = ["AS", "AR", "MR", "MS", "N"]
sample_weights = compute_sample_weights(dp.audio_label_dataset, labels)

sample_weights = np.delete(sample_weights, np.where(dp.audio_label_dataset["audio recording"] == "MD_085_sit_Tri.wav")[0])

print("Missing audio file deleted!")


def effective_number_of_samples(df, sample_weights, labels):
    effective_samples = {label: 0 for label in labels}
    for i, row in df.iterrows():
      if i < len(sample_weights):
        for label in labels:
          if row[label] == 1:
            effective_samples[label] += sample_weights[i]
    return effective_samples

effective_samples = effective_number_of_samples(dp.audio_label_dataset[dp.audio_label_dataset["audio recording"] != "MD_085_sit_Tri.wav"], sample_weights, labels)

original_counts = {label: dp.audio_label_dataset[dp.audio_label_dataset["audio recording"] != "MD_085_sit_Tri.wav"][label].sum() for label in labels}

balance_ratios = {label: effective_samples[label] / original_counts[label] for label in labels}


#Just as the dataset was split between training and testing, we are also going to
# split the sample weights accordingly with the same test size. We will apply the
# sample_weight_train during the training phase but it is not neccessary for us to apply
# the sample_weights_val when we evaluate the model's performance on the validation dataset after training.
# In my case, I will apply it.

sample_weights_train, sample_weights_val = train_test_split(sample_weights, test_size=0.2, random_state=42, stratify=dp.y)

sample_weights_train = sample_weights_train
sample_weights_val = sample_weights_val