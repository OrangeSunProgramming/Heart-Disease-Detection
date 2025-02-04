import os
import numpy as np
import pandas as pd
import tensorflow as tf
import data_augmentation as da
from sklearn.model_selection import train_test_split


 #Since we want to ensure reproducibility, we need to set the seed at the
# beginning of the project. Neural networks rely on the initialization of weights and biases,
# shuffling of data before each epoch, and splitting the dataset into training and validation sets.
# When we set the seed, we ensure that these random processes produce the same outputs every time the code runs.

seed = 42
tf.random.set_seed(seed)

#Since the model can be complicated and give us ResourceExhaustedError, we set a batch size of 16
# instead of 32 or higher in order to constraint the training phase and successfully get outputs.
# This ensures the model is learning in sets of 16 batches and produce results as we expect.
# We personally don't want the batch size to be very big (to not constraint the model) or very low
# since it can negatively impact the learning process of the model. Setting batch sizes that are multiples
# of eight seems to be ideal.

batch_size = 16

#Since we are dealing with a multi-label task, we are organizing the dataset into
# six columns, where the first column is the audio and the other columns are labeled [AS, AR, MR, MS, N] respectively.
# Each of those columns contain numbers between 0 and 1, which indicates if the respective audio belongs to the label or not.
# Since this is a multi-label task, an audio (heart beats in this case) can belong to multiple labels since heart beats can have rithms
# similar under different labels. For example, a heart beat can be classified as MR (Mitral Regurgitation) while also being classified as
# N (Normal) because it exhibits rythmic activities similar to both label cases, which could be interpreted as someone that has a relatively normal Mitral Regurgitation
# compare to other patients with a very more serious Mitral Regurgitation.

df = pd.read_csv("drive/MyDrive/heart_disease_multi_label/train.csv")
data_d = []

for index in range(108):
  for j in range(8):
    d = []
    recording = df[f"recording_{j+1}"][index]
    d.append(f"{recording}.wav")
    for label in ["AS", "AR", "MR", "MS", "N"]:
      heart_binary_labels = df[label][index]
      d.append(heart_binary_labels)
    data_d.append(d)

audio_label_dataset = pd.DataFrame(data_d, columns=["audio recording", "AS", "AR", "MR", "MS", "N"])

#Data folder path
data_folder = "drive/MyDrive/heart_disease_multi_label"


#Since we are going to use a CNN, it will be better to turn the audios into spectrogram images that a
# convolutional neural network can work and learn from the details in the spectrograms. The CNN architecture will
# use these spectrograms to learn low and high level features from the audios and make the necessary connections to have a
# well-performance model.

def create_mel_spectrogram(audio_path, augment=False):
    audio = tf.io.read_file(audio_path)
    audio, sample_rate = tf.audio.decode_wav(audio)
    audio = tf.squeeze(audio, axis=-1)

    '''if augment:
        audio = da.augment_audio(audio, sample_rate)'''

    sample_rate = 4000
    num_mel_bins = 128
    frame_length = 625
    frame_step = 313

    spectrogram = tf.signal.stft(audio, frame_length=frame_length, frame_step=frame_step, pad_end=True)
    magnitude = tf.abs(spectrogram)

    mel_filter = tf.signal.linear_to_mel_weight_matrix(num_mel_bins, magnitude.shape[-1], sample_rate, 0, sample_rate / 2)
    mel_spectrogram = tf.tensordot(magnitude, mel_filter, 1)
    mel_spectrogram.set_shape(magnitude.shape[:-1].concatenate(mel_filter.shape[-1:]))

    mel_spectrogram_db = 10 * tf.math.log(tf.maximum(mel_spectrogram, 1e-6)) / tf.math.log(10.0)

    '''if augment:
        mel_spectrogram_db = da.spec_augment(mel_spectrogram_db)'''

    return mel_spectrogram_db


#Preparing the data
spectrograms = []
labels = []
target_shape = (128, 128)

for index, row in audio_label_dataset.iterrows():
    audio_file = os.path.join(data_folder, "train", row["audio recording"])
    if os.path.isfile(audio_file):
        #Original sample
        mel_spectrogram = create_mel_spectrogram(audio_file, augment=False)
        mel_spectrogram_resized = tf.image.resize(mel_spectrogram[..., tf.newaxis], [target_shape[0], target_shape[1]])
        spectrograms.append(mel_spectrogram_resized.numpy())  # Converting to NumPy
        labels.append(row[['AS', 'AR', 'MR', 'MS', 'N']].values)

        #Augmented sample
        '''mel_spectrogram_augmented = create_mel_spectrogram(audio_file, augment=True)
        mel_spectrogram_resized_augmented = tf.image.resize(mel_spectrogram_augmented[..., tf.newaxis], [target_shape[0], target_shape[1]])
        spectrograms.append(mel_spectrogram_resized_augmented.numpy())  # Converting to NumPy
        labels.append(row[['AS', 'AR', 'MR', 'MS', 'N']].values)'''
    else:
        print(f"File not found: {audio_file}. Skipping.")


# Converting to NumPy arrays
X = np.array(spectrograms)
y = np.array(labels)

# Normalizing the spectrograms
X = (X - np.mean(X)) / np.std(X)
X = X[..., np.newaxis]  # Adding channel dimension

# Splitting the dataset
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

y_train = np.array(y_train, dtype=np.float32)
y_val = np.array(y_val, dtype=np.float32)