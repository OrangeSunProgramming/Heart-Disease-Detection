import librosa
import tensorflow as tf

#Since the dataset is imbalanced and is relatively small, then it would ideally be beneficial to augment the dataset.
# Here we use time stretch, pitch shift, and we add a background noise. These augmentations are implemented in order to help the model
# generalize much better or be exposed to different somewhat realistic audio examples. After I have personally tried many times to run the model
# where the dataset has been augmented, I have seen the model downgrade in performance compared to when its run (trained) on the non-augmented dataset.
# For that reason I decided not to augment the dataset since it seems to be introducing more noise than expected. I have commented the code here so you can see
# the data augmentation implementation which could help you out in other projects.

def time_stretch(audio, rate=1.1):
  return librosa.effects.time_stretch(audio.numpy(), rate=rate)

def pitch_shift(audio, sample_rate, n_steps=2):
  return librosa.effects.pitch_shift(audio.numpy(), sr=int(sample_rate), n_steps=n_steps)

def add_background_noise(audio, noise_factor=0.005):
  noise = tf.random.normal(shape=tf.shape(audio), mean=0.0, stddev=noise_factor, dtype=tf.float32)
  return audio + noise

#SpecAugment
def spec_augment(mel_spectrogram):
  time_mask_param = 10
  freq_mask_param = 10
  num_time_masks = 1
  num_freq_masks = 1

  for _ in range(num_time_masks):
    t = tf.random.uniform([], minval=0, maxval=time_mask_param, dtype=tf.int32)
    t0 = tf.random.uniform([], minval=0, maxval=tf.shape(mel_spectrogram)[1] - t, dtype=tf.int32)
    mel_spectrogram = tf.concat([mel_spectrogram[:, :t0], tf.zeros_like(mel_spectrogram[:, t0:t0 + t]), mel_spectrogram[:, t0 + t:]], axis=1)

  for _ in range(num_freq_masks):
    f = tf.random.uniform([], minval=0, maxval=freq_mask_param, dtype=tf.int32)
    f0 = tf.random.uniform([], minval=0, maxval=tf.shape(mel_spectrogram)[0] - f, dtype=tf.int32)
    mel_spectrogram = tf.concat([mel_spectrogram[:f0, :], tf.zeros_like(mel_spectrogram[f0:f0 + f, :]), mel_spectrogram[f0 + f:, :]], axis=0)

  return mel_spectrogram


def augment_audio(audio, sample_rate):
  audio = tf.convert_to_tensor(time_stretch(audio, rate=1.1))
  audio = tf.convert_to_tensor(pitch_shift(audio, sample_rate, n_steps=2))
  audio = add_background_noise(audio)
  return audio