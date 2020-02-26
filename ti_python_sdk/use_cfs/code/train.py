#!/usr/bin/env python

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import sys, os, time, argparse
print(tf.__version__)
# 例如 IO.py a,b,c
print(sys.argv)  # 有a,b,c被打印出来
# 打印所有环境变量
print(os.environ)

# A utility method to create a tf.data dataset from a Pandas Dataframe
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('target')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

if __name__ == '__main__':
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  # Data and model checkpoints directories
  parser.add_argument('--batch_size', type=int, default=32,
                      help='batch size, default 32')
  parser.add_argument(
    '--model_dir',
    type=str,
    required=False,
    help='The directory where the model will be stored.')
  parser.add_argument(
    '--ti_submit_directory',
    type=str,
    required=False,
    help='The directory where the code')
  args = parser.parse_args()

  dataframe = pd.read_csv('/opt/ml/input/data/training/heart.csv')
  dataframe.head()
  train, test = train_test_split(dataframe, test_size=0.2)
  train, val = train_test_split(train, test_size=0.2)
  print(len(train), 'train examples')
  print(len(val), 'validation examples')
  print(len(test), 'test examples')

  age = feature_column.numeric_column("age")
  age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
  thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])

  thal_one_hot = feature_column.indicator_column(thal)
  # Notice the input to the embedding column is the categorical column
  # we previously created
  thal_embedding = feature_column.embedding_column(thal, dimension=8)
  thal_hashed = feature_column.categorical_column_with_hash_bucket(
    'thal', hash_bucket_size=1000)

  crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)

  feature_columns = []

  # numeric cols
  for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

  # bucketized cols
  age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
  feature_columns.append(age_buckets)

  # indicator cols
  thal = feature_column.categorical_column_with_vocabulary_list(
    'thal', ['fixed', 'normal', 'reversible'])
  thal_one_hot = feature_column.indicator_column(thal)
  feature_columns.append(thal_one_hot)

  # embedding cols
  thal_embedding = feature_column.embedding_column(thal, dimension=8)
  feature_columns.append(thal_embedding)

  # crossed cols
  crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
  crossed_feature = feature_column.indicator_column(crossed_feature)
  feature_columns.append(crossed_feature)

  feature_layer = tf.keras.layers.DenseFeatures(feature_columns)
  batch_size = args.batch_size
  print("BATCH_SIZE", batch_size)

  train_ds = df_to_dataset(train, batch_size=batch_size)
  val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
  test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

  model = tf.keras.Sequential([
    feature_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
  ])

  model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'],
                run_eagerly=True)

  # Create a callback that saves the model's weights
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='/opt/ml/model/cp.ckpt',
                                                   save_weights_only=True,
                                                   verbose=1)
  model.fit(train_ds,
            validation_data=val_ds,
            epochs=5,
            callbacks=[cp_callback])
  loss, accuracy = model.evaluate(test_ds)
  print("Accuracy", accuracy)

  # time.sleep(60 * 2)
