import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import argparse
import numpy as np
import os

IMAGE_SIZE = 28
LABELS_SIZE = 10
HIDDEN_SIZE = 2048
INPUT_FEATURE = 'image'

def raw_input_fn(dataset):
    return dataset.images, dataset.labels.astype(np.int32)

def serve_input_fn():
    reciever_tensors = {
        INPUT_FEATURE: tf.placeholder(tf.float32, [None, None, None, 1]),
    }

    features = {
        INPUT_FEATURE: tf.image.resize_images(reciever_tensors[INPUT_FEATURE], [IMAGE_SIZE, IMAGE_SIZE]),
    }
    return tf.estimator.export.ServingInputReceiver(receiver_tensors=reciever_tensors,
                    features=features)

def train(data_dir, model_dir, checkpoints_dir, train_steps, batch_size):
    # Set logging level
    tf.logging.set_verbosity(tf.logging.INFO)
    
    # Read Mnist dataset
    mnist = input_data.read_data_sets(data_dir, one_hot=False)

    # Define training checkpoints config
    checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs = 30,
        keep_checkpoint_max = 100,    
    )
    
    # Define training classifier
    feature_columns = [tf.feature_column.numeric_column(INPUT_FEATURE, shape=[IMAGE_SIZE, IMAGE_SIZE])]
    classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[HIDDEN_SIZE],
                                            n_classes=LABELS_SIZE,
                                            optimizer=tf.train.AdamOptimizer(),
                                            config=checkpointing_config,
                                            model_dir=checkpoints_dir)

    # Define training model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={INPUT_FEATURE : raw_input_fn(mnist.train)[0]},
        y=raw_input_fn(mnist.train)[1],
        num_epochs=None,
        batch_size=batch_size,
        shuffle=True
    )

    classifier.train(input_fn=train_input_fn, steps=train_steps)

    # Save training Model
    model_dir = classifier.export_savedmodel("/opt/ml/model", serve_input_fn)
    print("model path: %s" % model_dir)
    
if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        '--data_dir',
        default='/opt/ml/input/data/training',
        type=str,
        help='the training data directory.')

    args_parser.add_argument(
        '--model_dir',
        default='/opt/ml/model',
        type=str,
        help='the model directory.')

    args_parser.add_argument(
        '--checkpoints_dir',
        default='/opt/ml/checkpoints',
        type=str,
        help='the checkpoints directory.')

    args_parser.add_argument(
        '--train-steps',
        type=int,
        default=100,
        help='train steps number.')

    args_parser.add_argument(
        '--batch-size',
        type=int,
        default=os.environ["batch_size"],
        help='train batch size.')
        
    args = args_parser.parse_args()
    train(**vars(args))