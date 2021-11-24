import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pickle
from iotai_sensor_classification.plot_util import plot_confusion_matrix
from iotai_sensor_classification.motion_records import parse_motions


def make_gesture_model(input_shape, num_gestures):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(50, activation='relu', input_shape=input_shape))  # relu is used for performance
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(15, activation='relu'))
    model.add(tf.keras.layers.Dense(num_gestures,
                                    activation='softmax'))  # softmax is used, because we only expect one gesture to occur per input
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


def test_predictions(model, x_test, y_test, params, output_dir=None):
    """Test model performance and write TFlite models """
    motion_summaries = pd.read_csv(os.path.join(params['output_dir'], 'motion_summaries.csv'))
    predictions = model.predict(x_test)
    prediction_indexes = np.argmax(predictions, axis=1)
    prediction_words = []
    for pre_ind in prediction_indexes:
        prediction_words.append(motion_summaries['gesture'][pre_ind])
    y_test_indexes = np.argmax(y_test, axis=1)
    y_test_words = []
    for y_ind in y_test_indexes:
        y_test_words.append(motion_summaries['gesture'][y_ind])
    test_matrix = metrics.confusion_matrix(y_true=y_test_words, y_pred=prediction_words)
    plot_confusion_matrix(test_matrix, classes=motion_summaries['gesture'], title="Motion confusion matrix",
                          output_path=os.path.join(params['output_dir'], "{}_confusionMatrix.png".format(
                              params['model_name'])))
    accuracy = metrics.accuracy_score(y_test_words, prediction_words)
    print(test_matrix)
    print("test accuracy={}".format(accuracy))

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.experimental_new_converter = False
    tflite_model = converter.convert()
    with open(os.path.join(params['output_dir'], "{}.tflite".format(params['model_name'])), 'wb') as wp:
        wp.write(tflite_model)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = False

    def representative_dataset_generator():
        for value in x_test:
            yield [np.array(value, dtype=np.float32, ndmin=len(motion_summaries))]
    converter.representative_dataset = representative_dataset_generator
    quantized_model = converter.convert()
    with open(os.path.join(params['output_dir'], "{}.quantized".format(params['model_name'])), "wb") as mp:
        mp.write(quantized_model)


def train(params):
    motion_input, motion_output, motion_summaries, motion_lookup = parse_motions(params['motion_dir'], params['samples_per_gesture'],
                                                                  params['output_dir'])
    motion_summaries.to_csv(os.path.join(params['output_dir'], "motion_summaries.csv"))
    with open(os.path.join(params['output_dir'], 'motion_lookup.pickle'), 'wb') as wp:
        pickle.dump(motion_lookup, wp)

    # np.save(os.path.join(params['output_dir'], "motion_input.npy"), motion_input)
    # np.save(os.path.join(params['output_dir'], "motion_output.npy"), motion_output)
    train_input, val_test_input, train_output, val_test_output = train_test_split(motion_input, motion_output,
                                                                                  test_size=params['test_split'])
    val_input, test_input, val_output, test_output = train_test_split(val_test_input, val_test_output, test_size=0.45)
    np.save(os.path.join(params['output_dir'], "test_input.npy"), test_input)
    np.save(os.path.join(params['output_dir'], "test_output.npy"), test_output)

    num_gestures = len(motion_summaries['gesture'])
    model = make_gesture_model(params['input_shape'], num_gestures)
    log_dir = os.path.join(params['output_dir'], 'logs')
    os.makedirs(log_dir, exist_ok=True)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(train_input, train_output, epochs=params['epochs'], batch_size=params['batch_size'],
                        validation_data=(val_input, val_output), callbacks=[tensorboard_callback])
    model.save(os.path.join(params['output_dir'], params["model_name"]))


if __name__ == '__main__':
    import argparse
    params = {}
    params['motion_dir'] = '/home/ktdiedrich/Arduino/karduino/motionCapture/motions'
    params['output_dir'] = '/home/ktdiedrich/output/motionCapture'
    params['samples_per_gesture'] = 119
    params['factors'] = 6
    params['input_shape'] = (params['samples_per_gesture'], params['factors'])
    params['test_split'] = 0.30
    params['train'] = False
    params['test'] = False
    params['model_name'] = "gesture_model"
    params['epochs'] = 600
    params['batch_size'] = 10

    parser = argparse.ArgumentParser(description='Load images from ground truth sub directories for model training.')
    parser.add_argument("--train", "-t", action="store_true", default=params['train'], required=False,
                        help="train, default {}".format(params['train']))
    parser.add_argument("--test", "-T", action="store_true", default=params['test'], required=False,
                        help="test predictions, default {}".format(params['test']))
    args = parser.parse_args()
    params['train'] = args.train
    params['test'] = args.test

    SEED = 1337
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    os.makedirs(params['output_dir'], exist_ok=True)

    if params['train']:
        train(params)
    if params['test']:
        model = tf.keras.models.load_model(os.path.join(params['output_dir'], params['model_name']))
        test_input = np.load(os.path.join(params['output_dir'], "test_input.npy"))
        test_output = np.load(os.path.join(params['output_dir'], "test_output.npy"))
        test_predictions(model, test_input, test_output, params, params['output_dir'])
    print('fin')