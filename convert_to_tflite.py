import tensorflow as tf
import os
import cv2
import numpy as np

input_mean = 127.5
input_std = 127.5


def representative_dataset_gen():
    # https://stackoverflow.com/questions/58775848/tflite-cannot-set-tensor-dimension-mismatch-on-model-conversion
    # https://stackoverflow.com/questions/75267305/how-can-i-pass-my-proper-normalisation-mean-std-values-used-in-training-to-the-t
    # https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
    for f_name in os.listdir('images'):
        file_path = os.path.normpath(os.path.join('images', f_name))
        img = cv2.imread(file_path)
        img = cv2.resize(img, (320, 320))
        img = 2. * (img - np.min(img)) / np.ptp(img) - 1
        img = np.reshape(img, (1, 320, 320, 3))
        image = img.astype(np.float32)
        yield [image]


converter = tf.lite.TFLiteConverter.from_saved_model('inference_graph_adam/saved_model')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
# converter.inference_output_type = tf.int8  # alebo tf.uint8
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with open('detect.tflite', 'wb') as f:
    f.write(tflite_model)
