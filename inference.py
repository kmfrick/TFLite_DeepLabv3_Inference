#!/usr/bin/env python

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import tensorflow as tf
import cv2
import sys

if len(sys.argv) < 2:
    print("Usage: " + sys.argv[0] + " [test_image]")
    quit()

interpreter = tf.lite.Interpreter(model_path="deeplabv3_257_mv_gpu.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = cv2.normalize(cv2.imread(sys.argv[1]).astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
input_data = cv2.resize(input_data, (input_shape[1], input_shape[2]))
input_data = tf.reshape(input_data, input_shape)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
output_classes = np.uint8(tf.argmax(output_data, axis=3)[0])
output_classes_rgb = cv2.cvtColor(output_classes, cv2.COLOR_GRAY2RGB)
colormap = cv2.imread("pascal.png").astype(np.uint8)
output_img = cv2.LUT(output_classes_rgb, colormap)
cv2.imshow("test_image", output_img)
cv2.waitKey(0)


