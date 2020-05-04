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

if len(sys.argv) < 4:
    print("Usage: " + sys.argv[0] + " model_path image_path colormap_path")
    quit()

model_path = sys.argv[1]
image_path = sys.argv[2]
colormap_path = sys.argv[3]

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
img = cv2.imread(image_path)
img = cv2.resize(img, (input_shape[1], input_shape[2]))
input_data = cv2.normalize(img.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
input_data = tf.reshape(input_data, input_shape)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
output_classes = np.uint8(tf.argmax(output_data, axis=3)[0])
output_classes_rgb = cv2.cvtColor(output_classes, cv2.COLOR_GRAY2RGB)
colormap = cv2.imread(colormap_path).astype(np.uint8)
output_img = cv2.LUT(output_classes_rgb, colormap)
h = int(output_img.shape[0]/2)
w = int(output_img.shape[1]/2)
output_img = output_img[:h, :w] # Slice the upper left quarter of the result image
cv2.imshow("test_image", output_img)
cv2.waitKey(0)


