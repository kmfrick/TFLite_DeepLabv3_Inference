#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

export_dir = "./deeplabv3_mnv2_ade20k_train_2018_12_03_saved"
graph_pb = "./deeplabv3_mnv2_ade20k_train_2018_12_03/frozen_inference_graph.pb"

builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

with tf.gfile.GFile(graph_pb, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

sigs = {}

with tf.Session(graph=tf.Graph()) as sess:
    # name="" is important to ensure we don't get spurious prefixing
    tf.import_graph_def(graph_def, name="")
    g = tf.get_default_graph()
    inp = g.get_tensor_by_name("ImageTensor:0")
    out = g.get_tensor_by_name("ResizeBilinear_2:0")
    print(out.get_shape())
    out = tf.argmax(out, axis=3)
    out = tf.cast(out, dtype=tf.uint8)
    out = tf.squeeze(out)
    print(out.name)


    sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
        tf.saved_model.signature_def_utils.predict_signature_def(
            {"in": inp}, {"out": out})

    builder.add_meta_graph_and_variables(sess,
                                         [tag_constants.SERVING],
                                         signature_def_map=sigs)

builder.save()
