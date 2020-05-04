# TensorFlow Lite inference with DeepLab v3

This repository contains a Python script to infer semantic segmentation from an image using the pre-trained TensorFlow Lite DeepLabv3 model trained on the PASCAL VOC or ADE20K datasets. 

It also includes instruction to generate a TFLite model with various degrees of quantization that is trained on the ADE20K dataset.

## Models

DeepLab v3 trained on the PASCAL VOC dataset is provided [here](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite) and is already in `.tflite` format.

DeepLab v3 trained on the ADE20K dataset is available [here](http://download.tensorflow.org/models/deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz) ([source](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md)) but has to be converted.

## Model conversion

The frozen inference graph will first have to be converted to a SavedModel, then it can be converted to a TFLite flatbuffer.

Ensure you have `python3-venv` installed.

Clone this repo, download and extract the frozen inference graph and weights

```bash
git clone https://github.com/kmfrick/TFLite_DeepLabv3_Inference
cd TFLite_DeepLabv3_Inference
wget http://download.tensorflow.org/models/deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz
tar xf deeplabv3_mnv2_ade20k_train_2018_12_03.tar.gz
```

Create a python virtualenv and install Tensorflow:

```bash
python3 -m venv venv
deactivate # This will fail if you don't have any venv activated, don't worry and try not to copy and paste code next time
source ./venv/bin/activate
python -m pip install -r requirements.txt
```

Then you can run conversion to a SavedModel:

```bash
python ./convert_to_saved.py
```

You will now have a `saved` subfolder in the current directory where the newly created SavedModel will reside.

You can now convert the SavedModel to a TFLite flatbuffer

```bash
python ./convert_to_tflite.py
```

This will generate three files, `deeplabv3_mnv2_ade20k_dyr.tflite`, `deeplabv3_mnv2_ade20k_f16.tflite`, `deeplabv3_mnv2_ade20k_f32.tflite`, respectively with dynamic range, float16 and no quantization.

## Inference

In the virtualenv with TF2 installed, run one of the scripts

```bash
python inference_pascal.py <ModelFile> <ImageName> <ColorMap>
```


```bash
python inference_ade20k.py <ModelFile> <ImageName> <ColorMap>
```

Where `<ColorMap>` is `pascal.png` or `ade20k.png`.

## Related publications

If you use this in a project of yours, please cite the original paper as such:

```tex
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}
```

