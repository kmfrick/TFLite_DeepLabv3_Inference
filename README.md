# TensorFlow Lite inference with DeepLab v3

This repository contains a Python script to infer semantic segmentation from an image using the pre-trained TensorFlow Lite [DeepLab v3](https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite) model. 

If you use this in a project of yours, please cite the original paper as such:

```tex
@inproceedings{deeplabv3plus2018,
  title={Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation},
  author={Liang-Chieh Chen and Yukun Zhu and George Papandreou and Florian Schroff and Hartwig Adam},
  booktitle={ECCV},
  year={2018}
}
```

To run inference, clone this repo

```bash
git clone https://github.com/kmfrick/TFLite_DeepLabv3_Inference
cd TFLite_DeepLabv3_Inference
```

install TensorFlow and OpenCV (preferably in a [virtualenv](https://docs.python.org/3/library/venv.html))

```bash
python -m venv venv
source venv/bin/activate
PIP_REQUIRE_VIRTUALENV=true pip install -r requirements.txt
```

download the pre-trained model in the same directory as the repository

```bash
wget https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/deeplabv3_257_mv_gpu.tflite
```

run the script

```bash
python inference.py <ImageName>
```

