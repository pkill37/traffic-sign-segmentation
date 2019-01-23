# traffic-sign-segmentation

Binary image segmentation of traffic sign images (i.e. each pixel is classified as either background or foreground) which we approach from a transfer learning standpoint by taking a VGG16 model trained on ImageNet and fine-tuning it to our dataset (a collection of 1000 original images of traffic signs from the city of Zagreb which we manually labeled).

## Install

If you have Python 3.6 you can work directly on your machine within a Python virtual environment:

```
$ python -m venv venv
$ pip install -r requirements.txt
```

Alternatively you can run inside a Docker container:

```
$ docker run -it -p 6006:6006 -v $(pwd):/tmp -w /tmp tensorflow/tensorflow:latest-py3 bash
```

## Usage

Train the model:

```
$ python src/train.py
```

Monitor training:

```
$ rm -f ./out/tensorboard/* && tensorboard --logdir ./out/tensorboard
```

Evaluate the model:

```
$ python src/test.py
```
