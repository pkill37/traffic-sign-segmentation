# traffic-sign-segmentation

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
