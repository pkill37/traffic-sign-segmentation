# traffic-sign-segmentation

## Install

If you have Python 3.6 you can work directly on your machine within a Python virtual environment:

```
$ python -m venv venv
$ pip install -r requirements.txt
```

Alternatively you can run inside a Docker container:

```
$ docker run -it -p 6006:6006 -p 8888:8888 -v $(pwd):/tmp -w /tmp -u $(id -u):$(id -g) tensorflow/tensorflow:nightly-py3
```

## Usage

Train the model:

```
$ python src/train.py
```

Evaluate the model:

```
$ python src/test.py
```
