FROM tensorflow/tensorflow:2.1.0-gpu-py3

WORKDIR /tf/src

# Only need the base requirements (i.e. excluding Tensorflow)
COPY requirements_base.txt .
RUN pip install -r requirements_base.txt
