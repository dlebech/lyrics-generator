FROM tensorflow/tensorflow:1.14.0-gpu-py3

# Only need the base requirements (i.e. excluding Tensorflow)
COPY requirements_base.txt .
RUN pip install -r requirements_base.txt
