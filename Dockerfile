FROM tensorflow/tensorflow:2.5.0-gpu

WORKDIR /tf/src

# Only need the base requirements (i.e. excluding Tensorflow)
COPY requirements_base.txt .
RUN pip install pip==21.1
RUN pip install -r requirements_base.txt
