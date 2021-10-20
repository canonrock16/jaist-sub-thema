FROM python:3

RUN apt update -y
RUN apt install -y \
    openmpi-bin \
    libopenmpi-dev

WORKDIR /root/src

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt