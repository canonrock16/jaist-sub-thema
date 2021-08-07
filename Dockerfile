FROM python:3

WORKDIR /root/src

RUN apt update -y
RUN apt install -y \
	openmpi-bin \
	libopenmpi-dev

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# CMD [ "python", "./your-daemon-or-script.py" ]