FROM python:3

RUN apt update -y
RUN apt install -y \
	openmpi-bin \
	libopenmpi-dev

# ARG UID=25160
# ARG USER=user
# ARG PASS=qazxsw
# RUN useradd -m --uid ${UID} --groups sudo ${USER} \
# 	&& echo ${USER}:${PASS} | chpasswd
# USER ${UID}

# WORKDIR /${USER}/src
WORKDIR /root/src

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# add path
# ENV PATH $PATH:~/.local/lib

# CMD [ "python", "./your-daemon-or-script.py" ]