FROM python:3.9-slim-buster

WORKDIR /root
ENV VENV /opt/venv 
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
ENV PYTHONPATH /root

RUN apt-get ipdate && \
    apt-get install -y \
    ffmpeg \
    build-essential

ENV VENV /opt/venv

# virtual environment
RUN python3 -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

COPY  requirements.txt /root

RUN pip install -r /root/requirements.txt

COPY . /root