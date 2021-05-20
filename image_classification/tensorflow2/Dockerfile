FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04


WORKDIR /research

RUN apt-get update

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    git \
    python3 \
    python3-pip \
    python \
    python-pip

ENV HOME /research
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN apt-get install -y python-setuptools

RUN apt-get install -y virtualenv htop
RUN pip3 install --upgrade --ignore-installed setuptools
RUN pip3 install --upgrade numpy scipy sklearn tensorflow-gpu==2.2.0

# Mount data into the docker
ADD . /research/resnet


WORKDIR /research/resnet
RUN pip3 install -r official/requirements.txt

ENTRYPOINT ["/bin/bash"]

