FROM mosaicml/pytorch:2.6.0_cu124-python3.12-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install some basic utilities
RUN apt-get update --fix-missing && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    checkinstall \
    locales \
    lsb-release \
    mesa-utils \
    subversion \
    terminator \
    xterm \
    wget \
    htop \
    libssl-dev \
    build-essential \
    dbus-x11 \
    software-properties-common \
    gdb valgrind \
    libeigen3-dev \
    libboost-all-dev \
    fish \
    libglfw3-dev \
    libgl-dev \
    unzip \
    cmake \
    libglu-dev \
    fish \
&& rm -rf /var/lib/apt/lists/*

RUN echo "alias ..='cd ..'" >> ~/.bashrc
RUN echo "alias ...='cd .. && cd ..'" >> ~/.bashrc
RUN echo "alias py=/usr/bin/python3" >> ~/.bashrc

RUN pip3 install pandas tensorboardx
