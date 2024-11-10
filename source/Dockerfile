# Use an official Ubuntu Linux runtime as a parent image
FROM ubuntu:latest

# Update package list and install bash utilities and software-properties-common
# Update package list
RUN apt-get update

# Install software-properties-common
RUN apt-get install -y software-properties-common

# Install apt-utils
RUN apt-get install -y apt-utils

# Install bash
RUN apt-get install -y bash

# Install unzip
RUN apt-get install -y unzip

# Install tshark
RUN apt-get install -y tshark

# Install xxd
RUN apt-get install -y xxd

# Install parallel
RUN apt-get install -y parallel

# Add the deadsnakes PPA to install Python 3.9
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-distutils \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /usr/src/app

# Create a virtual environment
RUN python3.9 -m venv /usr/src/app/venv

# Activate the virtual environment and install required Python packages
RUN /bin/bash -c "source /usr/src/app/venv/bin/activate && \
    pip install --upgrade pip && \
    pip install numpy pandas torch scikit-learn scipy scapy transformers \
    tensorflow datasets pyshark joblib regex dpkt python-dateutil pytz \
    six threadpoolctl gensim spicy"

# Keep the container running indefinitely
CMD ["tail", "-f", "/dev/null"]