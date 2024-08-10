# Use an official Python runtime as a parent image, specifically for ARM64
FROM arm64v8/python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install system dependencies and build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    libhdf5-serial-dev \
    gcc \
    pkg-config \
    python3-dev \
    libqt5gui5 \
    libqt5webkit5-dev \
    libqt5test5 \
    libxvidcore4 \
    x264 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Set HDF5 directory for h5py
ENV HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial

# Install h5py separately with specific compile flags
RUN CFLAGS="-I/usr/include/hdf5/serial -L/usr/lib/aarch64-linux-gnu/hdf5/serial" pip install --no-cache-dir h5py --no-binary=h5py

# Install build dependencies for packages using pyproject.toml
RUN pip install --no-cache-dir build

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run simulation.py when the container launches
CMD ["python", "Basketball_Passing_Simulation.py"]