# Use a ARM64 slim Python image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Copy the relevant files to process files
### Relevant Setup Files
COPY requirements.txt .
COPY run_sequence.sh .
COPY utils.py .
### Simulation
#COPY Passing_Simulation.py .
COPY RandomMovement_Simulation.py .
COPY "assets/Basketball_Court_Diagram.jpg" ./assets/
### Tracking
COPY Object_Tracking.py .
COPY "assets/YOLOv10s_custom.pt" ./assets/
COPY deep_sort/ ./deep_sort/
### Feature Engineering
COPY Feature_Engineering.py .


# Install system dependencies and build tools
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential=12.9 \
    libhdf5-dev=1.10.8+repack1-1 \
    gcc=4:12.2.0-3 \
    pkg-config=1.8.1-1 \
    python3-dev=3.11.2-1+b1 \
    libqt5gui5=5.15.8+dfsg-11+deb12u2 \
    libqt5webkit5-dev=5.212.0~alpha4-30 \
    libqt5test5=5.15.8+dfsg-11+deb12u2 \
    libxvidcore4=2:1.3.7-1 \
    x264=2:0.164.3095+gitbaee400-3 \
    ffmpeg=7:5.1.6-0+deb12u1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install build dependencies
RUN pip install --no-cache-dir pip==21.0.1 setuptools==57.0.0 wheel==0.36.2

# Set HDF5 directory for h5py
ENV HDF5_DIR=/usr/lib/aarch64-linux-gnu/hdf5/serial

# Install h5py separately with specific compile flags
RUN CFLAGS="-I/usr/include/hdf5/serial -L/usr/lib/aarch64-linux-gnu/hdf5/serial" pip install h5py==3.11.0 --no-binary=h5py

# Install required packages
COPY requirements.txt .
RUN pip install -v -r requirements.txt

# Create Environment Variables
### Input
ENV ASSETS_DIR=/app/assets

### Output
ENV OUTPUT_DIR=/app/output
ENV LOG_DIR=/app/output/logs
ENV VIDEO_DIR=/app/output/simulations
ENV TRACKING_DIR=/app/output/tracking_data

# Create folder for directories
RUN mkdir -p ${LOG_DIR} ${VIDEO_DIR} ${TRACKING_DIR}

# Create volume for output directory
VOLUME $OUTPUT_DIR

# Enable run sequence script is executable
RUN chmod +x run_sequence.sh

# Run the script
CMD ["./run_sequence.sh"]