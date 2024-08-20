# Use a ARM64 slim Python image
FROM python:3.11-slim


# Set working directory in the container
WORKDIR /app


### Copy the relevant files to process files
# Dockerfile
COPY requirements.txt .
COPY run_sequence.sh .
# Simulation
COPY Passing_Simulation.py .
COPY "assets/Basketball_Court_Diagram.jpg" ./assets/
# Tracking
COPY Object_Tracking.py .
COPY "assets/YOLOv10m_custom.pt" ./assets/
COPY deep_sort/ ./deep_sort/
# Feature Engineering
COPY Feature_Engineering.py .


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
RUN CFLAGS="-I/usr/include/hdf5/serial -L/usr/lib/aarch64-linux-gnu/hdf5/serial" pip install h5py --no-binary=h5py

# Install required packages
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