#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

LOG_FILE="/app/output/logs/run_sequence.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Ensure output directories exist
mkdir -p /app/output/logs /app/output/tracking_data /app/output/simulations

# Run the scripts in sequence

# Passing Simulation
log_message "Starting Basketball_Passing_Simulation.py"
python3 Basketball_Passing_Simulation.py
if [ $? -ne 0 ]; then
    log_message "Error: Basketball_Passing_Simulation.py failed"
    exit 1
fi
log_message "Completed Basketball_Passing_Simulation.py"

# Object Tracking
log_message "Starting Basketball_Object_Tracking.py"
python3 Basketball_Object_Tracking.py
if [ $? -ne 0 ]; then
    log_message "Error: Basketball_Object_Tracking.py failed"
    exit 1
fi
log_message "Completed Basketball_Object_Tracking.py"

# Check if the required file exists before proceeding
if [ ! -f "/app/output/tracking_data/detected_objects.csv" ]; then
    log_message "Error: detected_objects.csv not found. Cannot proceed with Feature Engineering."
    exit 1
fi

# Feature Engineering
log_message "Starting Feature_Engineering.py"
python3 Feature_Engineering.py
if [ $? -ne 0 ]; then
    log_message "Error: Feature_Engineering.py failed"
    exit 1
fi
log_message "Completed Feature_Engineering.py"


log_message "All scripts completed successfully"