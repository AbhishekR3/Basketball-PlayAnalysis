#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Use the environment variable for the log directory
LOG_DIR="${LOG_DIR:-/app/output/logs}"
LOG_FILE="$LOG_DIR/run_sequence.log"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Function to check for success in output file
check_success() {
    local script_name="$1"
    local output_file="$LOG_DIR/${script_name}_output.log"
    
    # Check if the output file exists
    if [ ! -f "$output_file" ]; then
        log_message "Error: Output file $output_file not found for $script_name"
        exit 1
    fi
    
    # Check if the word "succeeded" is in the output file
    if ! grep -q "succeeded" "$output_file"; then
        log_message "Error: 'succeeded' not found in output of $script_name"
        exit 1
    fi
    
    # If we've made it here, both checks passed
    log_message "Successfully completed $script_name"
}

# Ensure output directories exist
mkdir -p "$LOG_DIR" /app/output/tracking_data /app/output/simulations

# Run the scripts in sequence

# Passing Simulation
log_message "Starting Passing_Simulation.py"
# The 2>&1 redirects both stdout (1) and stderr (2) to the log file
python3 Passing_Simulation.py > "$LOG_DIR/Passing_Simulation_output.log" 2>&1
check_success "Passing_Simulation"

# Object Tracking
log_message "Starting Object_Tracking.py"
python3 Object_Tracking.py > "$LOG_DIR/Object_Tracking_output.log" 2>&1
check_success "Object_Tracking"

# Feature Engineering
log_message "Starting Feature_Engineering.py"
python3 Feature_Engineering.py > "$LOG_DIR/Feature_Engineering_output.log" 2>&1
check_success "Feature_Engineering"

log_message "All scripts completed successfully"