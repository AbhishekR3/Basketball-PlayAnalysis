#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Use the environment variable for the log directory
LOG_DIR="${LOG_DIR:-/app/output/logs}"
RESOURCE_DIR="${LOG_DIR}/resources"
LOG_FILE="$LOG_DIR/run_sequence.log"

# Create resource monitoring directory
mkdir -p "$RESOURCE_DIR"

# Function to log messages
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$LOG_FILE"
}

# Ensure output directories exist
mkdir -p "$LOG_DIR" /app/output/tracking_data /app/output/simulations

# Run the scripts in sequence with resource monitoring
log_message "Starting Passing_Simulation.py with resource monitoring"
python3 -c "
from resource_monitor import monitor_resources
import Passing_Simulation

@monitor_resources('${RESOURCE_DIR}')
def run_simulation():
    Passing_Simulation.main()

run_simulation()
"

log_message "Starting Object_Tracking.py with resource monitoring"
python3 -c "
from resource_monitor import monitor_resources
import Object_Tracking

@monitor_resources('${RESOURCE_DIR}')
def run_tracking():
    Object_Tracking.main()

run_tracking()
"

log_message "Starting Feature_Engineering.py with resource monitoring"
python3 -c "
from resource_monitor import monitor_resources
import Feature_Engineering

@monitor_resources('${RESOURCE_DIR}')
def run_engineering():
    Feature_Engineering.main()

run_engineering()
"

# Generate resource usage report
python3 -c "
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def generate_report():
    resource_dir = '${RESOURCE_DIR}'
    dfs = []
    
    for file in os.listdir(resource_dir):
        if file.endswith('_resources.csv'):
            df = pd.read_csv(os.path.join(resource_dir, file))
            script_name = file.replace('_resources.csv', '')
            df['script'] = script_name
            dfs.append(df)
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Create visualizations
    plt.figure(figsize=(15, 15))
    
    # CPU Usage
    plt.subplot(3, 2, 1)
    sns.lineplot(data=combined_df, x='timestamp', y='cpu_percent', hue='script')
    plt.title('CPU Usage')
    
    # Memory Usage
    plt.subplot(3, 2, 2)
    sns.lineplot(data=combined_df, x='timestamp', y='memory_percent', hue='script')
    plt.title('Memory Usage')
    
    # GPU Memory
    plt.subplot(3, 2, 3)
    sns.lineplot(data=combined_df, x='timestamp', y='gpu_memory_allocated', hue='script')
    plt.title('GPU Memory Usage')
    
    # Disk I/O
    plt.subplot(3, 2, 4)
    sns.lineplot(data=combined_df, x='timestamp', y='disk_io_write', hue='script')
    plt.title('Disk Write Activity')
    
    # Energy Usage - CPU
    plt.subplot(3, 2, 5)
    sns.lineplot(data=combined_df, x='timestamp', y='cpu_energy', hue='script')
    plt.title('CPU Energy Consumption (W)')
    
    # Energy Usage - Total
    plt.subplot(3, 2, 6)
    sns.lineplot(data=combined_df, x='timestamp', y='total_energy', hue='script')
    plt.title('Total Energy Consumption (W)')
    
    plt.tight_layout()
    plt.savefig('${RESOURCE_DIR}/resource_usage_report.png')
    
    # Generate summary statistics
    summary = combined_df.groupby('script').agg({
        'cpu_percent': ['mean', 'max'],
        'memory_percent': ['mean', 'max'],
        'gpu_memory_allocated': ['mean', 'max'],
        'disk_io_write': ['sum']
    }).round(2)
    
    summary.to_csv('${RESOURCE_DIR}/resource_summary.csv')

generate_report()
"

log_message "All scripts completed successfully"
log_message "Resource monitoring results saved in ${RESOURCE_DIR}"