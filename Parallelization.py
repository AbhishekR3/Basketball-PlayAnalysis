'''
Parallelization.py

This file parallelizes the basketball analysis pipeline, allowing multiple instances to run concurrently.
'''

# Import Libraries
import subprocess
import os
import uuid
import argparse
from utils import configure_logger
import multiprocessing
from multiprocessing import Pool
import cv2
import pandas as pd
import numpy as np

#%%
def passing_simulation(output_path):
    """
    Objective:
    Simulate basketball passing and generate a video file.

    Parameters:
    [str] output_path - Path to save the output video file

    Returns:
    None
    """
    try:
        logger.info(f"Passing simulation started. Output will be saved to {output_path}")

        # Your Passing_Simulation.py code here
        # Make sure to save the output to output_path

        logger.info(f"Passing simulation completed. Output saved to {output_path}")
    except Exception as e:
        logger.error(f"Error in passing simulation: {str(e)}")
        raise

#%%
def object_tracking(input_path, output_path):
    """
    Objective:
    Perform object tracking on the input video and generate a CSV file.

    Parameters:
    [str] input_path - Path to the input video file
    [str] output_path - Path to save the output CSV file

    Returns:
    None
    """
    try:
        logger.info(f"Object tracking started. Input from {input_path}, output will be saved to {output_path}")

        # Your Object_Tracking.py code here
        # Make sure to read from input_path and save the output to output_path

        logger.info(f"Object tracking completed. Output saved to {output_path}")
    except Exception as e:
        logger.error(f"Error in object tracking: {str(e)}")
        raise

#%%
def feature_engineering(input_path, output_path):
    """
    Objective:
    Perform feature engineering on the input CSV and generate a new CSV file.

    Parameters:
    [str] input_path - Path to the input CSV file
    [str] output_path - Path to save the output CSV file

    Returns:
    None
    """
    try:
        logger.info(f"Feature engineering started. Input from {input_path}, output will be saved to {output_path}")

        # Your Feature_Engineering.py code here
        # Make sure to read from input_path and save the output to output_path

        logger.info(f"Feature engineering completed. Output saved to {output_path}")
    except Exception as e:
        logger.error(f"Error in feature engineering: {str(e)}")
        raise

#%%
def run_pipeline(instance_id):
    """
    Objective:
    Run a single instance of the complete pipeline.

    Parameters:
    [str] instance_id - Unique identifier for this pipeline instance

    Returns:
    [str] final_output_path - Path to the final output file, or None if an error occurred
    """
    try:
        logger.info(f"Starting pipeline {instance_id}")
        
        # Generate unique filenames for this pipeline instance
        sim_output = f"simulation_{instance_id}.mp4"
        tracking_output = f"object_tracked_{instance_id}.csv"
        features_output = f"feature_engineered_{instance_id}.csv"

        # Run Passing Simulation
        passing_simulation(sim_output)

        # Run Object Tracking
        object_tracking(sim_output, tracking_output)

        # Run Feature Engineering
        feature_engineering(tracking_output, features_output)

        logger.info(f"Pipeline {instance_id} completed successfully.")

        return features_output

    except Exception as e:
        logger.error(f"Error in pipeline {instance_id}: {str(e)}")
        return None

#%%
def run_parallel_pipelines(num_instances):
    """
    Objective:
    Run multiple pipeline instances in parallel.

    Parameters:
    [int] num_instances - Number of pipeline instances to run

    Returns:
    [list] successful_results - List of paths to successfully completed pipeline outputs
    """
    logger.info(f"Running {num_instances} pipeline instances in parallel.")
    
    # Determine the number of CPU cores available
    num_cores = multiprocessing.cpu_count()
    
    # Use the minimum of num_instances and num_cores
    num_processes = min(num_instances, num_cores)

    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Generate unique IDs for each pipeline instance
        instance_ids = [str(uuid.uuid4()) for _ in range(num_instances)]

        # Run the pipelines in parallel
        results = pool.map(run_pipeline, instance_ids)

    # Filter out None results (failed pipelines)
    successful_results = [r for r in results if r is not None]

    return successful_results

#%%
def main():
    # Set up logging
    logger = configure_logger('parallelization')
    logger.info("Initialized logging for parallelization module.")

    try:
        parser = argparse.ArgumentParser(description="Run parallel basketball analysis pipelines")
        parser.add_argument("num_instances", type=int, help="Number of pipeline instances to run")
        args = parser.parse_args()

        results = run_parallel_pipelines(args.num_instances)
        logger.info(f"Completed {len(results)} out of {args.num_instances} pipelines successfully.")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}

if __name__ == "__main__":
    main()