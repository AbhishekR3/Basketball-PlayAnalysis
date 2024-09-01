'''
utils.py

This file contains utility functions that are used in other scripts
'''


#%%

# Importing necessary libraries
import pandas as pd
import os
import logging


#%% Export Dataframe to CSV file

def export_dataframe_to_csv(df, file_path, logger): 
    """
    Objective:
    Create a csv file of the objects tracked and its relevant features
    
    Parameters:
    [dataframe] df - Dataframe containing object's tracked and reelvant data
    [string] file_path - File path of where the csv file should be saved at
    """

    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Export the DataFrame to CSV
        df.to_csv(file_path)
        print(f"DataFrame successfully exported to {file_path}")
        logger.debug (f"DataFrame successfully exported to {file_path}")
    except Exception as e:
        logger.error (f"Error occured when exporting data file after extracting and transforming: {e}")
        raise

#%% Read Dataframe to CSV

def read_dataframe_to_csv(file_path, logger):
    """
    Objective:
    Create a csv file of the objects tracked and its relevant features
    
    Parameters:
    [string] file_path - File path of where the raw data is (csv file)

    Returns:
    [dataframe] raw_dataset - Converted csv file to dataframe
    """

    try:
        raw_dataset = pd.read_csv(file_path)

        return raw_dataset
        

    except Exception as e:
        logger.error (f"Error occured when reading raw data file: {e}")
        raise

#%% Configuring logging

def configure_logger(filename):
    """
    Objective:
    Configure the logger for the given script

    Parameters:
    [string] filename - The name of the file creating the logger file for
    """

    try:
        # Set up the log file path
        log_filename = filename + '.log'
        try:
            log_file_path = os.path.join(log_dir, log_filename)
        except Exception as e:
            print(f"Error in setting up log file path: {e}")
            log_file_path = log_filename

        # If the log file exists, delete it
        if os.path.exists(log_file_path):
            os.remove(log_file_path)
            print(f"The file {log_file_path} has been found thus deleted for the new run.")

        # Create a logger object
        logger_name = filename + 'Logger'
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)  # Set the minimum log level to debug

        # Create file handler which logs even debug messages
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.DEBUG)

        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add the handlers to the logger
        logger.addHandler(file_handler)
    
        return logger

    except Exception as e:
        print(f"Error in creating logging: {e}")
        raise