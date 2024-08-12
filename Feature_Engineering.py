'''
Feature Engineering
This file performs various extractions/transformations to the dataset for a better model

Key Concepts Implemented:
- One Hot Encoded
'''


#%%

# Import libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import logging


#%%

def export_dataframe_to_csv(df, file_path):
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

#%%

def read_dataframe_to_csv(file_path):
    """
    Objective:
    Create a csv file of the objects tracked and its relevant features
    
    Parameters:
    [string] file_path - File path of where the raw data is (csv file)
    
    """
    try:
        raw_dataset = pd.read_csv(file_path)

        return raw_dataset
        

    except Exception as e:
        logger.error (f"Error occured when reading raw data file: {e}")
        raise

#%% Configuring logging

try:
    log_file_path = 'feature_engineering_output.log'

    # If the log file exists, delete it
    if os.path.exists(log_file_path):
        os.remove(log_file_path)
        print(f"The file {log_file_path} has been found thus deleted for the new run.")

    # Create a logger object
    logger = logging.getLogger('FeatureEngineeringLogger')
    logger.setLevel(logging.DEBUG)  # Set the minimum log level to debug

    # Create file handler which logs even debug messages
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(file_handler)

except Exception as e:
    logger.error("Error in creating logging configuration: %s", e)
    raise


def one_hot_encode_class_id(df):
    try:
        df['is_Team_A'] = (df['ClassID'] == 'Team_A').astype(int)
        df['is_Team_B'] = (df['ClassID'] == 'Team_B').astype(int)
        df['is_Basketball'] = (df['ClassID'] == 'Basketball').astype(int)
        return df

    except Exception as e:
        logger.error("Error in performing one hot encoding for ClassID: %s", e) 
        raise   

def extract_mean_values(df):
    try:
        mean_columns = ['pos_x', 'pos_y', 'aspect_ratio', 'height', 'vel_x', 'vel_y', 'vel_aspect', 'vel_height']
        df[mean_columns] = df['Mean'].str.strip('[]').str.split(expand=True).astype(float)
        return df

    except Exception as e:
        logger.error("Error in extracing object's position: %s", e)    
        raise

def normalize_confidence_score(df):
    try:
        scaler = MinMaxScaler()
        df['ConfidenceScore_normalized'] = scaler.fit_transform(df['ConfidenceScore'].values.reshape(-1, 1))
        return df

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)    
        raise

def transform_state(df):
    try:
        df['state_tentative'] = (df['State'] == 1).astype(int)
        df['state_confirmed'] = (df['State'] == 2).astype(int)
        return df

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)    
        raise

def process_temporal_features(df, fps=30):
    try:
        df['time_since_start'] = df['Frame'] / fps
        df['delta_time'] = df.groupby('TrackID')['time_since_start'].diff()
        df['is_key_frame'] = (df['Frame'] % 30 == 0).astype(int)
        return df

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)    
        raise

def extract_feature_statistics(df):
    try:
        feature_stats = df['Features'].apply(lambda x: pd.Series({
            'feature_mean': np.mean(x),
            'feature_std': np.std(x),
            'feature_min': np.min(x),
            'feature_max': np.max(x)
        }))
        return pd.concat([df, feature_stats], axis=1)

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)    
        raise


def process_covariance(df):
    try:
        def extract_covariance_stats(cov_str):
            cov_matrix = np.array(eval(cov_str))
            return pd.Series({
                'cov_trace': np.trace(cov_matrix),
                'cov_determinant': np.linalg.det(cov_matrix),
                'cov_pos_variance': cov_matrix[0, 0],
                'cov_vel_variance': cov_matrix[4, 4]
            })
        
        cov_stats = df['Co-Variance'].apply(extract_covariance_stats)
        return pd.concat([df, cov_stats], axis=1)

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)
        raise

#%%

def feature_transformation(dataset):
    try:
        transformed_dataset = one_hot_encode_class_id(dataset)
        transformed_dataset = extract_mean_values(transformed_dataset)
        transformed_dataset = normalize_confidence_score(transformed_dataset)
        transformed_dataset = transform_state(transformed_dataset)
        transformed_dataset = process_temporal_features(transformed_dataset)
        transformed_dataset = extract_feature_statistics(transformed_dataset)
        transformed_dataset = process_covariance(transformed_dataset)
        return transformed_dataset

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)
        raise

#%%

def convert_string_to_array(string_data):
    try:
        # Replace 'array([' with '['
        formatted_string = string_data.replace('array([', '[')

        # Remove the dtype part
        formatted_string = formatted_string.replace('], dtype=float32)', ']')

        # Convert the string to a numpy array
        features_array = np.array(eval(formatted_string))

        return features_array

    except Exception as e:
        logger.error("Error in converting features from string to array: %s", e)
        raise 


#%%

def prep_transformation(dataset):
    try:
        # Convert features column from string to array
        dataset['Features'] = dataset['Features'].apply(convert_string_to_array)

        return dataset

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)
        raise
#%%

def main():
    try:
        raw_dataset_file_path = 'assets/detected_objects.csv'
        raw_dataset = read_dataframe_to_csv(raw_dataset_file_path)
        prepped_dataset = prep_transformation(raw_dataset)
        extracted_feature_dataset = feature_transformation(prepped_dataset)
        extracted_feature_dataset_file_path = 'assets/extracted_features.csv'
        export_dataframe_to_csv(extracted_feature_dataset ,extracted_feature_dataset_file_path)

    except Exception as e:
        logger.error (f"Error occured when reading raw data file: {e}")
        raise

if __name__ == "__main__":
    main()


'[array([    0.24891,   -0.085511,    -0.12444,     -0.1798,    0.003194,   -0.005902,     0.01754,   0.0066565,    0.020735,  -0.0073673,   -0.081884,   -0.068911,   0.0025285,    0.069156,    0.029617,    -0.16428,     0.12657,    0.025046,    0.020961,   -0.047191,     0.14786,     0.20735,    0.086927,   -0.018191,\n           0.05566,    0.016369,    0.047447,     0.12192,  -0.0067973,    0.084716,   -0.074108,     0.11123,    0.078967,   -0.055416,    0.017184,     0.04102,    0.016254,   -0.080344,     0.13042,    -0.08577,   -0.024072], dtype=float32)]'