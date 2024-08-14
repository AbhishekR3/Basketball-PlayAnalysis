'''
Feature Engineering
This file performs various extractions/transformations to the dataset for a better model

Key Concepts Implemented:
- Categorical / One Hot Encoding 
- Normalize/Log Transformation
- Temporal Encoding
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

    Returns:
    [dataframe] raw_dataset - Converted csv file to dataframe
    """

    try:
        raw_dataset = pd.read_csv(file_path)

        return raw_dataset
        

    except Exception as e:
        logger.error (f"Error occured when reading raw data file: {e}")
        raise

#%%

def one_hot_encode_class_id(df):
    """
    Objective:
    One Hot Encode the Class_ID to be converted to Team_A, Team_B, Basketball
    
    Parameters:
    [dataframe] df - Dataframe containing object's tracked and relvant data

    Returns:
    [dataframe] df - Dataframe containing new Class_ID column
    """
        
    try:
        df['is_Team_A'] = (df['ClassID'] == 'Team_A').astype(int)
        df['is_Team_B'] = (df['ClassID'] == 'Team_B').astype(int)
        df['is_Basketball'] = (df['ClassID'] == 'Basketball').astype(int)
        return df

    except Exception as e:
        logger.error("Error in performing one hot encoding for ClassID: %s", e) 
        raise   

#%%

def extract_mean_values(df):
    """
    Objective:
    Extract values from mean column which reprsents DeepSORT extracted features of each object
    
    Parameters:
    [dataframe] df - Dataframe containing object's tracked and relvant data

    Returns:
    [dataframe] df - Dataframe containing each object's x,y, velocity and related values
    """

    try:
        mean_columns = ['pos_x', 'pos_y', 'aspect_ratio', 'height', 'vel_x', 'vel_y', 'vel_aspect', 'vel_height']
        df[mean_columns] = df['Mean'].str.strip('[]').str.split(expand=True).astype(float)
        return df

    except Exception as e:
        logger.error("Error in extracing object's position: %s", e)    
        raise

#%%

def normalize_confidence_score(df):
    """
    Objective:
    Normalize confidence score
    
    Parameters:
    [dataframe] df - Dataframe containing object's tracked and relvant data

    Returns:
    [dataframe] df - Dataframe containing normalized confidence score
    """

    try:
        scaler = MinMaxScaler()
        df['ConfidenceScore_normalized'] = scaler.fit_transform(df['ConfidenceScore'].values.reshape(-1, 1))
        return df

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)    
        raise

#%%

def transform_state(df):
    """
    Objective:
    One hot encoded for object's tentative/confirmed state
    
    Parameters:
    [dataframe] df - Dataframe containing object's tracked and relvant data

    Returns:
    [dataframe] df - Dataframe containing object's tentative/confirmed state
    """

    try:
        df['state_tentative'] = (df['State'] == 1).astype(int)
        df['state_confirmed'] = (df['State'] == 2).astype(int)
        return df

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)    
        raise

#%%

def process_temporal_features(df, fps=30):
    
    try:
        df['time_since_start'] = df['Frame'] / fps
        df['delta_time'] = df.groupby('TrackID')['time_since_start'].diff()
        df['key_frame_s'] = (df['Frame'] % 30 == 0).astype(int)
        return df

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)    
        raise

#%%

def extract_feature_statistics(df):
    try:
        feature_stats = df['Features'].apply(calculate_feature_stats)

        return pd.concat([df, feature_stats], axis=1)

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)    
        raise

#%%

def calculate_feature_stats(feature_series):
    # NEED TO UPDATE TO GET BEST CALCULATION instead of 0
    # Maybe rolling average or something
    try:
        stats = {}
        
        # Calculate mean
        try:
            stats['feature_mean'] = np.mean(feature_series)
        except Exception:
            stats['feature_mean'] = 0
        
        # Calculate standard deviation
        try:
            stats['feature_std'] = np.std(feature_series)
        except Exception:
            stats['feature_std'] = 0
        
        # Calculate minimum
        try:
            stats['feature_min'] = np.min(feature_series)
        except Exception:
            stats['feature_min'] = 0
        
        # Calculate maximum
        try:
            stats['feature_max'] = np.max(feature_series)
        except Exception:
            stats['feature_max'] = 0
        
        return pd.Series(stats)
    
    except Exception as e:
        logger.error(f"Error in calculate_feature_stats: {e}")
        return pd.Series({
            'feature_mean': 0,
            'feature_std': 0,
            'feature_min': 0,
            'feature_max': 0
        })

#%%

def convert_string_array(cov_str):
    try:
        cleaned_string = cov_str.strip('[]')
        rows = cleaned_string.split('\n')
        arrays = []

        for row in rows:
            cleaned_row = row.strip().strip('[]')
            values = [float(val) for val in cleaned_row.split()]
            # Create a numpy array from the values
            array = np.array(values)
            arrays.append(array)

        return arrays
    
    except Exception as e:
        logger.error(f"Error in converting covariance matrix to arrays: {e}")

#%%

def covariance_stats_calculation(pre_covariance):
    try:
        pre_covariance = np.array(pre_covariance)
        stats = [
            np.trace(pre_covariance), #cov_trace
            np.linalg.det(pre_covariance), #cov_determinant
            pre_covariance[0, 0] if pre_covariance.shape[0] > 0 and pre_covariance.shape[1] > 0 else np.nan, #cov_pos_variance_x
            pre_covariance[1, 1] if pre_covariance.shape[0] > 1 and pre_covariance.shape[1] > 1 else np.nan, #cov_pos_variance_y
            pre_covariance[2, 2] if pre_covariance.shape[0] > 2 and pre_covariance.shape[1] > 2 else np.nan, #cov_pos_variance_acceleration
            pre_covariance[3, 3] if pre_covariance.shape[0] > 3 and pre_covariance.shape[1] > 3 else np.nan, #cov_pos_variance_height
            pre_covariance[4, 4] if pre_covariance.shape[0] > 4 and pre_covariance.shape[1] > 4 else np.nan, #cov_vel_variance_x
            pre_covariance[5, 5] if pre_covariance.shape[0] > 5 and pre_covariance.shape[1] > 5 else np.nan, #cov_vel_variance_y
            pre_covariance[6, 6] if pre_covariance.shape[0] > 6 and pre_covariance.shape[1] > 6 else np.nan, #cov_vel_variance_accelerate
            pre_covariance[7, 7] if pre_covariance.shape[0] > 7 and pre_covariance.shape[1] > 7 else np.nan  #cov_vel_variance_height
        ]

        stats = np.array(stats)

        return stats

    except Exception as e:
        logger.error("Error in calculating co-variance statistics: %s", e)
        raise

#%%

def process_covariance(df):
    try:
        def extract_covariance_stats(cov_str):
            cov_matrix = convert_string_array(cov_str)

        cov_stats = df['Co-Variance'].apply(convert_string_array)
        df = df.drop(columns='Co-Variance')

        flattened_arrays = [arr.flatten() if isinstance(arr, np.ndarray) else np.array(arr).flatten() 
            for arr in cov_stats]
        
        # Stack the flattened arrays vertically to create a 2D matrix
        matrix = np.vstack(flattened_arrays)

        # Add Co-Variance column
        df = pd.concat([df, cov_stats], axis=1)

        # Calculate and add 9 new columns based on co-variance column calculation
        df_covar_columns = ['cov_trace', 'cov_determinant', 
                            'cov_pos_variance_x', 'cov_pos_variance_y', 'cov_pos_variance_acceleration', 'cov_pos_variance_height',
                            'cov_vel_variance_x', 'cov_vel_variance_y', 'cov_vel_variance_acceleration', 'cov_vel_variance_height']
        for col in df_covar_columns:
            df[col] = np.nan

        for i in range(df.shape[0]):
            cov_stats = covariance_stats_calculation(df.loc[i, 'Co-Variance'])
            df.loc[i, 'cov_trace'] = cov_stats[0]
            df.loc[i, 'cov_determinant'] = cov_stats[1]
            df.loc[i, 'cov_pos_variance_x'] = cov_stats[2]
            df.loc[i, 'cov_pos_variance_y'] = cov_stats[3]
            df.loc[i, 'cov_pos_variance_acceleration'] = cov_stats[4]
            df.loc[i, 'cov_pos_variance_height'] = cov_stats[5]
            df.loc[i, 'cov_vel_variance_x'] = cov_stats[6]
            df.loc[i, 'cov_vel_variance_y'] = cov_stats[7]
            df.loc[i, 'cov_vel_variance_acceleration'] = cov_stats[8]
            df.loc[i, 'cov_vel_variance_height'] = cov_stats[9]

        df = df.drop(columns='Co-Variance')

        return df

    except Exception as e:
        logger.error("Error in performing features transformation: %s", e)
        raise

#%%

def log_transformation(df, column_names):
    try:
        for ith_col in column_names:
            df[ith_col] = np.log(df[ith_col])
        return df

    except Exception as e:
        logger.error("Error in performing log transformation on dataset: %s", e)
        raise

#%%

def reliability(group, window_size = 30):
    '''
    Objective:
    Calculate the recent consistency of a track for it's reliability.
    Take the last 30 frames for a specific detected track.
    Calculate hits / age to give recent consistency.

    Parameters:


    Returns:

    
    '''
    try:
        group = group.sort_values('Frame')

        rolling_avg = group['DetectionConsistency'].rolling(
                window=min(window_size, len(group)),
                min_periods=1
            ).mean()
        
        return rolling_avg

    except Exception as e:
        logger.error("Error when calculating recent reliability metric: %s", e)
        raise 

def rolling_detection_consistency(df, window_size=30):
    """
    Objective:
    Calculate the rolling average of 'DetectionConsistency' for each 'TrackID' in the DataFrame

    Parameters:
    df (pd.DataFrame): Input DataFrame containing 'TrackID', 'Frame', and 'DetectionConsistency' columns
    window_size (int): The maximum number of frames to consider for the rolling average

    Returns:
    pd.Series: A series with the rolling average of 'DetectionConsistency' for all TrackIDs
    """
    try:
        # Apply the reliability function to each group
        result = df.groupby('TrackID', group_keys=False).apply(
            lambda x: reliability(x, window_size), include_groups=False
        ).reset_index(level=0, drop=True)

        return result

    except Exception as e:
        logger.error("Error in calculating rolling detection consistency: %s", e)
        raise

def recent_reliability_correction(row):
    try:
        if row['Age'] < 30:
            return row['Age'] / 30
        
        else:
            return row['RecentReliability']
        
    except Exception as e:
        logger.error("Error in calculating correcting rolling detection: %s", e)
        raise
        
#%%

def hits_age(df):
    try:
        # Fill Hits/Age with 0 in case there are null values
        df['Hits'] = df['Hits'].fillna(0)
        df['Age'] = df['Age'].fillna(0)

        # Calculate Occlusion Frequency, Detection Consistency, Recent Reliability
        df['OcclusionFrequency'] = (df['Age']-df['Hits'])/df['Age']
        df['DetectionConsistency'] = df['Hits']/df['Age']
        df['RecentReliability'] = rolling_detection_consistency(df, window_size = 30)
        df['RecentReliability'] = df.apply(recent_reliability_correction, axis=1) # Update recent reliability so tracked frames < 30 are updated accordingly

        return df
    
    except Exception as e:
        logger.error("Error in performing log transformation on dataset: %s", e)
        raise

def extract_acceleration(df):
    try:
        df = df.sort_values(['TrackID', 'Frame'], ascending=[True, False])

        # Calculate the previous velocity value for acceleration calculation 
        df['prev_vel_x'] = df.groupby('TrackID')['vel_x'].shift(-1)
        df['prev_vel_y'] = df.groupby('TrackID')['vel_y'].shift(-1)
        df['prev_vel_aspect'] = df.groupby('TrackID')['vel_aspect'].shift(-1)
        df['prev_vel_height'] = df.groupby('TrackID')['vel_height'].shift(-1)

        df = df.fillna(0)

        # Calculate acceleration
        df['accel_x'] = ((df['vel_x']-df['prev_vel_x'])/df['vel_x'])/df['delta_time']
        df['accel_y'] = ((df['vel_x']-df['prev_vel_y'])/df['vel_y'])/df['delta_time']
        df['accel_aspect'] = ((df['vel_aspect']-df['prev_vel_aspect'])/df['vel_aspect'])/df['delta_time']
        df['accel_height'] = ((df['vel_height']-df['prev_vel_height'])/df['vel_height'])/df['delta_time']

        df = df.sort_values(['TrackID', 'Frame'], ascending=[True, True])

        return df
    
    except Exception as e:
        logger.error("Error in performing log transformation on dataset: %s", e)
        raise

#%%

def feature_extraction(dataset):
    try:
        transformed_dataset = one_hot_encode_class_id(dataset)
        transformed_dataset = extract_mean_values(transformed_dataset)
        transformed_dataset = normalize_confidence_score(transformed_dataset)
        transformed_dataset = transform_state(transformed_dataset)
        transformed_dataset = process_temporal_features(transformed_dataset)
        transformed_dataset = extract_feature_statistics(transformed_dataset)
        transformed_dataset = process_covariance(transformed_dataset)
        transformed_dataset = hits_age(transformed_dataset)
        transformed_dataset = log_transformation(transformed_dataset, ['Age', 'Hits'])
        transformed_dataset = extract_acceleration(transformed_dataset)
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

def optimize_dataset(dataset):
    try:
        # Remove unecessary columns
        columns_dropped = ['Mean', 'Unnamed: 0', 'ConfidenceScore', 'State', 'Features', 'ClassID']
        dataset = dataset.drop(columns=columns_dropped)

        # Convert NaN values to 0 for delta time
        dataset = dataset.fillna(0)

        dataset = dataset.reset_index(drop = True)

        return dataset

    except Exception as e:
        logger.error("Error in optimizing/cleaning dataset: %s", e)
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

#%%

def main():
    try:
        # Import detected objects dataset
        raw_dataset_file_path = 'assets/detected_objects.csv'
        raw_dataset = read_dataframe_to_csv(raw_dataset_file_path)

        # Prep dataset for feature extraction
        prepped_dataset = prep_transformation(raw_dataset)

        # Extracting features from the dataset
        extracted_feature_dataset = feature_extraction(prepped_dataset)

        # Optimize / Clean up the dataset with all the extracted features
        cleaned_feature_dataset = optimize_dataset(extracted_feature_dataset)

        # Export the finalized dataset into a csv
        processed_feature_dataset_file_path = 'assets/processed_features.csv'
        export_dataframe_to_csv(cleaned_feature_dataset ,processed_feature_dataset_file_path)

    except Exception as e:
        logger.error (f"Error occured in main function: {e}")
        raise

if __name__ == "__main__":
    main()