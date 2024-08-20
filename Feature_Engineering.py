'''
Feature Engineering
This file performs various extractions/transformations to the dataset for a better model

Key Concepts Implemented:
- Mean, SD, Max, Min, Skewness, Kurtosis 
- Categorical / One Hot Encoding 
- Normalization/Log Transformation
- Rolling averages
- Temporal Encoding
- PCA (Principal Component Analysis)
'''


#%%

# Import libraries
import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.stats import kurtosis
import os
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

def rolling_average_calculation(group, window_size = 30, rolling_column = None):
    '''
    Objective:
    Calculate the rolling average for a specific column
    Take the last 30 frames for a specific detected track

    Parameters:


    Returns:

    
    '''
    try:
        group = group.sort_values('Frame')

        rolling_avg = group[rolling_column].rolling(
                window=min(window_size, len(group)),
                min_periods=1
            ).mean()
        
        return rolling_avg

    except Exception as e:
        logger.error("Error when calculating rolling average: %s", e)
        raise  

def rolling_avgs(df):
    try:
        """
        
        """
        rolling_avgs_columns = ["pos_x","pos_y","aspect_ratio","height","vel_x","vel_y","vel_aspect","vel_height",
                                "feature_mean","feature_std","feature_max","cov_determinant",
                                "prev_vel_x","prev_vel_aspect","accel_x","accel_y","accel_aspect","accel_height"]
        
        for col in rolling_avgs_columns:
            column_name = col + '_rolling_avg'
            df[column_name] = pd.Series(dtype='float64')

            # Create a mask for the varying rolling average winow
            condition = df['is_Basketball'] == 1

            # Apply different window sizes based on the condition
            df.loc[condition, column_name] = rolling_average(df, window_size = 3, rolling_column = col) #Basketball
            df.loc[~condition, column_name] = rolling_average(df, window_size = 5, rolling_column = col) #Players
        
        # Remove the columns that have been converted to rolling average
        df = df.drop(columns=rolling_avgs_columns)

        return df

    except Exception as e:
        logger.error("Error when calculating rolling averages through rolling_avgs(): %s", e)
        raise     


def rolling_average(df, window_size=30, rolling_column = None):
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
            lambda x: rolling_average_calculation(x, window_size, rolling_column)
            #include_groups=False
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
        df['RecentReliability'] = rolling_average(df, window_size = 30, rolling_column = 'DetectionConsistency')
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
        transformed_dataset = transform_state(transformed_dataset)
        transformed_dataset = process_temporal_features(transformed_dataset)
        transformed_dataset = extract_feature_statistics(transformed_dataset)
        transformed_dataset = process_covariance(transformed_dataset)
        transformed_dataset = hits_age(transformed_dataset)
        transformed_dataset = log_transformation(transformed_dataset, ['Age', 'Hits'])
        transformed_dataset = extract_acceleration(transformed_dataset)
        transformed_dataset = trackid_temporal_encoding(transformed_dataset)
        transformed_dataset = rolling_avgs(transformed_dataset)

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
        # Convert NaN values to 0
        dataset = dataset.fillna(0)

        dataset = dataset.reset_index(drop = True)

        # Normalize data
        dataset = normalize_numerical_columns(dataset)

        # Perform PCA
        feature_importance, feature_covariation, pca_model = perform_pca(dataset, n_components=16, variance_threshold=0.85)

        # Remove unecessary columns (Including PCA analysis)
        columns_dropped = ['Mean', 'Unnamed: 0', 'ConfidenceScore', 'State', 'Features', 'ClassID',
                            'RecentReliability', 'Hits', 'delta_time', 'feature_min',
                            'cov_trace', 'cov_pos_variance_y', 'cov_pos_variance_x', 'cov_pos_variance_height',
                            'cov_vel_variance_height', 'cov_vel_variance_y', 'cov_vel_variance_x', 'cov_pos_variance_acceleration', 'cov_vel_variance_acceleration'
                            ]
        dataset = dataset.drop(columns=columns_dropped)

        return dataset

    except Exception as e:
        logger.error("Error in optimizing/cleaning dataset: %s", e)
        raise

#%%

def perform_pca(df, n_components=20, variance_threshold=0.9):
    """
    Objective:
    Perform Principal Component Analysis (PCA) on the input dataset.

    Parameters:
    [pd.DataFrame] df - Input dataframe containing numeric columns for PCA
    [int] n_components - Number of components to keep
    [float] variance_threshold - Minimum cumulative varianace ratio

    Returns:
    [pd.DataFrame] feature_importance - Dataframe showing the importance of original features in each principal component
    [PCA] pca_model - The fitted PCA model
    """
    try:
        # Select numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        X = df[numeric_columns]

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Initialize PCA
        pca = PCA(n_components)

        # Fit PCA on the scaled data
        X_pca = pca.fit_transform(X_scaled)

        # Get feature importance
        loadings = pd.DataFrame(
            data=pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=numeric_columns
        )

        # Calculate importance score of each feature 
        importance = loadings.abs().sum(axis=1).sort_values(ascending=False)
        feature_importance = pd.DataFrame({
            'Feature': importance.index,
            'Importance': importance.values
        })
        print('Feature Importance')
        print(feature_importance)
        logger.debug(feature_importance)
        print("")

        # Plot explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                 np.cumsum(pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance vs. Number of Components')
        plt.axhline(y=variance_threshold, color='r', linestyle='--')
        plt.axvline(x=n_components, color='r', linestyle='--')
        plt.show()

        return feature_importance, loadings, pca

    except Exception as e:
        logger.error("Error in performing PCA: %s", e)
        raise
    
#%%

def trackid_temporal_encoding(df):
    try:
        # Calculate max number of frames
        max_frame = df['Frame'].max()

        # Calculate normalized time based on current_frame/max_frame
        df['Normalized_Time_Frame'] = df['Frame']/max_frame

        # Calculate TrackID with Temporal Factor
        df['Temporal_TrackID'] = df['Normalized_Time_Frame']+df['TrackID']

        return df

    except Exception as e:
        logger.error("Error when peforming temporal encoding for an exponential decay on TrackID: %s", e)
        raise

#%%

def normalize_numerical_columns(df):
    """
    Objective:
    Normalize all numerical columns in the given DataFrame to a range between 0 and 1,
    replacing the original columns with their normalized versions.

    Parameters:
    [pd.DataFrame] df - Input DataFrame containing the columns to be normalized

    Returns:
    [pd.DataFrame] df - Original DataFrame with numerical columns replaced by their normalized versions
    """

    try:
        # Initialize the MinMaxScaler
        scaler = MinMaxScaler()
        
        # Identify numerical columns
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        
        # Apply normalization to numerical columns and replace original columns
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
        
        return df
    
    except Exception as e:
        print(f"An error occurred during normalization: {e}")
        raise

#%%
#'''
try:
    log_dir = os.environ.get('LOG_DIR', '/app/logs')
    tracking_dir = os.environ.get('TRACKING_DIR', '/app/tracking_data')

    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    # Use it before writing files
    ensure_dir(log_dir)
    ensure_dir(tracking_dir)

    print('Log Directory:', log_dir)
    print('Tracking Data Directory:', tracking_dir)

except Exception as e:
    print(f"Error in creating environment for containers: {e}")
    raise
#'''
#%% Configuring logging
try:
    try:
        log_file_path = os.path.join(log_dir, 'feature_engineering_output.log')
    except:
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
    print(f"Error in creating logging: {e}")
    raise
#%% Main Function for Feature Engineering of the assets 

def main():
    try:
        # Import detected objects dataset
        try:
            raw_dataset_file_path = os.path.join(tracking_dir, 'detected_objects.csv')
        except:
            raw_dataset_file_path = 'assets/detected_objects.csv'

        raw_dataset = read_dataframe_to_csv(raw_dataset_file_path)

        # Prep dataset for feature extraction
        prepped_dataset = prep_transformation(raw_dataset)

        # Extracting features from the dataset
        extracted_feature_dataset = feature_extraction(prepped_dataset)

        # Optimize / Clean up the dataset with all the extracted features
        cleaned_feature_dataset = optimize_dataset(extracted_feature_dataset)
 
        # Export the finalized dataset into a csv
        try:
            processed_feature_dataset_file_path = os.path.join(tracking_dir, 'processed_features.csv')
        except:
            processed_feature_dataset_file_path = 'assets/processed_features.csv'

        export_dataframe_to_csv(cleaned_feature_dataset ,processed_feature_dataset_file_path)

    except Exception as e:
        logger.error (f"Error occured in main function: {e}")
        raise

if __name__ == "__main__":
    main()