'''
Data Loading
This file creates the loads the dataset (after feature engineering) into AWS spatial database

Key Concepts Implemented:
- Constraints 
- Spatial Index
'''

#%%

# Import Libraries
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, Integer, Float, Boolean
from geoalchemy2 import Geometry

#%% 

Base = declarative_base()

class TrackingData(Base):
    __tablename__ = 'tracking_data'

    id = Column(Integer, primary_key=True)
    frame = Column(Integer, nullable=False)
    age = Column(Integer, nullable=False)
    is_Team_A = Column(Boolean, nullable=False)
    is_Team_B = Column(Boolean, nullable=False)
    is_Basketball = Column(Boolean, nullable=False)
    state_tentative = Column(Boolean, nullable=False)
    state_confirmed = Column(Boolean, nullable=False)
    key_frame_s = Column(Boolean, nullable=False)
    occlusionFrequency = Column(Integer, nullable=False)
    detectionConsistency = Column(Integer, nullable=False)
    time_since_start = Column(Float, nullable=False)
    prev_vel_y = Column(Float, nullable=False)
    prev_vel_height = Column(Float, nullable=False)
    normalized_Time_Frame = Column(Float, nullable=False)
    temporal_TrackID = Column(Float, nullable=False)
    pos_x_rolling_avg = Column(Float, nullable=False)
    pos_y_rolling_avg = Column(Float, nullable=False)
    aspect_ratio_rolling_avg = Column(Float, nullable=False)
    height_rolling_avg = Column(Float, nullable=False)
    vel_x_rolling_avg = Column(Float, nullable=False)
    vel_y_rolling_avg = Column(Float, nullable=False)
    vel_aspect_rolling_avg = Column(Float, nullable=False)
    vel_height_rolling_avg = Column(Float, nullable=False)
    feature_mean_rolling_avg = Column(Float, nullable=False)
    feature_std_rolling_avg = Column(Float, nullable=False)
    feature_max_rolling_avg = Column(Float, nullable=False)
    cov_determinant_rolling_avg = Column(Float, nullable=False)
    prev_vel_x_rolling_avg = Column(Float, nullable=False)
    prev_vel_aspect_rolling_avg = Column(Float, nullable=False)
    accel_x_rolling_avg = Column(Float, nullable=False)
    accel_y_rolling_avg = Column(Float, nullable=False)
    accel_aspect_rolling_avg = Column(Float, nullable=False)
    accel_height_rolling_avg = Column(Float, nullable=False)

#%%

def create_sqlalchemy_engine():
    try:
        # Use environment variables for security
        db_username = 'abhishek'
        db_password = 'UXoQJBJ7ujSLTbXANtu0'
        db_host = 'basketballintelligence-dev.cn0ycgwe8d0i.us-east-2.rds.amazonaws.com'
        db_port = 5432
        db_name = 'postgres'

        # Create the connection string
        connection_string = f"postgresql://{db_username}:{db_password}@{db_host}:{db_port}/{db_name}"

        # Create the SQLAlchemy engine
        engine = create_engine(connection_string)

        return engine

    except Exception as e:
        print(f"Error creating SQLAlchemy Engine connection: {e}")
        return None

#%%

def execute_sql(conn, sql_command):
    """
    Objective:
    Execute a SQL command on the given database connection.

    Parameters:
    [psycopg2.extensions.connection] conn - The database connection
    [str] sql_command - The SQL command to execute

    Returns:
    [bool] success - True if the command was executed successfully, False otherwise
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql_command)
        conn.commit()
        logging.info("SQL command executed successfully")
        return True
    except Exception as e:
        logging.error(f"Error executing SQL command: {e}")
        conn.rollback()
        return False

#%% Configuring logging
try:
    try:
        log_file_path = os.path.join(log_dir, 'data_loading.log')
    except:
        log_file_path = 'data_loading.log'

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

def preprocess_dataset(df):
    try:
        # Convert data types
        df = df.astype({
            'Frame': int,
            'Age': int,
            'is_Team_A': bool,
            'is_Team_B': bool,
            'is_Basketball': bool,
            'state_tentative': bool,
            'state_confirmed': bool,
            'key_frame_s': bool,
            'OcclusionFrequency': float,
            'DetectionConsistency': float,
            'time_since_start': float,
            'prev_vel_y': float,
            'prev_vel_height': float,
            'Normalized_Time_Frame': float,
            'Temporal_TrackID': float,
            'pos_x_rolling_avg': float,
            'pos_y_rolling_avg': float,
            'aspect_ratio_rolling_avg': float,
            'height_rolling_avg': float,
            'vel_x_rolling_avg': float,
            'vel_y_rolling_avg': float,
            'vel_aspect_rolling_avg': float,
            'vel_height_rolling_avg': float,
            'feature_mean_rolling_avg': float,
            'feature_std_rolling_avg': float,
            'feature_max_rolling_avg': float,
            'cov_determinant_rolling_avg': float,
            'prev_vel_x_rolling_avg': float,
            'prev_vel_aspect_rolling_avg': float,
            'accel_x_rolling_avg': float,
            'accel_y_rolling_avg': float,
            'accel_aspect_rolling_avg': float,
            'accel_height_rolling_avg': float
        })

        # Replace NaN values with None
        df = df.replace({np.nan: None})

        return df

    
    except Exception as e:
        logger.error (f"Error occured when pre-processing dataset: {e}")
        raise

#%%

def run_quick_SQL(conn):
    """
    Objective:
    Run a specific SQL command generally to update data structure

    Parameters:
    [psycopg2.extensions.connection] conn - The database connection

    Returns:
    [bool] success - True if the table was altered successfully, False otherwise
    """
    try:
        sql_command = """
        # SQL COMMAND
        """
        return execute_sql(conn, sql_command)
    except Exception as e:
        logger.error(f"Error in performing command: {e}")
        return False

def main():
    try:
        engine = create_sqlalchemy_engine()

        # Import object tracked dataset into a dataframe
        processed_dataset_file_path = 'assets/processed_features.csv'
        raw_dataset = read_dataframe_to_csv(processed_dataset_file_path)
        processed_dataset = preprocess_dataset(raw_dataset)

        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()

        # Add data to spatial data structure
        try:
            for _, row in processed_dataset.iterrows():
                try:
                    tracking_data = TrackingData(
                        frame=row["Frame"],
                        age=row["Age"],
                        is_Team_A=row["is_Team_A"],
                        is_Team_B=row["is_Team_B"],
                        is_Basketball=row["is_Basketball"],
                        state_tentative=row["state_tentative"],
                        state_confirmed=row["state_confirmed"],
                        key_frame_s=row["key_frame_s"],
                        occlusionFrequency=row["OcclusionFrequency"],
                        detectionConsistency=row["DetectionConsistency"],
                        time_since_start=row["time_since_start"],
                        prev_vel_y=row["prev_vel_y"],
                        prev_vel_height=row["prev_vel_height"],
                        normalized_Time_Frame=row["Normalized_Time_Frame"],
                        temporal_TrackID=row["Temporal_TrackID"],
                        pos_x_rolling_avg=row["pos_x_rolling_avg"],
                        pos_y_rolling_avg=row["pos_y_rolling_avg"],
                        aspect_ratio_rolling_avg=row["aspect_ratio_rolling_avg"],
                        height_rolling_avg=row["height_rolling_avg"],
                        vel_x_rolling_avg=row["vel_x_rolling_avg"],
                        vel_y_rolling_avg=row["vel_y_rolling_avg"],
                        vel_aspect_rolling_avg=row["vel_aspect_rolling_avg"],
                        vel_height_rolling_avg=row["vel_height_rolling_avg"],
                        feature_mean_rolling_avg=row["feature_mean_rolling_avg"],
                        feature_std_rolling_avg=row["feature_std_rolling_avg"],
                        feature_max_rolling_avg=row["feature_max_rolling_avg"],
                        cov_determinant_rolling_avg=row["cov_determinant_rolling_avg"],
                        prev_vel_x_rolling_avg=row["prev_vel_x_rolling_avg"],
                        prev_vel_aspect_rolling_avg=row["prev_vel_aspect_rolling_avg"],
                        accel_x_rolling_avg=row["accel_x_rolling_avg"],
                        accel_y_rolling_avg=row["accel_y_rolling_avg"],
                        accel_aspect_rolling_avg=row["accel_aspect_rolling_avg"],
                        accel_height_rolling_avg=row["accel_height_rolling_avg"]
                    )
                except:
                    print('failed')
                session.add(tracking_data)

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting data: {e}")
        finally:
            session.close()


        logger.info("Data Loading successful")

    except:
        logger.error(f"Error in main function: {e}")
        print("Error in main function")

#%% Main execution
if __name__ == "__main__":
    main()


#%% 
# Setup commands to be executed #


# create_basketball_table() - Created tracking_data table
'''
def create_basketball_table(conn):
    """
    Objective:
    Create a table to store basketball game data.

    Parameters:
    [psycopg2.extensions.connection] conn - The database connection

    Returns:
    [bool] success - True if the table was created successfully, False otherwise
    """
    try:
        sql_command = """
        CREATE TABLE tracking_data (
            id SERIAL PRIMARY KEY,
            Frame INTEGER NOT NULL,
            Age INTEGER NOT NULL,
            is_Team_A BOOLEAN NOT NULL,
            is_Team_B BOOLEAN NOT NULL,
            is_Basketball BOOLEAN NOT NULL,
            state_tentative BOOLEAN NOT NULL,
            state_confirmed BOOLEAN NOT NULL,
            key_frame_s INTEGER NOT NULL,
            OcclusionFrequency INTEGER NOT NULL,
            DetectionConsistency INTEGER NOT NULL,
            time_since_start FLOAT NOT NULL,
            prev_vel_y FLOAT NOT NULL,
            prev_vel_height FLOAT NOT NULL,
            Normalized_Time_Frame FLOAT NOT NULL,
            Temporal_TrackID FLOAT NOT NULL,
            pos_x_rolling_avg FLOAT NOT NULL,
            pos_y_rolling_avg FLOAT NOT NULL,
            aspect_ratio_rolling_avg FLOAT NOT NULL,
            height_rolling_avg FLOAT NOT NULL,
            vel_x_rolling_avg FLOAT NOT NULL,
            vel_y_rolling_avg FLOAT NOT NULL,
            vel_aspect_rolling_avg FLOAT NOT NULL,
            vel_height_rolling_avg FLOAT NOT NULL,
            feature_mean_rolling_avg FLOAT NOT NULL,
            feature_std_rolling_avg FLOAT NOT NULL,
            feature_max_rolling_avg FLOAT NOT NULL,
            cov_determinant_rolling_avg FLOAT NOT NULL,
            prev_vel_x_rolling_avg FLOAT NOT NULL,
            prev_vel_aspect_rolling_avg FLOAT NOT NULL,
            accel_x_rolling_avg FLOAT NOT NULL,
            accel_y_rolling_avg FLOAT NOT NULL,
            accel_aspect_rolling_avg FLOAT NOT NULL,
            accel_height_rolling_avg FLOAT NOT NULL,

            CONSTRAINT unique_frame_object UNIQUE (Temporal_TrackID, is_Team_A, is_Team_B, is_Basketball),
            CONSTRAINT check_team_basketball CHECK (
                (CASE WHEN is_Team_A THEN 1 ELSE 0 END +
                CASE WHEN is_Team_B THEN 1 ELSE 0 END +
                CASE WHEN is_Basketball THEN 1 ELSE 0 END) = 1
            ),
            CONSTRAINT check_occlusion_frequency CHECK (OcclusionFrequency >= 0 AND OcclusionFrequency <= 1),
            CONSTRAINT check_detection_consistency CHECK (DetectionConsistency >= 0 AND DetectionConsistency <= 1),
            CONSTRAINT check_normalized_time CHECK (Normalized_Time_Frame >= 0 AND Normalized_Time_Frame <= 1)
        );
        """

        return execute_sql(conn, sql_command)
    

    except Exception as e:
        logging.error(f"Error creating basketball table: {e}")
        return False
'''

#create_spatial_index() - Created spatial index based on x, y coordinate (rolling average)
'''
def create_spatial_index(conn):
    """
    Objective:
    Create a spatial index on the tracking_data table.

    Parameters:
    [psycopg2.extensions.connection] conn - The database connection

    Returns:
    [bool] success - True if the index was created successfully, False otherwise
    """
    try:
        sql_command = """
        CREATE INDEX idx_spatial_position ON tracking_data 
        USING GIST (ST_SetSRID(ST_MakePoint(pos_x_rolling_avg, pos_y_rolling_avg), 4326));
        """
        return execute_sql(conn, sql_command)
    except Exception as e:
        logging.error(f"Error creating spatial index: {e}")
        return False
'''

#enable_postgis() - Enable PostGIS extension for spatial databases
'''
def enable_postgis(conn):
    """
    Objective:
    Enable the PostGIS extension in the database.

    Parameters:
    [psycopg2.extensions.connection] conn - The database connection

    Returns:
    [bool] success - True if PostGIS was enabled successfully, False otherwise
    """
    try:
        sql_command = "CREATE EXTENSION IF NOT EXISTS postgis;"
        return execute_sql(conn, sql_command)
    except Exception as e:
        logging.error(f"Error enabling PostGIS extension: {e}")
        return False
'''



