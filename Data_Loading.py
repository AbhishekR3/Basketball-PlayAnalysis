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
import logging
import os
import threading
import psycopg2
from psycopg2 import sql
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, Float, Boolean, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.exc import SQLAlchemyError
from geoalchemy2 import Geometry

#%% 

Base = declarative_base()

class TrackingData(Base):
    __tablename__ = 'tracking_data'

    id = Column(Integer, primary_key=True)
    frame = Column(Integer, nullable=False)
    age = Column(Integer, nullable=False)
    is_team_a = Column(Boolean, nullable=False)
    is_team_b = Column(Boolean, nullable=False)
    is_basketball = Column(Boolean, nullable=False)
    state_tentative = Column(Boolean, nullable=False)
    state_confirmed = Column(Boolean, nullable=False)
    key_frame_s = Column(Boolean, nullable=False)
    occlusionfrequency = Column(Float, nullable=False)
    detectionconsistency = Column(Float, nullable=False)
    time_since_start = Column(Float, nullable=False)
    prev_vel_y = Column(Float, nullable=False)
    prev_vel_height = Column(Float, nullable=False)
    normalized_time_frame = Column(Float, nullable=False)
    temporal_trackid = Column(Float, nullable=False)
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
        engine = create_engine(connection_string, pool_pre_ping=True)

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
        raw_dataset = pd.read_csv(file_path, index_col=None)

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

def spatial_data_structure(data_row):
    # Add data to spatial data structure
    try:
        tracking_data = TrackingData(
            frame=data_row["Frame"],
            age=data_row["Age"],
            is_team_a=data_row["is_Team_A"],
            is_team_b=data_row["is_Team_B"],
            is_basketball=data_row["is_Basketball"],
            state_tentative=data_row["state_tentative"],
            state_confirmed=data_row["state_confirmed"],
            key_frame_s=data_row["key_frame_s"],
            occlusionfrequency=data_row["OcclusionFrequency"],
            detectionconsistency=data_row["DetectionConsistency"],
            time_since_start=data_row["time_since_start"],
            prev_vel_y=data_row["prev_vel_y"],
            prev_vel_height=data_row["prev_vel_height"],
            normalized_time_frame=data_row["Normalized_Time_Frame"],
            temporal_trackid=data_row["Temporal_TrackID"],
            pos_x_rolling_avg=data_row["pos_x_rolling_avg"],
            pos_y_rolling_avg=data_row["pos_y_rolling_avg"],
            aspect_ratio_rolling_avg=data_row["aspect_ratio_rolling_avg"],
            height_rolling_avg=data_row["height_rolling_avg"],
            vel_x_rolling_avg=data_row["vel_x_rolling_avg"],
            vel_y_rolling_avg=data_row["vel_y_rolling_avg"],
            vel_aspect_rolling_avg=data_row["vel_aspect_rolling_avg"],
            vel_height_rolling_avg=data_row["vel_height_rolling_avg"],
            feature_mean_rolling_avg=data_row["feature_mean_rolling_avg"],
            feature_std_rolling_avg=data_row["feature_std_rolling_avg"],
            feature_max_rolling_avg=data_row["feature_max_rolling_avg"],
            cov_determinant_rolling_avg=data_row["cov_determinant_rolling_avg"],
            prev_vel_x_rolling_avg=data_row["prev_vel_x_rolling_avg"],
            prev_vel_aspect_rolling_avg=data_row["prev_vel_aspect_rolling_avg"],
            accel_x_rolling_avg=data_row["accel_x_rolling_avg"],
            accel_y_rolling_avg=data_row["accel_y_rolling_avg"],
            accel_aspect_rolling_avg=data_row["accel_aspect_rolling_avg"],
            accel_height_rolling_avg=data_row["accel_height_rolling_avg"]
        )

        return tracking_data
    
    except:
        logger.error(f"Failed storing data into tracking_data structure: {e}")
        raise



#%%

#run_select_query() - Run a select query
def run_select_query(engine, sql_command, params=None):
    try:
        with engine.connect() as connection:
            # Convert sql_command to SQL text
            if isinstance(sql_command, str):
                sql_command = text(sql_command)
            
            # If parameters exist, execute code with parameters 
            if params:
                result = connection.execute(sql_command, {"frame": params})
            else:
                result = connection.execute(sql_command)

            rows = result.fetchall()

            return rows
    
    except SQLAlchemyError as e:
        logging.error(f"SQLAlchemy error occurred while performin select query: {str(e)}")
        raise

    except:
        print('Messed up the select statement')
        logger.error(f"Error in running select query: {e}")
        raise

#run_commit_query() - Run a commit query
def run_commit_query(engine):
    try:
        # SQL command to create the spatial index
        sql_command = text("""

        """)

        # Execute the SQL command
        with engine.connect() as connection:
            connection.execute(sql_command)
            connection.commit()

    except SQLAlchemyError as e:
        logging.error(f"SQLAlchemy error occurred while committing the query: {str(e)}")
        raise

    except:
        print('Messed up the commit statement')
        logger.error(f"Error in committing the above query: {e}")
        raise

def get_basketball_for_frame(engine, frame):
    """
    Objective:
    Retrieve the basketball object for a given frame.

    Parameters:
    engine (sqlalchemy.engine.base.Engine): SQLAlchemy database engine
    frame_id (int): The frame number to query

    Returns:
    tuple: A tuple containing (id, point_geom) for the basketball, or None if not found
    """
    try:
        sql_command = text("""
            SELECT id, point_geom
            FROM public.tracking_data
            WHERE is_basketball = TRUE AND frame = :frame;
        """)
        
        result = run_select_query(engine, sql_command, frame)
        
        return result[0] if result else None
    
    except Exception as e:
        logging.error(f"Error occurred while getting basketball for frame {frame}: {str(e)}")
        raise

########## CREATE A SEPARATE FILE TO INPUT BASKETBALL DISTANCE
def calculate_basketball_distances(engine):
    try:
        # Collect the distinct frame_ids
        sql_command = text("""
        SELECT DISTINCT frame FROM public.tracking_data ORDER BY frame;
        """)

        frames_list = run_select_query(engine, sql_command)

        for (frame_id, ) in frames_list:
            basketball_data = get_basketball_for_frame(engine, frame_id)
            
            # Find the basketball info for this frame
            
            if basketball_data is not None:
                basketball_id, basketball_geom = basketball_data
                print(basketball_id, basketball_geom)
                # Calculate distances to all other objects in the same frame
                ### INSERT Query with variables

                '''
                cur.execute("""
                    INSERT INTO basketball_distances (frame_id, basketball_id, object_id, distance, rank)
                    SELECT 
                        %s,
                        %s,
                        o.id,
                        ST_Distance(ST_GeomFromEWKB(%s), o.point_geom),
                        ROW_NUMBER() OVER (ORDER BY ST_Distance(ST_GeomFromEWKB(%s), o.point_geom))
                    FROM objects_table o
                    WHERE o.frame = %s AND o.id != %s
                    ORDER BY ST_Distance(ST_GeomFromEWKB(%s), o.point_geom) ASC
                    LIMIT 3;
                """, (frame_id, basketball_id, basketball_geom, basketball_geom, frame_id, basketball_id, basketball_geom))
                '''
            #conn.commit()
        

    except SQLAlchemyError as e:
        logging.error(f"SQLAlchemy error occurred while committing the query: {str(e)}")
        raise

    except:
        print('Messed up the commit statement')
        logger.error(f"Error in committing the above query: {e}")
        raise

####################################################################
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

        '''
        # Add data to spatial data structure
        try:
            print('Starting data input')
            for _, row in processed_dataset.iterrows():
                structured_data = spatial_data_structure(row)
                session.add(structured_data)

            session.commit()
            print('Completed data input')
            logger.info(f"Committed dataset to data structure")
            # UPDATED: Include a log info of how many rows was imported + data table size

        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting data: {e}")
        '''

        #calculate_basketball_distances(engine)

        #run_select_query(engine)
        run_commit_query(engine)

        session.close()

        logger.info("Data Loading successful")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print("Error in main function")

#%% Main execution
if __name__ == "__main__":
    main()


#%% Database setup commands executed #%%

#setup query
'''
Spatial Index
1. Create point_geom which represent the geospatial data of point_geom
ALTER TABLE tracking_data ADD COLUMN point_geom geometry(POINT);

2. Set the values in point_geom
UPDATE tracking_data
SET point_geom = ST_MakePoint(pos_x_rolling_avg, pos_y_rolling_avg);

3. Create spatial index
CREATE INDEX idx_tracking_data_point_geom ON tracking_data USING GIST (point_geom);

'''


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
            frame INTEGER NOT NULL,
            age INTEGER NOT NULL,
            is_team_a BOOLEAN NOT NULL,
            is_team_b BOOLEAN NOT NULL,
            is_basketball BOOLEAN NOT NULL,
            state_tentative BOOLEAN NOT NULL,
            state_confirmed BOOLEAN NOT NULL,
            key_frame_s BOOLEAN NOT NULL,
            occlusionfrequency FLOAT NOT NULL,
            detectionconsistency FLOAT NOT NULL,
            time_since_start FLOAT NOT NULL,
            prev_vel_y FLOAT NOT NULL,
            prev_vel_height FLOAT NOT NULL,
            normalized_time_frame FLOAT NOT NULL,
            temporal_trackid FLOAT NOT NULL,
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

            CONSTRAINT unique_frame_object UNIQUE (temporal_trackid, is_team_a, is_team_b, is_basketball),
            CONSTRAINT check_team_basketball CHECK (
                (CASE WHEN is_team_a THEN 1 ELSE 0 END +
                CASE WHEN is_team_b THEN 1 ELSE 0 END +
                CASE WHEN is_basketball THEN 1 ELSE 0 END) = 1
            ),
            CONSTRAINT check_occlusion_frequency CHECK (occlusionfrequency >= 0 AND occlusionfrequency <= 1),
            CONSTRAINT check_detection_consistency CHECK (detectionconsistency >= 0 AND detectionconsistency <= 1),
            CONSTRAINT check_normalized_time CHECK (normalized_time_frame >= 0 AND normalized_time_frame <= 1)
        );
        """

        return execute_sql(conn, sql_command)
    

    except Exception as e:
        logging.error(f"Error creating basketball table: {e}")
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