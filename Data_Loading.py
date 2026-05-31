'''
Data Loading
This file creates the loads the dataset (after feature engineering) into AWS spatial database

Key Concepts Implemented:
- Constraints / ACID Properties
- Spatial Index
- Test the execution time of proximity queries
'''

#%% Import Libraries
import math
import numpy as np
import pandas as pd
import logging
import os
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, Float, Boolean, text
from sqlalchemy.orm import declarative_base, sessionmaker

import config
from sqlalchemy.exc import SQLAlchemyError
from geoalchemy2 import Geometry
from geoalchemy2.elements import WKTElement

# Module-level log directory so the logging block below has a defined value
# (it previously referenced an undefined `log_dir`); respects LOG_DIR in CI/Docker.
log_dir = os.environ.get('LOG_DIR', os.getcwd())

#%% Create a base class

Base = declarative_base()

class TrackingData(Base):
    "TrackingData class to store the basketball data"

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
    # PostGIS point built from the rolling-average position; GIST-indexed for
    # spatial proximity queries (see run_commit_query).
    point_geom = Column(Geometry(geometry_type='POINT', srid=0), nullable=True)

#%% Create connection to database

def get_connection_string():
    """
    Objective:
    Resolve the PostgreSQL connection string. Honors the DATABASE_URL env var
    first, then falls back to the centralized config.DB_URL (which itself reads
    the DB_URL env var / project default). Kept separate from engine creation so
    the env-driven selection is unit-testable without a live database.

    Returns:
    [str] connection_string - SQLAlchemy database URL
    """
    return os.environ.get('DATABASE_URL', config.DB_URL)

def create_sqlalchemy_engine():
    """ 
    Objective: 
    Creates a SQLAlchemy engine for a PostgreSQL database

    Parameters: 
    None

    Returns: 
    [sqlalchemy.engine.base.Engine] engine - SQLAlchemy database engine
    """

    try:
        # Connection string is sourced from the DATABASE_URL env var (no hardcoded
        # credentials); see get_connection_string for the default fallback.
        connection_string = get_connection_string()

        try:
            # Create the SQLAlchemy engine
            engine = create_engine(connection_string, pool_pre_ping=True)
        except Exception as e:
            print(f"Error creating SQLAlchemy Engine connection: {e}")
            return None

        return engine

    except Exception as e:
        print(f"Error creating SQLAlchemy Engine connection: {e}")
        return None

#%% Create CSV file

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

#%% Preprocess dataset

def preprocess_dataset(df):
    """ 
    Objective: 
    Preprocesses the input dataset for further analysis.

    Parameters: 
    [DataFrame] dataset - The input dataset to be preprocessed.

    Returns: 
    [DataFrame] preprocessed_dataset - The preprocessed dataset. 
    """

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

#%% Create spatial data structure

def spatial_data_structure(data_row):
    """ 
    Objective: 
    Generates a spatial data structure for efficient spatial queries

    Parameters: 
    [pandas DataFrame] points - The input dataset of points to be structured

    Returns: 
    [Spatial Data Structure] tracking_data - The generated spatial data structure
    """

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
            accel_height_rolling_avg=data_row["accel_height_rolling_avg"],
            # Spatial point from the rolling-average position for PostGIS queries
            point_geom=WKTElement(
                f"POINT({data_row['pos_x_rolling_avg']} {data_row['pos_y_rolling_avg']})",
                srid=0,
            ),
        )

        return tracking_data
    
    except Exception as e:
        logger.error(f"Failed storing data into tracking_data structure: {e}")
        raise



#%% Run select query

def run_select_query(engine, sql_command, params=None):
    """ 
    Objective: 
    Runs a SELECT SQL query on a given database connection.

    Parameters: 
    [SQLAlchemy Engine] engine - The database connection. 
    [str] sql_command - The SELECT SQL query to be executed.
    [dict] params - The parameters to be passed to the query.

    Returns: 
    [pandas DataFrame] result - The result of the query as a DataFrame. 
    """

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

#%% Run a specific commit query

def run_commit_query(engine):
    """ 
    Objective: 
    Executes a COMMIT SQL query on a database connection.

    Parameters: 
    [SQLAlchemy Engine] engine - The database connection.

    Returns: 
    None
    """

    try:
        # SQL command to create the GIST spatial index on the PostGIS point column
        sql_command = text("""
            CREATE INDEX IF NOT EXISTS idx_tracking_data_point_geom
            ON public.tracking_data
            USING GIST (point_geom);
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

#%% Get basketball details for frame

def get_basketball_for_frame(engine, frame):
    """ 
    Objective: 
    Retrieves basketball data for a specific frame.

    Parameters: 
    [SQLAlchemy Engine] engine - The database connection.
    [int] frame - The frame number to retrieve data for.

    Returns: 
    [pandas DataFrame] data - The basketball data for the specified frame. 
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

#%% Pure distance helper

def calculate_point_distances(basketball_xy, player_rows):
    """
    Objective:
    Compute Euclidean distances from the basketball to each player and return
    them ranked nearest-first. Pure function (no DB) so it is unit-testable.

    Parameters:
    [tuple] basketball_xy - (x, y) of the basketball
    [list] player_rows - list of (player_id, x, y) tuples

    Returns:
    [list] ranked - list of (player_id, distance) sorted ascending by distance
    """
    bx, by = basketball_xy
    distances = [
        (player_id, math.hypot(px - bx, py - by)) for player_id, px, py in player_rows
    ]
    return sorted(distances, key=lambda item: item[1])

#%% Calculate basketball distance within database

def calculate_basketball_distances(engine, n_closest=3):
    """ 
    Objective: 
    Calculates the distances between basketball and players in a given dataset.

    Parameters: 
    [SQLAlchemy Engine] engine - The database connection.

    Returns:
    None
    """

    try:
        # Collect the distinct frame_ids
        frames_list = run_select_query(
            engine,
            text("SELECT DISTINCT frame FROM public.tracking_data ORDER BY frame;"),
        )

        # Per-frame query: id, x, y (extracted from the PostGIS point) and class
        objects_sql = text("""
            SELECT id, ST_X(point_geom), ST_Y(point_geom), is_basketball
            FROM public.tracking_data
            WHERE frame = :frame;
        """)

        results = {}
        for (frame_id,) in frames_list:
            rows = run_select_query(engine, objects_sql, frame_id)

            # Split the frame's objects into the basketball and the players
            basketball_xy = None
            player_rows = []
            for obj_id, x, y, is_basketball in rows:
                if is_basketball:
                    basketball_xy = (x, y)
                else:
                    player_rows.append((obj_id, x, y))

            # No basketball detected in this frame -> nothing to rank
            if basketball_xy is None:
                continue

            # Rank players by proximity and keep the n closest
            ranked = calculate_point_distances(basketball_xy, player_rows)[:n_closest]
            results[frame_id] = ranked

        return results

    except SQLAlchemyError as e:
        logging.error(f"SQLAlchemy error occurred while calculating distances: {str(e)}")
        raise

#%% Configuring logging
try:
    try:
        log_file_path = os.path.join(log_dir, 'data_loading.log')
    except Exception as e:
        print(f"Error in creating logging: {e}")
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

#%% Main function
def main():

    try:
        # Create SQLAlchemy engine
        engine = create_sqlalchemy_engine()

        # Import object tracked dataset into a dataframe
        processed_dataset_file_path = 'assets/processed_features.csv'
        raw_dataset = read_dataframe_to_csv(processed_dataset_file_path)
        processed_dataset = preprocess_dataset(raw_dataset)

        # Create session
        Session = sessionmaker(bind=engine)
        session = Session()

        #'''
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
        #'''

        # Calculate basketball distances
        calculate_basketball_distances(engine)

        run_commit_query(engine)

        session.close()

        logger.info("Data Loading successful")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        print("Error in main function")

#%% Main execution
if __name__ == "__main__":
    main()