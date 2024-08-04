'''
Multi Object Tracking Validation file
This file calculates the following metrics:
- MOTA - Evaluates overall tracking accuracy
- MOTP - Assesses the spatial precision
- IDF1 - Measures the accuracy of maintaining correct object identities
'''

#%%

#Import Libraries
import pandas as pd
import defusedxml.ElementTree as ET
import numpy as np
from collections import defaultdict
import motmetrics as mm


#%%

def parse_cvat_xml(xml_file):
    """
    Objective:
    Parse the XML file of the annotated frames to an easier processable format
    
    Parameters:
    [str] xml_file - File path of the ground truth data in an xml file
    
    Returns:
    [dictionary] annotations - Dictionary collection of the annotations 
    """
    
    # Parse data from file path
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Create data storage for annotations (ground truth data)
    annotations = defaultdict(list)

    # Extract data
    for track in root.findall('.//track'):
        track_id = int(track.get('id'))
        label = track.get('label')
        for box in track.findall('box'):
            frame = int(box.get('frame'))
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            annotations[frame].append({
                'id': track_id,
                'bbox': [xtl, ytl, xbr, ybr],
                'class': [label]
            })

    return annotations

#%%

def annotation_dataframe(annotations):
    """
    Objective:
    Convert annotations into panads dataframe
    
    Parameters:
    [dictionary] annotations - Dictionary collection of the annotations

    Returns:
    [dataframe] annotation_data - Pandas dataframe to contain annotated data
    """

    nth_entry = 0

    data = []
    for key1 in annotations:
        for key2 in annotations[key1]:
            # Extract the values
            id_value = int(key2['id'])+1 #Object ID-Value
            bbox_value = key2['bbox'] # BBox coordiantes (Top Left, Bottom Right)
            class_value = key2['class'] #Class Value
            n_frame = key1 # Nth Frame
            nth_entry += 1
            
            # Append the values as a tuple to the rows list
            data.append((nth_entry, id_value, bbox_value, class_value, n_frame))

    # Convert data to dataframe
    annotation_data = pd.DataFrame(data, columns=['ID', 'TrackID', 'BBox', 'ClassID', 'Frame'])

    return annotation_data

#%%

def string_to_array(string_data):
    """
    Objective:
    Convert string to array

    Parameters:
    [array] string_data -  BBox in string format

    Returns:
    [array] array_data - BBox in array format
    """

    string_data = string_data.strip('[]').split()
    array_data = np.array(string_data, dtype=float)

    return array_data

#%%

def calculate_tracking_metrics(ds_df, gt_df):
    """
    Objective:
    Calculate MOTA, MOTP, and IDF1 metrics
    
    Parameters:
    [dataframe] ds_df - Dataframe containing the features extracted from DeepSORT algorithm
    [dataframe] gt_df - Pandas dataframe to contain annotated data

    Returns:
    [dict] MOTA, MOTP, and IDF1 scores
    """

    # Create an accumulator object
    acc = mm.MOTAccumulator(auto_id=True)

    # Group data by frame
    ds_by_frame = ds_df.groupby('Frame')
    gt_by_frame = gt_df.groupby('Frame')

    # Process each frame
    for frame, frame_data in ds_by_frame:
        # Get ground truth for this frame
        gt_frame = gt_by_frame.get_group(frame) if frame in gt_by_frame.groups else pd.DataFrame()

        # Extract object IDs and bounding boxes
        track_ids = frame_data['TrackID'].values
        gt_ids = gt_frame['TrackID'].values if not gt_frame.empty else []

        # Extract bounding boxes
        track_bboxes = np.stack(frame_data['BBox'].values)
        gt_bboxes = np.stack(gt_frame['BBox'].values) if not gt_frame.empty else np.empty((0, 4))

        # Calculate IoU distances
        distances = mm.distances.iou_matrix(gt_bboxes, track_bboxes, max_iou=0.5)

        # Update accumulator
        acc.update(gt_ids, track_ids, distances)

    # Compute metrics
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=['mota', 'motp', 'idf1'], name='acc')

    return {
        'MOTA': '{:.4g}'.format(summary['mota']['acc']),
        'MOTP': '{:.4g}'.format(summary['motp']['acc']),
        'IDF1': '{:.4g}'.format(summary['idf1']['acc'])
    }

#%%

def main():
    # Load and convert ground-truth data to pandas dataframe
    groundtruth_xml = 'Custom_Detection_Model/Object Tracking Metrics/annotations_10s.xml'
    annotations = parse_cvat_xml(groundtruth_xml)
    groundtruth_data = annotation_dataframe(annotations)

    # Load and convert object detections's features to pandas dataframe
    DeepSORT_data_file = 'Custom_Detection_Model/Object Tracking Metrics/MOT_validationmetrics.csv'
    DeepSORT_data = pd.read_csv(DeepSORT_data_file)
    DeepSORT_data['BBox'] = DeepSORT_data['BBox'].apply(string_to_array)
    
    # Calculate validation metrics
    validation_metrics = calculate_tracking_metrics(DeepSORT_data, groundtruth_data)
    print(validation_metrics)

if __name__ == "__main__":
    main()