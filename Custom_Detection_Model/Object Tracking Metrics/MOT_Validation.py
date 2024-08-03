'''
Multi Object Tracking Validation file
This file calculates the following metrics:
- MOTA - Evaluates overall tracking accuracy
- MOTP - Assesses the spatial precision
- FID1 - Measures the accuracy of maintaining correct object identities
'''

#%%

#Import Libraries
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

#%%

def parse_cvat_xml(xml_file):
    """
    Objective:
    Parse the XML file of the annotated frames from CVAT
    
    Parameters:
    [???] xml_file - 
    
    Returns:
    [dictionary] annotations - Dictionary collection of the annotations 
    """
        
    tree = ET.parse(xml_file)
    root = tree.getroot()

    annotations = defaultdict(list)
    
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

def annotation_dataframe(annotations, annotation_data):
    """
    Objective:
    Convert annotations into panads dataframe
    
    Parameters:
    [dictionary] annotations - Dictionary collection of the annotations 
    [dataframe] annotation_data - Pandas dataframe to contain annotated data

    Returns:
    [dataframe] annotation_data - Pandas dataframe to contain annotated data
    """

    data = []
    for key1 in annotations:
        for key2 in annotations[key1]:
            # Extract the values
            id_value = key2['id']
            bbox_value = key2['bbox']
            class_value = key2['class']
            
            # Append the values as a tuple to the rows list
            data.append((id_value, bbox_value, class_value))

    annotation_data = pd.DataFrame(data, columns=['id', 'bbox', 'class'])

    return annotation_data

#%%

def load_detections(detections_file):
    """
    Objective:

    
    Parameters:

    
    Returns:

    """
    # Implement this function based on your detection data format
    # It should return a dictionary similar to the annotations
    pass

#%%

def calculate_iou(box1, box2):
    """
    Objective:

    
    Parameters:

    
    Returns:

    """
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    xi1, yi1 = max(x1, x3), max(y1, y3)
    xi2, yi2 = min(x2, x4), min(y2, y4)
    
    intersection = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0

#%%

def calculate_metrics(annotations, detections):
    """
    Objective:

    
    Parameters:

    
    Returns:

    """
    total_frames = max(max(annotations.keys()), max(detections.keys()))
    
    total_gt = sum(len(objs) for objs in annotations.values())
    total_detections = sum(len(objs) for objs in detections.values())
    
    matches = 0
    misses = 0
    false_positives = 0
    id_switches = 0
    total_iou = 0
    
    last_matched_ids = {}
    
    for frame in range(total_frames + 1):
        gt_objects = annotations.get(frame, [])
        det_objects = detections.get(frame, [])
        
        cost_matrix = np.zeros((len(gt_objects), len(det_objects)))
        for i, gt in enumerate(gt_objects):
            for j, det in enumerate(det_objects):
                cost_matrix[i, j] = 1 - calculate_iou(gt['bbox'], det['bbox'])
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] < 0.5:  # IOU threshold
                matches += 1
                total_iou += 1 - cost_matrix[i, j]
                
                gt_id = gt_objects[i]['id']
                det_id = det_objects[j]['id']
                
                if gt_id in last_matched_ids and last_matched_ids[gt_id] != det_id:
                    id_switches += 1
                
                last_matched_ids[gt_id] = det_id
            else:
                misses += 1
        
        false_positives += len(det_objects) - len(row_ind)
    
    mota = 1 - (misses + false_positives + id_switches) / total_gt
    motp = total_iou / matches if matches > 0 else 0
    
    # Calculate IDF1
    idtp = matches - id_switches
    idfp = total_detections - idtp
    idfn = total_gt - idtp
    idf1 = (2 * idtp) / (2 * idtp + idfp + idfn) if (2 * idtp + idfp + idfn) > 0 else 0
    
    return mota, motp, idf1

#%%

def main():

    # Load and convert annotations to pandas dataframe
    cvat_xml_file = 'Custom_Detection_Model/Object Tracking Metrics/annotations_10s.xml'
    annotation_data = pd.DataFrame(columns=['id', 'bbox', 'class'])
    annotations = parse_cvat_xml(cvat_xml_file)
    annotation_data = annotation_dataframe(annotations, annotation_data)

    # Load and convert object detections's features to pandas dataframe
    detections_file = 'path/to/your/detections.json'  # Adjust based on your format
    detections = load_detections(detections_file)
    
    mota, motp, idf1 = calculate_metrics(annotations, detections)
    
    print(f"MOTA: {mota:.4f}")
    print(f"MOTP: {motp:.4f}")
    print(f"IDF1: {idf1:.4f}")

if __name__ == "__main__":
    main()




