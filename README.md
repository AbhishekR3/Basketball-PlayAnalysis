# PlayBook AI: Basketball Intelligence

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/caa2d542ea8e47b597b3712cbc4236cb)](https://app.codacy.com/gh/AbhishekR3/Basketball-PlayAnalysis/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Release Badge](https://img.shields.io/github/v/release/AbhishekR3/Basketball-PlayAnalysis.svg?color=orange)](https://github.com/AbhishekR3/Basketball-PlayAnalysis/releases)

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)


## Project-Description

This project aims to analyze and classify the type of offensive and defensive plays executed by basketball teams.
This will be performed using various algorithms such as multi-object tracking, spatial databases, and neural networks/transformers.

The movement of players and objects is similar to the data displayed on CourtVision by the LA Clippers.

CourtVision Sample Frame

![CourtVision Sample Frame](https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/main/assets/Clippers%20CourtVision.png)

Object Tracking on a Basketball Simulation created with Pygame

![ObjectTracking_Demo](https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/DEV_Code/assets/ObjectTracking%20Demo.gif)


![DataFlowDiagram](assets/PlayBook-AI%20Data%20Flow%20Diagram.png)


For the project outline containing detailed information on the relevant concepts/algorithms applied in this project, please refer the following
https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/main/Basketball%20Play%20Classification%20Project%20Outline
(Do note this outline is also used for project planning and note taking)

## Installation

### Prerequisites

- Python 3.x (latest version recommended)

### Dependencies

Ensure you have the following packages installed:

Refer Requirements.txt file and install the libraries mentioned

Clone the Repository
```bash
git clone https://github.com/AbhishekR3/Basketball-PlayAnalysis.git
cd Basketball-PlayAnalysis
```

## Usage

These files were built on a MacOS build. 
First run Basketball_Game_Simulation.py to generate video simulation
Next run Basketball_Object_Tracking.py for object tracking for the relevant video simulation

```bash
python Basketball_Passing_Simulation.py
python Basketball_Object_Tracking.py
```

## Project-Structure
Important files for this project

```bash
Basketball-PlayAnalysis/
├── assets/                                         # Containing referenced images and diagrams
│   ├── simulation.mp4
│   ├── simulation_tracked.mp4
│   ├── detected_objects.csv                        # Features of the detected objects in the simulation
│   ├── PlayBook-AI Data Flow Diagram.png           # PlayBook-AI Data Flow Diagram
├── Custom_Detection_Model/                         # Custom Model related files such as training/validation
│   ├── CustomObjectDetection_Data/                 # Training data and Validation Results for custom YOLO object detection model
│   ├── Object Tracking Metrics/                    # Multi-Object Tracking (DeepSORT) validation
├── deep_sort/                                      # DeepSORT related files (Mutli-Object Tracking)
│   ├── deep_sort/
│   ├── tools/
│   ├── model_data/
├── References/                                     # References for the development of the project
├── Basketball_Passing_Simulation.py                # Script for simulating basketball plays
├── Basketball_Object_Tracking.py                   # Script for tracking objects in the simulation
├── object_tracking_output.log                      # Object Tracking Log Details containing relevant metrics
├── PlayBook AI: Basketball Intelligence Outline    # Project Outline
├── README.md                                       # Project documentation
├── Requirements.txt                                # Project library requirements
├── YOLOv10m_custom.pt                              # Custom YOLO detection model based on YOLOv10m
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) License - see the LICENSE file for details.

