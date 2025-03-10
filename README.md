# PlayBook AI: Basketball Intelligence

[![Codacy Badge](https://app.codacy.com/project/badge/Grade/caa2d542ea8e47b597b3712cbc4236cb)](https://app.codacy.com/gh/AbhishekR3/Basketball-PlayAnalysis/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Release Badge](https://img.shields.io/github/v/release/AbhishekR3/Basketball-PlayAnalysis.svg?color=orange)](https://github.com/AbhishekR3/Basketball-PlayAnalysis/releases)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)
![AWS](https://img.shields.io/badge/Amazon_AWS-232F3E?style=flat&logo=amazon-web-services&logoColor=white)

## Table of Contents
- [Project Description](#project-description)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [License](#license)


## Project-Description

Project Goals:

PlayBook AI will help basketball professionals:
* Analyze offensive and defensive plays executed by teams
* Assess potential player fit within a team
* Understand play patterns in critical game moments
* Identify a team's most/least successful plays

Project Overview:
1. Simulate basketball plays with Pygame for training data
2. Implementing computer vision techniques for multi-object tracking
3. Leveraging spatial databases for efficient data management
4. Applying feature engineering for ML model optimization
5. Using Neural Networks (LSTM, TCN, C3D) and Transformers (TimeSformer, STN)
6. Enhancing model efficiency through quantization and pruning 
7. Deploying the software on AWS with GPU accelerated libraries and CI testing pipelines, and Docker for containerization

The movement of players and objects is similar to the data displayed on CourtVision by the LA Clippers.

CourtVision Sample Frame

![CourtVision Sample Frame](https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/main/assets/Clippers%20CourtVision.png)

Object Tracking on a Basketball Simulation created with Pygame

![ObjectTracking_Demo](https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/DEV_Code/assets/ObjectTracking%20Demo.gif)

Data Flow Diagram

![DataFlowDiagram](assets/PlayBook-AI%20Data%20Flow%20Diagram.png)

I have a project outline containing detailed information on the relevant concepts/algorithms planned for this project

[Refer the following](https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/main/PlayBook%20AI%3A%20Basketball%20Intelligence%20Outline)

## Installation

### Prerequisites

- Python 3.x (latest version recommended)

### Dependencies

Ensure you have the following installed:

1. Refer Requirements.txt file and install the libraries mentioned

2. Clone the Repository
```bash
git clone https://github.com/AbhishekR3/Basketball-PlayAnalysis.git
cd Basketball-PlayAnalysis
```

## Usage

These files were built on a MacOS build. 
First run Basketball_Passing_Simulation.py to generate video simulations of passes
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
│   ├── Basketball_Court_Diagram.jpg                # Basketball Court Diagram
│   ├── YOLOv10s_custom.pt                          # Custom Object Detection Model based on YOLOv10s
│   ├── detected_objects.csv                        # Features of the detected objects in the simulation
│   ├── PlayBook-AI Data Flow Diagram.png           # PlayBook-AI Data Flow Diagram
├── deep_sort/                                      # DeepSORT related files (Mutli-Object Tracking)
├── References/                                     # References for the development of the project
├── Custom_DetectionModel_Info.txt                  # Custom Detection Model information
├── Data_Loading.py                                 # Loading extracted object tracking information into database
├── dockerfile                                      # File to setup isolated environment to test code
├── Feature_Engineering.py                          # Optimizing the raw object tracking dataset for the neural network
├── Object_Tracking.py                              # Script for tracking objects in the simulation
├── Passing_Simulation.py                           # Script for simulating basketball plays
├── README.md                                       # Project documentation
├── Requirements.txt                                # Project library requirements
├── run_sequence.sh                                 # Sequence on how to execute files for isolated (Docker) environments
├── utils.py                                        # Commonly used functions to avoid duplication
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) License - see the LICENSE file for details.