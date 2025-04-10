# PlayBook AI: Basketball Intelligence

[![CI Testing](https://github.com/AbhishekR3/Basketball-PlayAnalysis/actions/workflows/ci.yml/badge.svg)](https://github.com/AbhishekR3/Basketball-PlayAnalysis/actions/workflows/ci.yml)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/caa2d542ea8e47b597b3712cbc4236cb)](https://app.codacy.com/gh/AbhishekR3/Basketball-PlayAnalysis/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Release Badge](https://img.shields.io/github/v/release/AbhishekR3/Basketball-PlayAnalysis.svg?color=orange)](https://github.com/AbhishekR3/Basketball-PlayAnalysis/releases)
![AWS](https://img.shields.io/badge/Amazon_AWS-232F3E?style=flat&logo=amazon-web-services&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)



## Table of Contents
- [Product Description](#product-description)
- [Installation](#installation)
- [Prerequisites](#prerequisites)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Product Structure](#product-structure)
- [License](#license)


## Product-Description

Product Goals:

PlayBook AI will help basketball professionals:
* Analyze offensive and defensive plays executed by teams
* Assess potential player fit within a team
* Understand play patterns in critical game moments
* Identify a team's most/least successful plays

Product Overview:
1. Simulate basketball plays for training data
2. Perform multi-object tracking with custom detection model
3. Optimize dataset using feature engineering techniques 
4. Leverage spatial databases with ACID and Spatial Index for efficient data management
5. Create a classification Neural Network based on LSTM
6. Deploy software on AWS with GPU accelerated libraries, CI testing pipelines, and Docker

Object Tracking on a simulation of a player dribbling

![ObjectTracking_Demo](https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/DEV_Code/assets/ObjectTracking%20Demo.gif)

Product Data Flow Diagram

![DataFlowDiagram](assets/PlayBook-AI%20Data%20Flow%20Diagram.png)

LSTM Model Architecture

![LSTMModelArchitecture](assets/LSTM%20Architecture.png)

## Installation

### Prerequisites

- Python 3.11

### Dependencies

Ensure you have the following installed:

1. Refer Requirements.txt file and install the libraries mentioned

2. Clone the Repository
```bash
git clone https://github.com/AbhishekR3/Basketball-PlayAnalysis.git
cd Basketball-PlayAnalysis
```

## Usage

These files are built on Python 3.11-slim.
Run files in this order
1. Passing_Simulation.py / RandomMovement_Simulation.py to generate video simulations of passes or random player/ball movement (not a pass)
2. Object_Tracking.py for object tracking for the relevant video simulation
3. Feature_Engineering.py for extracting relevant features and optimizing the dataset
4. Neural_Network.py to create custom LSTM model

```bash
python Passing_Simulation.py
python Object_Tracking.py
python Feature_Engineering.py
python Neural_Network.py
```

## Product-Structure
Relevant files

```bash
Basketball-PlayAnalysis/
├── assets/                                         # Containing referenced images and diagrams
│   ├── Basketball_Court_Diagram.jpg                # Basketball Court Diagram
│   ├── YOLOv10s_custom.pt                          # Custom Object Detection Model based on YOLOv10s
│   ├── PlayBook-AI Data Flow Diagram.png           # PlayBook-AI Data Flow Diagram
│   ├── LSTM Architecture.png                       # Neural Network LSTM Architecture layout
│   ├── basketball_lstm_model.pt                    # Best performing LSTM based model
│   ├── pruned_model.pt                             # Pruned model based on best performing LSTM model
├── deep_sort/                                      # DeepSORT related files (Mutli-Object Tracking)
├── References/                                     # References for the development of the product
│   ├── Custom_DetectionModel.txt                   # Info / Metrics on custom object detection model
│   ├── Citations                                   # Citations
│   ├── pruning_comparison.png                      # Pruned model compared to oringal model
├── Data_Loading.py                                 # Loading extracted object tracking information into database
├── dockerfile                                      # File to setup isolated environment to test code
├── Feature_Engineering.py                          # Optimizing the raw object tracking dataset for the neural network
├── Neural_Network.py                               # Script for performing LSTM model training
├── Object_Tracking.py                              # Script for tracking objects in the simulation
├── Passing_Simulation.py                           # Script for simulating passing plays
├── RandomMovement_Simulation.py                    # Script for simulating random object movement plays
├── README.md                                       # Product documentation
├── requirements.txt                                # Product library requirements
├── run_sequence.sh                                 # Sequence on how to execute files for isolated (Docker) environments
├── utils.py                                        # Commonly used functions to avoid duplication
```

## License

This product is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) License - see the LICENSE file for details.