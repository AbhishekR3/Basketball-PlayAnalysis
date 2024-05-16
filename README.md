# NBA Play Analysis

This project aims to analyze and classify the type of offensive and defensive plays executed by NBA teams using simulations and object tracking techniques. It creates simulations similar to CourtVision Clippers to generate training data for a time-spatial temporal-based classification model.

![CourtVision Sample Frame](https://github.com/AbhishekR3/NBA-PlayAnalysis/blob/main/NBA%20CourtVision.png)

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation

### Prerequisites
- Python 3.x (latest version recommended)

### Dependencies
Ensure you have the following packages installed:

Install the following libraries.
```bash
pip install pygame opencv-python numpy
```

Clone the Repository
```bash
git clone https://github.com/AbhishekR3/NBA-PlayAnalysis.git
cd NBA-PlayAnalysis
```

Project Description

The goal of this project is to create an algorithm that classifies offensive and defensive basketball plays. The project creates simulations similar to CourtVision Clippers and will eventually use these simulations to develop a classification model. This project aims to analyze the plays executed by the LA Clippers and their opponents.

The project uses a time-spatial temporal-based classification model. The simulations are created using pygame for generating training data. Once the model is accurate, it will be used to classify the data displayed in CourtVision on NBA.com.

The project roadmap can be found in the folder called "Basketball Offensive Play Analysis Roadmap"

Current Implementations
Simulation File:
- Passing of the ball
- Dribbling of the ball
- Movement of players

Object Tracking File:
- Object Detection
- Creating a mask to remove background


```bash
NBA-PlayAnalysis/
├── Basketball_Game_Simulation.py   # Script for simulating basketball games
├── Basketball_Object_Tracking.py   # Script for tracking objects in the simulation
├── README.md                       # Project documentation
├── assets/                         # Directory containing images and diagrams
│   ├── NBA Court Diagram.jpg
│   ├── NBA CourtVision.png
│   ├── NBA Simulation Image.png
```

I do not own the CourtVision data and will not be using this for commercial purposes. This is strictly for educational and learning purposes.
I am open to contributions. Please contact me at abhishektips3@gmail.com for contributions. Contributions are welcome in the following areas:

- Creating different offensive/defensive plays using pygame.
- Developing the neural network for the classification of the objects.
- Other ideas that can improve and expand the project.



Acknowledgments
- NBA LA Clippers for CourtVision data

