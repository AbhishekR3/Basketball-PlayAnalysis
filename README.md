# NBA Play Analysis

This project aims to analyze and classify the type of offensive and defensive plays executed by NBA teams using simulations and object tracking techniques. It creates simulations similar to CourtVision Clippers to generate training data for a time-spatial temporal-based classification model.

<p align="center">
  <img src="https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/main/assets/NBA%20CourtVision.png" alt="CourtVision Sample Frame" width="400" height="400">
  <img src="https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/main/assets/NBA%20Simulation%20Image.png" alt="Simulation Sample Frame" width="400" height="400">
</p>
## Table of Contents
- [Installation](#installation)
- [Project Description](#project-description)
- [Project Structure](#project-structure)
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

## Project-Description

The goal of this project is to create an algorithm that classifies offensive and defensive basketball plays. The project creates simulations similar to CourtVision Clippers and will eventually use these simulations to develop a classification model. This project aims to analyze the plays executed by the LA Clippers and their opponents.

The project uses a time-spatial temporal-based classification model. The simulations are created using pygame for generating training data. Once the model is accurate, it will be used to classify the data displayed in CourtVision on NBA.com.

The project roadmap outline can be found in the folder called "Basketball Offensive Play Analysis Roadmap"

I am open to contributions. Please contact me at abhishektips3@gmail.com for contributions. 

Contributions are welcome in the following areas:
- Creating different offensive/defensive plays using pygame.
- Developing the neural network for the classification of the objects.
- Other ideas that can improve and expand the project.


## Project-Structure
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


## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments
- NBA LA Clippers for CourtVision data

I do not own the CourtVision data and will not be using this for commercial purposes. This is strictly for educational and learning purposes.