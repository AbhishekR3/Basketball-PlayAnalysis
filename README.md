# NBA Play Analysis

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/caa2d542ea8e47b597b3712cbc4236cb)](https://app.codacy.com/gh/AbhishekR3/Basketball-PlayAnalysis/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
![Version Badge](https://img.shields.io/badge/version-0.3.2-orange)
[![Combined Status](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/AbhishekR3/8cd877c3426a17132649c9c3d3a9e8b0/raw/badge.json)](https://github.com/AbhishekR3/Basketball-PlayAnalysis/actions)

## Project Description

This project aims to analyze and classify the type of offensive and defensive plays executed by NBA teams using simulations and object tracking techniques. It creates simulations similar to CourtVision Clippers to generate training data for a temporal-spatial based neural networks to classify specific plays.

CourtVision Sample Frame

![CourtVision Sample Frame](https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/main/assets/Clippers%20CourtVision.png)

Pygame Simulation Frame

![Simulation Sample Frame](https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/main/assets/Basketball%20Simulation%20Image.png)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [License](#license)

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

## Usage

First run Basketball_Game_Simulation.py to generate video simulation
Next run Basketball_Object_Tracking.py for object tracking

```bash
python Basketball_Game_Simulation.py
python Basketball_Object_Tracking.py
```

## Project-Description

The goal of this project is to create an algorithm that classifies offensive and defensive basketball plays. The project creates simulations (similar to Clippers CourtVision) and will eventually use these custom simulations to develop a classification model.

The project will use a temporal-spatial based classification model. The simulations are created using pygame for generating training data.

The project outline can be found in the file called "Basketball Play Classification Project Outline"

I am open to contributions. Please contact me at <abhishektips3@gmail.com> for contributions. 

Contributions are welcome in the following areas:

- Creating different offensive/defensive plays using pygame.

- Developing the neural network for the classification of the objects.

- Other ideas that can improve and expand the project.

## Project-Structure

```bash
Basketball-PlayAnalysis/
├── Basketball_Game_Simulation.py                   # Script for simulating basketball games
├── Basketball_Object_Tracking.py                   # Script for tracking objects in the simulation
├── update_badge_overall_test.py                    # Script to update CI/CD pipeline badge
├── README.md                                       # Project documentation
├── Requirements.txt                                # Project requirements
├── assets/                                         # Directory containing images and diagrams
│   ├── Basketball Court Diagram.jpg
│   ├── object_tracking_video.mp4
│   ├── simulation.mp4
├── .github/workflows                               # Git Actions
│   ├── game-simulation.yaml
│   ├── object-tracking.yaml
│   ├── overall-test.yaml
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) License - see the LICENSE file for details.