# NBA Play Analysis

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-blue.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/caa2d542ea8e47b597b3712cbc4236cb)](https://app.codacy.com/gh/AbhishekR3/Basketball-PlayAnalysis/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
![Version Badge](https://img.shields.io/badge/version-0.3.0-orange)

## Project Description

This project aims to analyze and classify the type of offensive and defensive plays executed by NBA teams using simulations and object tracking techniques. It creates simulations similar to CourtVision Clippers to generate training data for a temporal-spatial based neural networks to classify specific plays.

CourtVision Frame

![CourtVision Sample Frame](https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/main/assets/Clippers%20CourtVision.png)

Pygame Simulation Frame

![Simulation Sample Frame](https://github.com/AbhishekR3/Basketball-PlayAnalysis/blob/main/assets/Basketball%20Simulation%20Image.png)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Project Description](#project-description)
- [Project Structure](#project-structure)
- [License](#license)
- [Acknowledgements](#acknowledgments)

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

The goal of this project is to create an algorithm that classifies offensive and defensive basketball plays. The project creates simulations similar to CourtVision Clippers and will eventually use these simulations to develop a classification model. This project aims to analyze the plays executed by the LA Clippers and their opponents.

The project uses a time-spatial temporal-based classification model. The simulations are created using pygame for generating training data. Once the model is accurate, it will be used to classify the data displayed in CourtVision on NBA.com.

The project outline can be found in the folder called "Basketball Play Classification Project Outline"

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
├── Basketball Play Classification Project Outline  # Project Outline
├── README.md                                       # Project documentation
├── assets/                                         # Directory containing images and diagrams
│   ├── Basketball Court Diagram.jpg
│   ├── Clippers CourtVision.png
│   ├── Basketball Simulation Image.png
```

## License

This project is licensed under the Creative Commons Attribution-NonCommercial (CC BY-NC) License - see the LICENSE file for details.

## Acknowledgments

- NBA LA Clippers for CourtVision data

I do not own the CourtVision data and will not be using this for commercial purposes. This is strictly for educational and learning purposes.