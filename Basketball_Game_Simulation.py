# Basketball Game Simulation

# Utilizing pygame, simulating basketball plays executed.

#%%
"Import Libraries"

import pygame
import random
import math
import time
import datetime
import numpy as np
import os
import cv2
import secrets


#%%

class Player:
    "Class for players" 

    def __init__(self, x, y, radius, color):
        """
        Objective:
        Initialize Player class
        """

        self.x = x #set x-coordinate
        self.y = y #set y-coordinate
        self.initial_x = x #set first x-coordinate after received by player
        self.initial_y = y #set first y-coordinate after received by player
        self.radius = radius #set radius
        self.color = color #set player color
        self.speed = random.uniform(0.2, 1.7) #set random speed
        self.angle = random.uniform(0, 2 * math.pi) #set random angle
        self.next_angle = self.angle + np.radians(np.random.normal(90, 1.5)) #set next angle
        self.last_update_time = time.time() #set last update time
        self.angle_update = False #Set angle update for basketball displacement
        self.change_ball_time_limit = random.uniform(3, 6) #Set time limit when ball should update
    
    def update_speed_angle(self, basketball):
        """
        Objective:
        Updates the player's speed and angle to create a realistic flow of the player.
        The values will update after the change_ball_time_limit is passed
        
        Parameters:
        [Class] self - Player
        [Class] basketball - basketball
        """
        
        # Current time - last updated time < Change time
        if time.time() - self.last_update_time >= self.change_ball_time_limit:

            #After angle has updated, change the angle_update and change_displacement to false
            self.angle_update = False
            basketball.change_displacement = False

            #Change speed
            self.speed = random.uniform(0.2, 1.7)

            #Change angle
            self.angle = self.next_angle

            #Update last update time to current time
            self.last_update_time = time.time()
            
            #Update when ball update should change
            self.change_ball_time_limit = random.uniform(3,6)

            self.initial_x = self.x #set first x-coordinate after received by player
            self.initial_y = self.y #set first y-coordinate after received by player

            #For a fair split of which side of the player the basketball is being dribbled
            #Increase/decrease the self.angle depending on the time
            if int(time.time()%60)%2 == 0:
                self.next_angle += np.radians(np.random.normal(90, 1.5))
            else:
                self.next_angle -= np.radians(np.random.normal(90, 1.5)) #Update next angle change
    
    def move(self, basketball):
        """
        Objective:
        Move the player depending on it's speed and angle
        If the player reaches the edge of the screen, change angle
        
        Parameters:
        [Class] self - Player
        [Class] basketball - basketball
        """
        
        self.update_speed_angle(basketball)

        self.x += self.speed * math.cos(self.angle)
        self.y += self.speed * math.sin(self.angle)
        
        # Prevent wall collision
        if self.x + self.radius > SCREEN_WIDTH-15 or self.x - self.radius < 0:
            self.angle = math.pi - self.angle
        if self.y + self.radius > SCREEN_HEIGHT-15 or self.y - self.radius < 0:
            self.angle = -self.angle
    
    def draw(self):
        """
        Objective: Draw the player on the simulation screen
        """
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)
        


#%%

class Basketball:
    "Class for the Basketball"
    
    def __init__(self, x, y, radius, color):
        """
        Objective:
        Initialize basketball class
        """
        
        self.x = x #set x-coordinate
        self.y = y #set y-coordinate
        self.radius = radius #set basketball radius
        self.color = color #set basketball color
        self.speed = random.uniform(0.2, 1.7) #set random speed
        self.angle = random.uniform(0, 2 * math.pi) #set random angle
        self.last_update_time = time.time()  #set last update time
        self.history = [] # To store positions for fluctuation effect
        self.offset = 0 #Offset refers to the displacment of basketball from the player. Initialize at 0, since this will update later.
        self.change_displacement = False #To prevent constant change in displacment from the player
        self.displacement_randomness = random.choice([True, False]) #Displacement randomness to choose a side for a player
        self.dribble_switch = False #When dribble switch is false, do not update position through basketball class
        self.stabalize_dribble_switch_x = None #Set the x starting value of the next side of the dribble switch
        self.stabalize_dribble_switch_y = None #Set the y starting value of the next side of the dribble switch
        self.stabalize_dribble_switch_check = False #Set the check to maintain the same stabalized value
    
    
    def update_position(self, dribbling_player, elapsed_time):
        """
        Objective:
        Update the position of the basketball relative to the current player dribbling the ball.
        
        Parameters:
        [Class] self - Basketball
        [Class] dribbling_player - The player currently dribbling the basketball
        [Float] elapsed_time - The time taken from when the player first received the basketball or switched dribble side
        """
        # Oscillation parameters
        amplitude = 15  # Amplitude of the oscillation
        frequency = 8  # Frequency of the oscillation
        basketball_displacement = 15 # Displacement of the basketball from the player

        # Oscillation using a sine function to create a dynamic, naturalistic movement
        self.offset = amplitude * math.sin(frequency * elapsed_time)

        # Update basketball's position
        (
            self.x,
            self.y,
            self.change_displacement,
            dribbling_player.angle_update,
            self.displacement_randomness
        ) = update_basketball_position(
            self,
            dribbling_player,
            basketball_displacement
        )
        
    def draw(self):
        """
        Objective: Draw the basketball on the simulation screen
        """
        pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)


#%%

def update_basketball_position(ball, dribbling_player, basketball_displacement):
    """
    Objective:
    Displace basketball position relative to current_player to make a more natural dribbling motion
    The if condition checks, whether the basketball's relative position has been set.
    
    If relative position has not been set, calculate the relative position  
    Else, use sine function to calculate the movement of the basketball
    
    Parameters:
    [Class] self - Basketball
    [Class] dribbling_player - The player currently dribbling the basketball
    [int]   basketball_displacement - From the center of the basketball, the displacement of the basketball to simulate realistic dribbling

    Returns:
    [float]   x_coordinate - new basketball x-coordinate 
    [float]   y_coordinate - new basketball y-coordinate
    [bool] change_displacement_value - basketball change displacement checker
    [bool] angle_update_value - current player update angle checker
    [bool]   displacement_randomness - new basketball displacement value is True/False resulting in different side of which the basketball is being dribbled
    """
    
    #Ball should change location if and change displacement player hasnt changed direction
    if dribbling_player.angle_update is False and ball.change_displacement is False:
        
        x_coordinate, y_coordinate = new_relative_basketball_position(
            ball.x,
            ball.y,
            ball.displacement_randomness,
            ball.offset,
            ball.angle,
            dribbling_player,
            basketball_displacement
        )

        change_displacement_value = True
        angle_update_value = True
        displacement_randomness = random.choice([True, False])
    
    else:
        x_coordinate = dribbling_player.x + basketball_displacement + ball.offset * math.cos(ball.angle)
        y_coordinate = dribbling_player.y + basketball_displacement + ball.offset * math.sin(ball.angle)
        change_displacement_value = ball.change_displacement
        angle_update_value = dribbling_player.angle_update
        displacement_randomness = ball.displacement_randomness
        
    return x_coordinate, y_coordinate, change_displacement_value, angle_update_value, displacement_randomness


#%%

def new_relative_basketball_position(ball_x, ball_y, ball_displacement_randomness, ball_offset, ball_angle, dribbling_player, basketball_displacement):
    """
    Objective:
    Create a natural dribbling motion near the player
    Position the basketball on the left or right side of the player based on ball_displacement_randomness
    
    Parameters:
    [float]   ball_x - current basketball x-coordinate
    [float]   ball_y - current basketball y-coordinate
    [bool] ball_displacement_randomness - 0/1 to create a randomness of the player dribbling either side 
    [float] ball_offset - Oscillation offset from the ball
    [float] ball_angle - The angle the ball is moving
    [Class]   dribbling_player - The player currently dribbling the basketball
    [int] basketball_displacement - From the center of the basketball, the displacement of the basketball to simulate realistic dribbling

    Returns:
    [float] x_coordinate - new basketball x-coordinate 
    [float] y_coordinate - new basketball y-coordinate
    """
    
    
    # Determine the side the basketball should move relative to the current player's coordinates
    right_side = math.cos(current_player.x) > math.sin(current_player.y)
    left_side = math.cos(current_player.x) < -(math.sin(current_player.y))

    # Calculate new position based on the conditions provided
    if right_side and left_side:
        if ball_displacement_randomness is True:
            x_coordinate = current_player.x + basketball_displacement + ball_offset * math.cos(ball_angle)
            y_coordinate = ball_y
        else:
            x_coordinate = current_player.x - basketball_displacement + ball_offset * math.cos(ball_angle)
            y_coordinate = ball_y

    elif right_side and not left_side:
        if ball_displacement_randomness is True:
            x_coordinate = ball_x
            y_coordinate = current_player.y + basketball_displacement + ball_offset * math.sin(ball_angle)
        else:
            x_coordinate = ball_x
            y_coordinate = current_player.y - basketball_displacement + ball_offset * math.sin(ball_angle)
    elif not right_side and left_side:
        if ball_displacement_randomness is True:
            x_coordinate = ball_x
            y_coordinate = current_player.y + basketball_displacement + ball_offset * math.sin(ball_angle)
        else:
            x_coordinate = ball_x
            y_coordinate = current_player.y - basketball_displacement + ball_offset * math.sin(ball_angle)
    else:
        if ball_displacement_randomness is True:
            x_coordinate = current_player.x + basketball_displacement + ball_offset * math.cos(ball_angle)
            y_coordinate = ball_y
        else:
            x_coordinate = current_player.x - basketball_displacement + ball_offset * math.cos(ball_angle)
            y_coordinate = ball_y

    return x_coordinate, y_coordinate


#%%

def move_basketball_to_location(ball, target_x, target_y, avoiding_players=None, exclusion_list=None):
    """
    Objective:
    Move the basketball to a specific location (either to a player or a specific x,y position)
    
    Parameters:
    [Class] ball - basketball
    [float] target_x - x coordinate of the end location
    [float] target_y - y coordinate of the end location
    [List]  avoiding_players - List of all the players except the 2 players being passed between
    [List]  exclusion_list - Exclusion list includes the current player and the player being passed to
    
    Returns:
    [float] ball.x - new x-coordinate of the basketball 
    [float] ball.y - new y-coordinate of the basketball 
    """
    
    # avoiding_players is set to None when the basketball is being dribbled to a new position by the same player
    if avoiding_players is None:
        # Calculate the direct path vector
        dx, dy = target_x - ball.x, target_y - ball.y

        #If movement is less than 7 pixels, stabalize the movement so we need to return the same ball.x, ball.y value.
        #Set diplsacement to 0
        if dx <= 7 or dy <= 7:
            displacement_x, displacement_y = 0,0
        
        else:
            path_vector = pygame.math.Vector2(dx, dy)
            path_vector = path_vector.normalize() * MOVE_SPEED
            displacement_x, displacement_y = path_vector.x, path_vector.y
            
    # If ball is being passed to another player, avoiding_players is set and the basketball avoids these players
    else:
        path_vector = adjust_path_if_needed(ball, target_x, target_y, avoiding_players, exclusion_list)
        displacement_x = path_vector.x
        displacement_y = path_vector.y
        
    # Add the displacement in the x, y coordinate to the basketball position
    ball.x += displacement_x
    ball.y += displacement_y
    
    return ball.x, ball.y


#%%

def ball_reached_player(ball, target, speed):
    """
    Objective:
    Return a boolean value whether or not the basketball reached the player

    Parameters:
    [Class] ball - basketball
    [Class] target - Player receiving the basketball (current_player)
    [float] speed - speed the basketball is moving to another player

    Returns:
    [bool] if the ball reached the player or not
    """

    # Stop loop if distance between player and basketball is less than the ball speed
    if pygame.math.Vector2(target.x - ball.x, target.y - ball.y).length() <= speed:
        if ((abs(ball.y - target.y) <= 5) and (abs(ball.x - target.x) <= 5)):
            return True
    else:
        return False


def ball_reached_position(ball, target_x, target_y):
    """
    Objective:
    Return a boolean value whether the basketball has reached within the distance of the specific position

    Parameters:
    [Class] ball - basketball
    [float] target_x - x coordinate of the end location
    [float] target_y - y coordinate of the end location

    Returns:
    [bool] if the ball reached the end location or not
    """

    # Stop loop if distance between player and basketball is less than 5 pixels
    ball_coordinate = (ball.x, ball.y)
    target_coordinate = (target_x, target_y)
    
    if math.dist(ball_coordinate, target_coordinate) < 5:
        return True
    else:
        return False


#%%

def adjust_path_if_needed(ball, target_x, target_y, opposite_players=None, exclusion_list=None):
    """
    Objective:
    If there is a player between the current_player and the player the ball is being passed to,
    adjust the path of the basketball
    
    Parameters:
    [Class] ball - basketball
    [float] target_x - x coordinate of the end location
    [float] target_y - y coordinate of the end location
    [List]  avoiding_players - List of all the players except the 2 players being passed between.
    [List]  exclusion_list - Exclusion list includes the current player and the player being passed to
    
    Returns:
    [Vector2] vector at which the player should move
    """
    
    # Calculate the direct path vector
    dx, dy = target_x - ball.x, target_y - ball.y
    path_vector = pygame.math.Vector2(dx, dy)

    # Check if each player is in the way
    for opposite_player in opposite_players:
        if opposite_player not in exclusion_list:
            # Calculate vector from ball to player
            to_player_vector = pygame.math.Vector2(opposite_player.x - ball.x, opposite_player.y - ball.y)

            # Check if the player is close to the direct path
            distance_to_path = path_vector.cross(to_player_vector) / path_vector.length()

            if abs(distance_to_path) < player.radius + ball.radius:
                # Adjust the path to avoid the player
                avoidance_vector = path_vector.normalize().rotate(15)
                return avoidance_vector * MOVE_SPEED
        
    # If no adjustment needed, return the original path vector
    return path_vector.normalize() * MOVE_SPEED

#%%

def is_valid_placement(new_player, existing_players):
    """
    Objective:
    Check if the placement of the new potential player is valid based on the position of the existing players
    
    Parameters:
    [Class] new_player - Proposed new player position
    [List]  existing_players - Exisitng players that have been finalized
    
    Returns:
    [bool] If proposed player position is 2*radii away from other players, then True else False
    """
    
    for ith_player in existing_players:
        
        # Calculate distance between player and existing players
        dx = new_player.x - ith_player.x
        dy = new_player.y - ith_player.y
        distance = math.sqrt(dx**2 + dy**2)

        # Set minimum allowed distance as 2*radii away from exisitng players
        min_allowed_distance = ith_player.radius*2
        
        if distance < min_allowed_distance:
            return False
    return True

def place_circle_with_constraints(existing_players, radius, color, simulation_width, simulation_height):
    """
    Objective:
    Check if the player can be placed in the specified location
    Stop trying to create another player if the loop crosses 1000 attempts
    
    Parameters:
    [list]  existing_players - A list of players already created
    [int]   radius - Player radius
    [tuple] Player color in RGB
    [int]   simulation_width - Simulation screen width
    [int]   simulation_height - Simulation screen height
    
    Returns:
    [Class] new_player - A player which is in a valid placement
    """
        
    attempts = 0
    while attempts < 1000:  # Limit attempts to prevent infinite loop
        new_player = Player(random.randint(radius, simulation_width - (2*radius)),
                            random.randint(radius, simulation_height - (2*radius)),
                            radius,
                            color)
        if is_valid_placement(new_player, existing_players):
            return new_player
        attempts += 1
        
    raise Exception("Failed to place a new player without exceeding overlap threshold.")


#%%

def circles_overlap(circle1, circle2, minimum_overlap_percentage):
    """
    Objective:
    Check if 2 circles (players or basketball) overlap 
    
    Parameters:
    [Class] Circle1 - The first circle to check
    [Class] Circle2 - The second circle to check
    [Float] minimum_overlap_percentage - How much of the 2 circles should overlap
    
    Returns:
    [bool] If the ball does overlap return True else False 
    """
        
    # Calculate the distance between the centers of the two circles
    distance_centers = math.sqrt((circle1.x - circle2.x) ** 2 + (circle1.y - circle2.y) ** 2)
    
    # Calculate the sum of the radii
    sum_of_radii = circle1.radius + circle2.radius
    
    # Overlap Percentage
    adjusted_distance_for_overlap = sum_of_radii * (1 - minimum_overlap_percentage)

    return distance_centers <= adjusted_distance_for_overlap


#%%

def initialize_simulation():
    """
    Objective:
    
    Initialize the simulation, basketball, and players
    """
        
    global screen, players, team_players, current_player, basketball
    
    # Initialize Pygame
    pygame.init()

    # Create screen
    screen = pygame.display.set_mode(SCREEN_DIMENSIONS)
    pygame.display.set_caption("Basketball Game Simulation")

    # Initialize 10 players on the court (5 on each side)
    players = []
    for i in range(NUM_PLAYERS):
        color = COLOR_BLUE if i < NUM_PLAYERS // 2 else COLOR_RED
        player = place_circle_with_constraints(players, PLAYER_RADIUS, color, SCREEN_WIDTH, SCREEN_HEIGHT)
        players.append(player)

    # Choose a random blue player to place the basketball with
    team_players = [player for player in players if player.color == COLOR_BLUE]
    current_player = random.choice(team_players)

    basketball = Basketball(
        current_player.x - basketball_relative_x,
        current_player.y - basketball_relative_y,
        BALL_RADIUS, COLOR_ORANGE
    )


#%% Set Simulation Parameters

# Import Basketball Court image

#Screen Dimensions

SCREEN_DIMENSIONS = (500, 500)
SCREEN_WIDTH = SCREEN_DIMENSIONS[0]
SCREEN_HEIGHT = SCREEN_DIMENSIONS[1]

#Load and scale the image of the basketball court

# Load and transform Basketball court diagram
script_directory = os.getcwd()
image_location = os.path.join(script_directory, 'assets/Basketball Court Diagram.jpg')

try:
    background_image = pygame.image.load(image_location)
    background_image = pygame.transform.scale(background_image, SCREEN_DIMENSIONS)
except Exception as e:
    print(f"An error occurred: {e}")

    
# Simulation Constants
NUM_PLAYERS = 10
PLAYER_RADIUS = 20
BALL_RADIUS = 10
COLOR_BLUE = (0, 0, 255)
COLOR_RED = (255, 0, 0)
COLOR_ORANGE = (255, 165, 0)
COLOR_WHITE = (255, 255, 255)
FPS = 30
MOVE_SPEED = np.random.normal(5.6, 1) #Speed at which basketball moves to the next player

# Simulation Variables
clock = pygame.time.Clock()
pass_timer = -1
pass_interval = random.uniform(4, 5)  #How many seconds before the player passes the ball
reached_player = True #When basketball is with a player
basketball_relative_x = 5 #X-axis displacemnt of the basketball compared to the player
basketball_relative_y = 5 #Y-axis displacemnt of the basketball compared to the player
basketball_player_overlap = 0.4 #When basketball and player overlap, set the relative distance between the two to prevent constant change
first_overlap = False #Calculate the basketball's relative x, y coordinate to the circle
last_update_time = datetime.datetime.now() #set update time
dribble_timer = time.time()
dribble_switch_timer = time.time() - dribble_timer
oscillation_start_time = time.time()
basketball_displacement = 15 #Basketball displacement from the player
simulating = True #Set simulation to true to start
simulation_limit = 3 # stop simulation after x minutes


#%%

" Simulate Basketball Game "

initialize_simulation()

start_time_simulation = time.time()

# Define the codec and create VideoWriter object
video_format = cv2.VideoWriter_fourcc(*'X264') # Using x264
video_location = os.path.join(script_directory, 'assets/simulation.mp4')
out = cv2.VideoWriter(video_location, video_format, FPS, SCREEN_DIMENSIONS)

frames_captured = 0
simulation_capture_max_time = 45 # x seconds the simulation will run to capture recordings
max_frames_caputured = FPS * simulation_capture_max_time

while simulating and (frames_captured < max_frames_caputured):
    
    elapsed_time_simulation = time.time() - start_time_simulation # Set elapsed time to stop simulation after mentioned time
    
    #Create screen with basketball court as the background
    screen.blit(background_image, (0,0))

    #Stop simulation
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            simulating = False

    # Update and draw players
    for player in players:
        player.move(basketball)
        player.draw()
        
    time_elapsed = time.time() - oscillation_start_time # Time elapsed since ball reached the player

    # Move basketball towards the current player
    if reached_player is False:
        
        #Restart the following variables
        current_player.angle_update = False
        basketball.change_displacement = False
        basketball.stabalize_dribble_switch_check = False
        
        # Create a new list excluding the current player and randomly choose a player to pass to
        team_players_excluding_current = [player for player in team_players if player != current_player]
        new_random_player = random.choice(team_players_excluding_current)
        
        pass_players = [current_player, new_random_player] #Both players the ball is being passed between
        basketball.x, basketball.y = move_basketball_to_location(
            basketball,
            current_player.x,
            current_player.y,
            players,
            pass_players
        )  # Move basketball closer to player
        reached_player = ball_reached_player(basketball, current_player, MOVE_SPEED) #Stop if ball reached player
        
        #Calculate the distance between the basketball and the receiving player.
        #This dictates which relative position the basketball should be from the player
        if first_overlap is False:
            if circles_overlap(basketball, current_player, basketball_player_overlap):
                
                #Calculate x, y displacement relative to player
                basketball_relative_x = basketball.x - current_player.x
                basketball_relative_y = basketball.y - current_player.y
                
                last_update_time = datetime.datetime.now() #set update time
                first_overlap = True
        
        oscillation_start_time = time.time() #Set the oscillation start time to current time to reset the timer
    
    #current_player dribbles the basketball
    else:
        MOVE_SPEED = np.random.normal(5.6, 1) #Update move_speed
        player_pass_time = player.last_update_time + player.change_ball_time_limit

        # Check if basketball should switch to another side of the player
        if basketball.dribble_switch is False:
            basketball.update_position(current_player, time_elapsed)
            basketball.speed = current_player.speed
            basketball.angle = current_player.angle
            basketball.stabalize_dribble_switch_check = False #Restart the stabalizing dribble switch check

        # Start timer once basketball reaches player
        if pass_timer < 0:
            pass_timer = 0  # Activate timer
            oscillation_start_time = time.time() #Set the oscillation start time to current time to reset the timer
        pass_timer += clock.get_time() / 1000.0  # Convert milliseconds to seconds

        # If pass_time greater or equal to when the ball should be passed
        if pass_timer >= player.change_ball_time_limit:
            # Time to pass the basketball to the next blue player
            current_index = team_players.index(current_player) + 1
            if current_index >= len(team_players):
                current_index = 0
            current_player = team_players[current_index] #set a new player to control the basketball
            pass_timer = -1  # Reset timer to deactivate
            first_overlap = False
            reached_player = False

    basketball.draw()  # Draw the basketball

    pygame.display.flip() #Update pygame simulation frame
    clock.tick(FPS) # Maintain frame rate (FPS)


    # Stop simulation after simulation limit time as been met
    if elapsed_time_simulation > simulation_limit*60:
        print("Simulation reached time limit of", simulation_limit, "minutes. Stopping simulation")
        break


    # Capture frame
    frame = pygame.surfarray.array3d(pygame.display.get_surface())
    frame = frame.transpose([1, 0, 2])  # transpose to the correct shape
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert from RGB to BGR

    # Press 'q' to quit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Write the frame
    out.write(frame)
    frames_captured += 1

out.release()
pygame.quit()