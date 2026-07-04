'''
Random Movement Simulation
Utilizing pygame, simulating random basketball and player movements without structured passing
'''

#%%

# Import Libraries
import pygame
import math
import time
import numpy as np
import os
import cv2
# Seeded RNG for reproducible simulations (replaces the stdlib secret-random
# draws). Seed via the SIM_SEED env var (default 42); call seed_rng(...) to
# reproduce a run exactly.
_rng = np.random.default_rng(int(os.environ.get('SIM_SEED', 42)))


def seed_rng(seed):
    "Reseed the module RNG so a simulation run can be reproduced bit-for-bit."
    global _rng
    _rng = np.random.default_rng(seed)


def _rng_choice(seq):
    "Deterministic replacement for a random choice over a sequence."
    seq = list(seq)
    return seq[int(_rng.integers(len(seq)))]


def _rng_below(n):
    "Deterministic replacement for a random integer in [0, n)."
    return int(_rng.integers(n))
from utils import configure_logger

#%%

class Player:
    "Class for players"

    def __init__(self, x, y, radius, color):
        """
        Objective:
        Initialize Player class
        
        Parameters:
        [float] x - x-coordinate
        [float] y - y-coordinate
        [int] radius - player radius
        [tuple] color - player color in RGB
        """

        self.x = x #set x-coordinate
        self.y = y #set y-coordinate
        self.radius = radius #set radius
        self.color = color #set player color
        self.speed = _rng.uniform(0.2, 2.5) #set random speed (slightly higher than original)
        self.angle = _rng.uniform(0, 2 * math.pi) #set random angle
        self.next_angle = self.angle + cryptographic_normal(90, 1.5, True) #set next angle
        self.last_update_time = time.time() #set last update time
        self.change_direction_time_limit = _rng.uniform(2, 5) #Set time limit for direction change

    def update_speed_angle(self):
        """
        Objective:
        Updates the player's speed and angle to create a realistic flow of the player.
        The values will update after the change_direction_time_limit is passed
        
        Parameters:
        [Class] self - Player
        """
        try:
            # Current time - last updated time < Change time
            if time.time() - self.last_update_time >= self.change_direction_time_limit:
                # Change speed
                self.speed = _rng.uniform(0.2, 2.5)

                # Change angle
                self.angle = self.next_angle

                # Update last update time to current time
                self.last_update_time = time.time()

                # Update when direction should change
                self.change_direction_time_limit = _rng.uniform(2, 5)

                # Update the angle randomly
                if int(time.time()%60)%2 == 0:
                    self.next_angle += cryptographic_normal(90, 1.5, True)
                else:
                    self.next_angle -= cryptographic_normal(90, 1.5, True) #Update next angle change
        except Exception as e:
            logger.error("Error in updating player speed and angle: %s", e)
            raise

    def move(self):
        """
        Objective:
        Move the player depending on it's speed and angle
        If the player reaches the edge of the screen, change angle

        Parameters:
        [Class] self - Player
        """
        try:
            self.update_speed_angle()

            vx = self.speed * math.cos(self.angle)
            vy = self.speed * math.sin(self.angle)
            self.x += vx
            self.y += vy

            # Valid interior bounds (single margin constant)
            min_x, max_x = self.radius, SCREEN_WIDTH - WALL_MARGIN - self.radius
            min_y, max_y = self.radius, SCREEN_HEIGHT - WALL_MARGIN - self.radius

            # #13 fix: reflect only when heading further out of bounds, so a
            # wall-adjacent player can't oscillate and stick, then clamp back
            # inside so the check can't re-trigger next frame.
            if (self.x > max_x and vx > 0) or (self.x < min_x and vx < 0):
                self.angle = math.pi - self.angle
            if (self.y > max_y and vy > 0) or (self.y < min_y and vy < 0):
                self.angle = -self.angle

            self.x = min(max(self.x, min_x), max_x)
            self.y = min(max(self.y, min_y), max_y)

        except Exception as e:
            logger.error("Error in moving player: %s", e)
            raise

    def draw(self):
        """
        Objective: 
        Draw the player on the simulation screen
        
        Parameters:
        [Class] self - Player
        """
        try:
            # Circle
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius+2)

            inverse_color = reverse_color(self.color) # Calculate the inverse of circle's color  
            
            # Perimeter
            draw_circle(screen, (int(self.x), int(self.y)), self.radius + 2, inverse_color, 2)

            # Center
            draw_circle(screen, (int(self.x), int(self.y)), 2, inverse_color, 2)

        except Exception as e:
            logger.error("Error in drawing player: %s", e)
            raise

#%%

class Basketball:
    "Class for the Basketball"

    def __init__(self, x, y, radius, color):
        """
        Objective:
        Initialize basketball class
        
        Parameters:
        [float] x - x-coordinate
        [float] y - y-coordinate
        [int] radius - basketball radius
        [tuple] color - basketball color in RGB
        """

        self.x = x #set x-coordinate
        self.y = y #set y-coordinate
        self.radius = radius #set basketball radius
        self.color = color #set basketball color
        self.speed = _rng.uniform(0.5, 3.0) #set random speed (faster than players)
        self.angle = _rng.uniform(0, 2 * math.pi) #set random angle
        self.last_update_time = time.time() #set last update time
        self.change_direction_time_limit = _rng.uniform(1, 4) #shorter time for more changes

    def update_movement(self):
        """
        Objective:
        Updates the basketball's speed and angle for random movement
        
        Parameters:
        [Class] self - Basketball
        """
        try:
            if time.time() - self.last_update_time >= self.change_direction_time_limit:
                # Change speed
                self.speed = _rng.uniform(0.5, 3.0)
                
                # Change angle more randomly than players
                self.angle = _rng.uniform(0, 2 * math.pi)
                
                # Update last update time
                self.last_update_time = time.time()
                
                # Update when direction should change
                self.change_direction_time_limit = _rng.uniform(1, 4)
        except Exception as e:
            logger.error("Error in updating basketball movement: %s", e)
            raise

    def move(self):
        """
        Objective:
        Move the basketball independently based on its speed and angle
        
        Parameters:
        [Class] self - Basketball
        """
        try:
            self.update_movement()

            vx = self.speed * math.cos(self.angle)
            vy = self.speed * math.sin(self.angle)
            self.x += vx
            self.y += vy

            # Valid interior bounds (single margin constant)
            min_x, max_x = self.radius, SCREEN_WIDTH - WALL_MARGIN - self.radius
            min_y, max_y = self.radius, SCREEN_HEIGHT - WALL_MARGIN - self.radius

            # #13 fix: reflect only when heading further out, then clamp inside.
            if (self.x > max_x and vx > 0) or (self.x < min_x and vx < 0):
                self.angle = math.pi - self.angle
            if (self.y > max_y and vy > 0) or (self.y < min_y and vy < 0):
                self.angle = -self.angle

            self.x = min(max(self.x, min_x), max_x)
            self.y = min(max(self.y, min_y), max_y)

        except Exception as e:
            logger.error("Error in moving basketball: %s", e)
            raise

    def draw(self):
        """
        Objective: 
        Draw the basketball on the simulation screen
        
        Parameters:
        [Class] self - Basketball
        """
        try:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), self.radius)

            inverse_color = reverse_color(self.color) # Calculate the inverse of circle's color  
            
            # Perimeter
            draw_circle(screen, (int(self.x), int(self.y)), self.radius + 2, inverse_color, 2)

            # Center
            draw_circle(screen, (int(self.x), int(self.y)), 2, inverse_color, 2)

        except Exception as e:
            logger.error("Error in drawing basketball: %s", e)
            raise

#%%

def cryptographic_normal(mu, sigma, radian=False):
    """
    Objective:
    Return a the normal radian value using secrets library

    Parameters:
    [float] mu - mean
    [float] sigma - standard deviation
    [bool] radian - whether to convert to radians

    Returns:
    [float] normal value - normally distributed random value
    """
    try:
        # Generate two uniform random integers
        raw1 = int(_rng.integers(0, 2**64, dtype=np.uint64))
        raw2 = int(_rng.integers(0, 2**64, dtype=np.uint64))

        # Convert to floats in the range [0, 1)
        u1 = raw1 / 2**64
        u2 = raw2 / 2**64

        # Box-Muller transform
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)

        # Return a normally distributed number with mean mu and standard deviation sigma
        normal_value = mu + z0 * sigma

        if radian is False:
            return normal_value

        elif radian is True:
            return np.radians(normal_value)
    
    except Exception as e:
        logger.error("Error in calculating normal radians using secrets")
        logger.debug("Exception details: %s", str(e))
        raise

#%%

def reverse_color(circle_color):
    """
    Objective:
    Reverse the circle's RGB scheme to create a more accuracte object detection/tracking
    For example: [255, 0, 0] -> [0, 255, 255]

    Parameters:
    [tuple] circle_color - Detected color of the circle

    Returns:
    [tuple] inverse_circle_color - Inverted color for better detection
    """

    try:
        inverse_circle_color = tuple(0 if c >= 128 else 255 for c in circle_color)

        return inverse_circle_color

    except Exception as e:
        logger.error("Error in calculating the inverse of the given color: %s", e)
        raise

#%% 

def draw_circle(surface, center, radius, color, thickness=0):
    """
    Objective:
    Draw a circle on a pygame surface, similar to cv2.circle
    
    Parameters:
    [pygame.Surface] surface - Simulation frame to draw on
    [tuple] center - (x, y) of circle center
    [int] radius - radius of the circle
    [tuple] color - (r, g, b) of circle color
    [int] thickness - thickness of circle outline
    """

    x, y = center

    if thickness == 0:
        # Draw a filled circle
        pygame.draw.circle(surface, color, (int(x), int(y)), radius)
    else:
        # Draw a hollow circle
        for i in range(thickness):
            pygame.draw.circle(surface, color, (int(x), int(y)), radius - i, 1)

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
    
    try: 
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
    
    except Exception as e:
        logger.error("Error in finding valid placement for a player: %s", e)
        raise

def place_circle_with_constraints(existing_players, radius, color, simulation_width, simulation_height):
    """
    Objective:
    Check if the player can be placed in the specified location
    Stop trying to create another player if the loop crosses 1000 attempts
    
    Parameters:
    [list]  existing_players - A list of players already created
    [int]   radius - Player radius
    [tuple] color - Player color in RGB
    [int]   simulation_width - Simulation screen width
    [int]   simulation_height - Simulation screen height
    
    Returns:
    [Class] new_player - A player which is in a valid placement
    """

    try:
        attempts = 0
        while attempts < 1000:  # Limit attempts to prevent infinite loop
            new_player = Player(
                radius + _rng_below(simulation_width - (3 * radius) + 1),
                radius + _rng_below(simulation_height - (3 * radius) + 1),
                radius,
                color)
            
            if is_valid_placement(new_player, existing_players):
                return new_player
            attempts += 1

    except Exception as e:
        logger.error("Error in placing the players without exceeding overlap threshold: %s", e)
        raise

#%% Configure Docker containerization

try:
    log_dir = os.environ.get('LOG_DIR', '/app/logs')
    video_dir = os.environ.get('VIDEO_DIR', '/app/simulations')
    assets_dir = os.environ.get('ASSETS_DIR', '/app/assets')

    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    print('Log Directory:', log_dir)
    print('Video Directory:', video_dir)
    print('Assets Directory:', assets_dir)

    # Use it before writing files
    ensure_dir(video_dir)
    ensure_dir(log_dir)
    ensure_dir(assets_dir)
except Exception as e:
    # Fallback for local development
    script_directory = os.getcwd()
    log_dir = os.path.join(script_directory, 'logs')
    video_dir = os.path.join(script_directory, 'simulations')
    assets_dir = os.path.join(script_directory, 'assets')
    
    def ensure_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    ensure_dir(log_dir)
    ensure_dir(video_dir)
    ensure_dir(assets_dir)
    
    print(f"Error in creating environment for containers: {e}")
    print('Using local directories:')
    print('Log Directory:', log_dir)
    print('Video Directory:', video_dir)
    print('Assets Directory:', assets_dir)

#%% Set Simulation Parameters

# Configuring logging
logger = configure_logger('random_movement')
logger.info("Random Movement Simulation started")

# Screen Dimensions
SCREEN_DIMENSIONS = (470, 500)
SCREEN_WIDTH = SCREEN_DIMENSIONS[0]
SCREEN_HEIGHT = SCREEN_DIMENSIONS[1]

# Load and transform Basketball court diagram
try:
    image_location = os.path.join(assets_dir, 'Basketball_Court_Diagram.jpg')
    background_image = pygame.image.load(image_location)
    background_image = pygame.transform.scale(background_image, SCREEN_DIMENSIONS)
except Exception as e:
    logger.error(f"An error occurred when loading basketball court diagram. {e}")
    # Try a fallback path for local development
    try:
        script_directory = os.getcwd()
        image_location = os.path.join(script_directory, 'assets/Basketball_Court_Diagram.jpg')
        background_image = pygame.image.load(image_location)
        background_image = pygame.transform.scale(background_image, SCREEN_DIMENSIONS)
    except Exception as e2:
        logger.error(f"Failed to load background image from fallback path. {e2}")
        # Create a blank background as last resort
        background_image = pygame.Surface(SCREEN_DIMENSIONS)
        background_image.fill((200, 200, 200))  # Light gray background

# Simulation Constants
NUM_PLAYERS = 10
BALL_RADIUS = 10
PLAYER_RADIUS = BALL_RADIUS*2
WALL_MARGIN = 15 #Single boundary margin used by the wall-collision handler
COLOR_BLUE = (0, 0, 255)    # Team A
COLOR_RED = (255, 0, 0)     # Team B
COLOR_ORANGE = (255, 165, 0)  # Basketball
COLOR_WHITE = (255, 255, 255)
FPS = 30

#%%

def initialize_simulation():
    """
    Objective:
    Initialize the simulation, basketball, and players with random positions
    """
    try: 
        global screen, players, team_a_players, team_b_players, basketball
        
        # Initialize Pygame
        pygame.init()

        # Create screen
        screen = pygame.display.set_mode(SCREEN_DIMENSIONS)
        pygame.display.set_caption("Random Movement Simulation")

        # Initialize 10 players on the court (5 on each side)
        players = []
        for i in range(NUM_PLAYERS):
            color = COLOR_BLUE if i < NUM_PLAYERS // 2 else COLOR_RED
            player = place_circle_with_constraints(players, PLAYER_RADIUS, color, SCREEN_WIDTH, SCREEN_HEIGHT)
            players.append(player)

        # Separate players by team
        team_a_players = [player for player in players if player.color == COLOR_BLUE]
        team_b_players = [player for player in players if player.color == COLOR_RED]

        # Initialize basketball at a random position
        attempts = 0
        while attempts < 1000:
            ball_x = BALL_RADIUS + _rng_below(SCREEN_WIDTH - (2 * BALL_RADIUS))
            ball_y = BALL_RADIUS + _rng_below(SCREEN_HEIGHT - (2 * BALL_RADIUS))
            
            # Check if the basketball is not too close to any player
            valid_position = True
            for player in players:
                dx = ball_x - player.x
                dy = ball_y - player.y
                distance = math.sqrt(dx**2 + dy**2)
                
                # Ensure basketball is not placed too close to players
                if distance < PLAYER_RADIUS + BALL_RADIUS:
                    valid_position = False
                    break
            
            if valid_position:
                basketball = Basketball(ball_x, ball_y, BALL_RADIUS, COLOR_ORANGE)
                break
                
            attempts += 1
            
        # If we couldn't find a valid position, just place it in the center
        if attempts >= 1000:
            basketball = Basketball(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2, BALL_RADIUS, COLOR_ORANGE)

    except Exception as e:
        logger.error("Error in initializing the simulation: %s", e)
        raise

#%% Random Movement Simulation

if __name__ == "__main__":
    try:
        initialize_simulation()

        start_time_simulation = time.time()

        # Define the codec and create VideoWriter object
        video_format = cv2.VideoWriter_fourcc(*'XVID')
        try:
            video_output_path = os.path.join(video_dir, 'random_movement_video.mp4')
        except TypeError:
            video_output_path = os.path.join(os.getcwd(), 'assets/random_movement_video.mp4')
        out = cv2.VideoWriter(video_output_path, video_format, FPS, SCREEN_DIMENSIONS)

        frames_captured = 0
        # Define how long the simulation will run
        if os.path.exists('/.dockerenv') or os.getenv('GITHUB_ACTIONS') == 'true': # If running in Docker or GitHub Actions
            simulation_capture_max_time = 1
        else:
            simulation_capture_max_time = 10  # 10 seconds for local development

        max_frames_captured = FPS * simulation_capture_max_time
        simulating = True

        # Pygame clock for maintaining frame rate
        clock = pygame.time.Clock()

        while simulating and (frames_captured < max_frames_captured):
        
            elapsed_time_simulation = time.time() - start_time_simulation
        
            # Create screen with basketball court as the background
            screen.blit(background_image, (0,0))

            # Check for quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    simulating = False
                # Press 'q' to quit
                elif (event.type == pygame.KEYDOWN) and (event.key == pygame.K_q):
                    simulating = False

            # Update and draw players
            for player in players:
                player.move()
                player.draw()
            
            # Update and draw basketball (independent movement)
            basketball.move()
            basketball.draw()

            pygame.display.flip()  # Update pygame simulation frame
            clock.tick(FPS)  # Maintain frame rate

            # Capture frame
            frame = pygame.surfarray.array3d(pygame.display.get_surface())
            frame = frame.transpose([1, 0, 2])  # transpose to the correct shape
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # convert from RGB to BGR

            # Skip the first frame to allow simulation to stabilize
            if frames_captured > 0:
                # Write the output frame
                out.write(frame)
            frames_captured += 1

        logger.debug("Random Movement Simulation succeeded")
        print("Random Movement Simulation succeeded")

    except Exception as e:
        logger.error("Error during simulation: %s", e)
        print("Random Movement Simulation failed")
        raise

    finally:
        out.release()
        pygame.quit()