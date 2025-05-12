import pygame
import random
import cv2
import mediapipe as mp
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
screen_width = 600
screen_height = 480
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Hand-Controlled Running Game")

# Colors
black = (0, 0, 0)
white = (255, 255, 255)
green = (0, 255, 0)  # Player Color
red = (255, 0, 0)  # Enemy Color
blue = (0, 0, 255)  # Background Color

# Player properties
player_size = 20
player_speed = 5
player_x = screen_width // 4  # Start near the left
player_y = screen_height // 2

# Enemy properties
enemy_size = 20

enemy_speed_base = 9  # Initial enemy speed
enemy_speed_increase = 0.2  # Rate at which enemy speed increases
enemy_speed_max = 15  # Maximum enemy speed
enemies = []  # List to hold multiple enemies

# Function to calculate the direction vector of the index finger, adapted for pygame
def get_finger_direction_and_position(landmarks, frame_width, frame_height):
    try:
        # Get index finger base and tip landmarks
        index_base = landmarks[mp_hands.HandLandmark.INDEX_FINGER_MCP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]

        # Calculate the direction from base to tip of the index finger
        direction = np.array([index_tip.x - index_base.x, index_tip.y - index_base.y, index_tip.z - index_base.z])

        # Normalize the direction vector
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction = direction / norm

        # Get position for player
        x_pos = int(index_tip.x * frame_width)
        y_pos = int(index_tip.y * frame_height)

        return direction, x_pos, y_pos
    except Exception as e:
        print(f"Error in get_finger_direction: {e}")
        return None, None, None

# Initialize MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Function to spawn an enemy
def spawn_enemy(score):
    enemy_x = screen_width  # Start off-screen to the right
    enemy_y = random.randint(0, screen_height - enemy_size)

    # Calculate enemy speed based on score, up to a maximum
    enemy_speed = min(enemy_speed_base + score * enemy_speed_increase, enemy_speed_max)

    enemies.append([enemy_x, enemy_y, enemy_speed])  # Store the enemy speed as well


# Game loop
def gameLoop():
    global player_x, player_y, enemies

    game_over = False
    score = 0

    # Spawn initial enemies
    for _ in range(10):  # Start with 10 enemies
        spawn_enemy(score)

    clock = pygame.time.Clock()

    # Font for displaying the score
    font = pygame.font.Font(None, 36)

    # Start webcam stream
    while not game_over:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip the frame horizontally for a selfie-view
        frame = cv2.flip(frame, 1)

        # Convert the frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(rgb_frame)

        # If hands are detected
        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Get the direction of the index finger
                direction, x_pos, y_pos = get_finger_direction_and_position(landmarks.landmark, frame.shape[1], frame.shape[0])

                if direction is not None:
                    # Calculate potential new player position
                    new_player_x = player_x + direction[0] * player_speed
                    new_player_y = player_y + direction[1] * player_speed

                    # Boundary checks - player dies if they hit a wall
                    if new_player_x < 0 or new_player_x > screen_width - player_size or new_player_y < 0 or new_player_y > screen_height - player_size:
                        game_over = True
                    else:
                        player_x = new_player_x
                        player_y = new_player_y

        # Enemy movement and collision detection
        for i, enemy in enumerate(enemies):
            enemy_x, enemy_y, enemy_speed = enemy
            enemy_x -= enemy_speed  # Move enemies to the left

            # Collision detection (simple rectangle collision)
            if (player_x < enemy_x + enemy_size and
                    player_x + player_size > enemy_x and
                    player_y < enemy_y + enemy_size and
                    player_y + player_size > enemy_y):
                game_over = True
                break  # End the game if there's a collision

            # Remove enemies that go off-screen and spawn new ones
            if enemy_x < -enemy_size:
                enemies.pop(i)
                spawn_enemy(score)  # Pass score to spawn_enemy
                score += 1  # Increment the score each time you avoid an enemy
            else:
                enemies[i] = [enemy_x, enemy_y, enemy_speed]  # Update enemy position and speed

        # Drawing everything
        screen.fill(blue)  # Background color
        pygame.draw.rect(screen, green, (player_x, player_y, player_size, player_size))  # Draw player

        # Draw enemies
        for enemy_x, enemy_y, _ in enemies:
            pygame.draw.rect(screen, red, (enemy_x, enemy_y, enemy_size, enemy_size))

        # Display the score
        score_text = font.render(f"Score: {score}", True, white)
        screen.blit(score_text, (10, 10))

        pygame.display.update()

        # Frame rate control
        clock.tick(30)

        # Press 'q' to exit the OpenCV part
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    cap.release()
    # Destroy all the windows
    cv2.destroyAllWindows()
    pygame.quit()
    quit()


# Run the game
gameLoop()