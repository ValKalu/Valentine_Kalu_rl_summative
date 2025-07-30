import pygame
import os
import numpy as np
import imageio

# Set SDL_VIDEODRIVER to 'dummy' for headless environments
os.environ["SDL_VIDEODRIVER"] = "dummy" 

pygame.init()

# Set up the drawing window
screen_width = 400
screen_height = 300
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pygame Headless Test")

# Fill the background
screen.fill((0, 0, 0)) # Black background

# Draw a simple red circle
pygame.draw.circle(screen, (255, 0, 0), (screen_width // 2, screen_height // 2), 50)

# Update the display (important for rendering to the buffer)
pygame.display.flip()

# Capture the screen as a numpy array
# Transpose is needed because pygame.surfarray.array3d returns (width, height, channels)
# but imageio expects (height, width, channels)
frame = np.transpose(pygame.surfarray.array3d(screen), (1, 0, 2))

# Save the frame as a PNG image
output_filename = "test_pygame_render.png"
try:
    imageio.imwrite(output_filename, frame)
    print(f"Test image saved to {output_filename}")
except Exception as e:
    print(f"Error saving image: {e}")

# Quit Pygame
pygame.quit()
