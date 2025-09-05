# This script calibrates a camera using a checkerboard pattern from a video file.
# Credit to Stefano Carlo Lambertenghi for the original code
# This code is adapted to work with this project's structure and requirements.

import numpy as np
import cv2

from config import CHECKERBOARD_ROWS, CHECKERBOARD_COLS


def generate_checkerboard(cols, rows, square_size_px=50, filename='checkerboard.png'):
    width = cols * square_size_px
    height = rows * square_size_px
    board = np.zeros((height, width), dtype=np.uint8)

    for y in range(rows):
        for x in range(cols):
            if (x + y) % 2 == 0:
                board[y*square_size_px:(y+1)*square_size_px,
                      x*square_size_px:(x+1)*square_size_px] = 255

    cv2.imwrite(filename, board)
    print(f"Checkerboard saved as {filename}")

# Example: 10x7 squares for 9x6 inner corners
generate_checkerboard(cols=CHECKERBOARD_COLS, rows=CHECKERBOARD_ROWS, square_size_px=60)