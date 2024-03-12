#!/usr/bin/env python3

import cv2
import numpy as np
from glob import glob
import argparse

###########################################

# Script to make a mp4 video out of tif, png, jpg frames
# made by Tancrede Leger & Guillaume Jouvet

###########################################

parser = argparse.ArgumentParser(description="Video maker from a serie of images")
 
parser.add_argument(
    "--pattern",
    type=str,
    default="*.png",
    help="Pattern to search for PNG files in the directory",
)
parser.add_argument(
    "--output_file",
    type=str,
    default="film",
    help="Name of the output video file",
)
parser.add_argument(
    "--fps",
    type=int,
    default=10,
    help="Frames per second for the output video",
)
parser.add_argument(
    "--resize",
    type=float,
    default=1,
    help="Resize factor for the PNG files",
)
parser.add_argument(
    "--reverse",
    type=int,
    default=1,
    help="Reverse the order of the PNG files",
)
parser.add_argument(
    "--flipud",
    type=int,
    default=0,
    help="flipud",
) 

config = parser.parse_args() 

###########################################
  
# Get the list of PNG files in the directory
file_list = glob(config.pattern)

# Sort the list of PNG files
file_list.sort()

# Reverse the list of PNG files if the reverse flag is set
if config.reverse:
    file_list.reverse()

img = cv2.imread(file_list[0])

#print(img.shape)

frame_size = (int(img.shape[1]*config.resize), 
              int(img.shape[0]*config.resize))

#print(frame_size)

# Create a VideoWriter object
video_out = cv2.VideoWriter(config.output_file+'.mp4', 
                            cv2.VideoWriter_fourcc(*'mp4v'), 
                            config.fps, 
                            frame_size)

# Loop through the PNG files and add them to the video
for png_file in file_list:
    print(png_file)
    img = cv2.imread(png_file)
    if config.flipud == 1:
        img = np.flipud(img)
    if config.resize != 1:
        img = cv2.resize(img, frame_size) 
    video_out.write(img)

# Release the VideoWriter object
video_out.release()
