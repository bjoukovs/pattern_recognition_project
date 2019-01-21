
try:
	import cv2
except ImportError:
	print("OpenCV is not installed. Please do this by running pip install opencv-python==3.4.4.19")
	quit()
try:
	import numpy as np
except ImportError:
	print("Numpy is not installed. Please do this by running pip install numpy")
	quit()
import os
from digit_extraction.refine import refine
from digit_extraction.detect import detect

ORIGINAL_IMG_DIR = 'handwritten_imgs/original/'
REFINIED_IMG_DIR = 'handwritten_imgs/refined/'
EXTRACTED_DIGITS_DIR = 'handwritten_extracted/'

print("please put the image containing the digits in the folder called \"{}\". Make sure that it is in JPEG format.".format(ORIGINAL_IMG_DIR))
input('Press ENTER to continue')
print("Refining images for better digit detection...", end='')
refine(input_dir=ORIGINAL_IMG_DIR, output_dir=REFINIED_IMG_DIR)
print("DONE")
print("Refinied images are in \"{}\"".format(REFINIED_IMG_DIR))
file = None
while True:
    print("Select a file to read the digits from:")
    listFiles = os.listdir(REFINIED_IMG_DIR)
    print(''.join(["[{}] {}\n".format(n+1, file) for n, file in enumerate(listFiles)]))
    index = input('Enter file number:')
    try:
        index = int(index)
    except:
        print("Invalid input")
        continue
    if index <= len(listFiles) and index > 0:
        file = listFiles[index-1]
        break
    else:
        print("Index out of bounds, try again")
print("Extracting digits from {}...".format(file), end='')
img, numd = detect(os.path.join(REFINIED_IMG_DIR, file), EXTRACTED_DIGITS_DIR)
print("DONE")
print("{} digits extracted".format(numd))
cv2.imshow('Found digits', img)
cv2.waitKey(0)


