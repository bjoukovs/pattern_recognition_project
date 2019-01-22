import cv2
import os, fnmatch

def refine(input_dir='handwritten_imgs/original/', output_dir='handwritten_imgs/refined/', pattern = "*"):
	listOfFiles = os.listdir(input_dir)
	#pattern = "handwritten*"

	for entry in listOfFiles:
	    if fnmatch.fnmatch(entry, pattern):
    		im = cv2.imread(input_dir + entry)
    		h, w, c = im.shape[:3]
#    		print(h,w,c)
    		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    		t, ret = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    		ret = cv2.GaussianBlur(ret, (3, 3), 0)
    		cv2.imwrite(output_dir+entry, ret)

	#cv2.imshow('img', ret)
	#cv2.waitKey(0)
