import cv2 as cv

img = cv.imread('SimpleTestImages/Seven.png') # Reading the specified image

cv.imshow('Seven', img) # Showing the specified image

cv.waitKey(0) # Not immediately ending the program