import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('SimpleTestImages/stopsign.jpg') # Reading the specified image

# Creates both a grayscale and color image
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Shows picture
plt.subplot(1,1,1)
plt.imshow(img_rgb)
plt.show()




