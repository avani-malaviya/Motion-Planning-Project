import cv2
import numpy as np
import csv
import pandas as pd

img = cv2.imread('D_star/Floor Plan.jpg')
image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
r_channel, g_channel, b_channel = cv2.split(image_rgb)
r_thresh = 230
g_thresh = 230
b_thresh = 230
# consider blue channel since b_brown = 0
r_threshed = cv2.threshold(r_channel, r_thresh, 255, cv2.THRESH_BINARY)[1]
g_threshed = cv2.threshold(g_channel, g_thresh, 255, cv2.THRESH_BINARY)[1]
b_threshed = cv2.threshold(b_channel, b_thresh, 255, cv2.THRESH_BINARY)[1]
# Denoise to remove grains
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
binaryMask = cv2.morphologyEx(b_threshed, cv2.MORPH_CLOSE, kernel)

binaryMask = cv2.resize(binaryMask,(0,0), fx=0.3, fy=0.3)
final_image = []

height, width = binaryMask.shape
for y in range(height):
    for x in range(width):

        grayscale_value = binaryMask[y, x]
        final_image.append([x,y,grayscale_value])
            

final_image = pd.DataFrame(final_image)
final_image.to_csv("D_star/refinedmap.csv", header=False, index=False)


img = cv2.circle(img, (1,1), 2, (255,133,133), 10)
cv2.imshow('window',binaryMask)

cv2.waitKey(0)