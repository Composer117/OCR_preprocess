import fpoint_transform
from fpoint_transform import four_point_transform
import skimage
from skimage.filters import threshold_local
import numpy as np
import cv2 
import imutils

path = r"C:\Users\nauri\source\repos\OCR_preprocess\OCR_preprocess\images"
image = cv2.imread(path + "\img1.jpg")
ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 75, 200)

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

for c in cnts:
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
	if len(approx) == 4:
		screenCnt = approx
		break

warped = four_point_transform(orig, screenCnt.reshape(4,2) * ratio)
warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
T = threshold_local(warped, 11, offset = 10, method = "gaussian")
warped = (warped > T).astype("uint8") * 255

p_image = cv2.imwrite(r'C:\Users\nauri\source\repos\OCR_preprocess\OCR_preprocess\images\res.jpg', warped)
print("Image processed and saved as 'res' in images folder")