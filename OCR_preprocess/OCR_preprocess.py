import fpoint_transform
from fpoint_transform import four_point_transform
import skimage
from skimage.filters import threshold_local
import numpy as np
import cv2 
import imutils

#Getting image from path and then resizing it for preprocessing
path = r"C:\Users\nauri\source\repos\OCR_preprocess\OCR_preprocess\images"
image = cv2.imread(path + "\image2.jpg")
ratio = image.shape[0]/500.0
orig = image.copy()
image = imutils.resize(image, height=500)

#converting image to grayscale and then applying filters
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 9, 75, 75)
img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 4)	
img = cv2.medianBlur(img, 11)	
img = cv2.copyMakeBorder(img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[0, 0, 0])	

#applying erosion and dilation techniques to reduce noise in regions as well as reducing smaller irrelevant regions
thresh = cv2.erode(img, None, iterations=7)
thresh = cv2.dilate(thresh, None, iterations=7)

#applying canny filter to get edges
edges = cv2.Canny(thresh, 200, 250)

#finding countours in the image
contours = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:5]

height = edges.shape[0]	
width = edges.shape[1]
#MAX_COUNTOUR_AREA = (width - 10) * (height - 10)	

#maxAreaFound = MAX_COUNTOUR_AREA * 0.5	

#default initialization for page countours
pageContour = np.array([[5, 5], [5, height-5], [width-5, height-5], [width-5, 5]])	

for cnt in contours:
    # Simplify contour
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.03 * perimeter, True)

    # Page has 4 corners and it is convex
    # Page area must be bigger than maxAreaFound 
    if (len(approx) == 4 and
            cv2.isContourConvex(approx) and
            """maxAreaFound < cv2.contourArea(approx) < MAX_COUNTOUR_AREA"""):

        maxAreaFound = cv2.contourArea(approx)
        pageContour = approx

#gray = cv2.GaussianBlur(gray, (5, 5), 0)
#edged = cv2.Canny(gray, 75, 200)

#cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#cnts = imutils.grab_contours(cnts)
#c = max(cnts, key=cv2.contourArea)
#cv2.drawContours(image, [c], -1, (0, 255, 255), 2)

#cv2.imshow("Image", image)
#cv2.waitKey(0)

#cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

#for c in cnts:
#	peri = cv2.arcLength(c, True)
#	approx = cv2.approxPolyDP(c, 0.02 * peri, True)
#	if len(approx) == 4:
#		screenCnt = approx
#		break

#applying four point transform function from fpoint_transform.py file
warped = four_point_transform(orig, pageContour[:, 0] * ratio)

#warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
#T = threshold_local(warped, 11, offset = 10, method = "gaussian")
#warped = (warped > T).astype("uint8") * 255

p_image = cv2.imwrite(r'C:\Users\nauri\source\repos\OCR_preprocess\OCR_preprocess\images\res.jpg', warped)
print("Image processed and saved as 'res' in images folder")
