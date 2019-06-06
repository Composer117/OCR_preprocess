import numpy as np
import cv2

def order_points(pts):
    #make a zeros array for 4 points
    rect = np.zeros((4,2), dtype = "float32")

    #get top left and bottom right point by getting sum of coordinates of all points
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    #get top left and bottom right point by getting difference of coordinates of all points
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]

    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    #computing width of new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    #computing length of new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    #making array containing positions of the pixels at 4 boundary points of the new image
    dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

    #computing transform array from the boundary points of new and old images
    M = cv2.getPerspectiveTransform(rect, dst)
    #applying perspective transform using transform array M
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    return warped