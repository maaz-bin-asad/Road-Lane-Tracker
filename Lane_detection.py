import cv2
import numpy as np
from matplotlib import pyplot as plt
def canny(lane_image):   #function to find sharp changes or derivatives of f(x,y) in adjacent pixels
    gray = cv2.cvtColor(lane_image, cv2.COLOR_BGR2GRAY)  # covert to grayscale to quickly appply canny edge
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply convolution with a 5 x 5 kernel to smoothen the image
    canny = cv2.Canny(blur, 50, 150)
    return canny
def region_of_interest(img):    #To focus on a lane we want to go
    height=img.shape[0]  # The method gives a list of height x width of image
    triangle=np.array([[(200,height),(1100,height),(550,250)]])  #To track the coordinates of the area we need
    mask=np.zeros_like(img)
    cv2.fillPoly(mask,triangle,255)
    masked_image=cv2.bitwise_and(img,mask)
    return masked_image
def display_lines(image,lines):
    line_image=np.zeros_like(image)   #To create image of same shape containing zeros
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)     #To split array of coordinates into four different variables
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
            return line_image
cap=cv2.VideoCapture('C:\Program Files\JetBrains\PyCharm Community Edition 2019.3.1\Detect.mp4')
while True:
    _,frame=cap.read()
    lane_image=np.copy(frame)    #Always work on a copy of original image
    canny_image=canny(lane_image)
    cropped_image=region_of_interest(canny_image)
    lines=cv2.HoughLinesP(cropped_image,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    line_image=display_lines(lane_image,lines)
    combo=cv2.addWeighted(lane_image,0.4,line_image,1,1)
    cv2.imshow('Tracked',combo)
    if cv2.waitKey(1)==27:
        break
cv2.destroyAllWindows()
cap.release()




