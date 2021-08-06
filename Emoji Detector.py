import cv2
import numpy as np
import time
import os
import HandTrackingModule as htm
from PIL import ImageFont, ImageDraw, Image



detector = htm.handDetector(max_num_hands = 1, min_detection_confidence = 0.7)



cap = cv2.VideoCapture(0)

wCam, hCam = 640,480

cap.set(3,wCam)
cap.set(4,hCam)

cTime = 0
pTime = 0



folderPath = "Emoji Pics"
myList = os.listdir(folderPath)
print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    overlayList.append(image)
print(len(overlayList))





while True:
    
    success, img = cap.read()
    
    imp = detector.findHands(img)
    lmList = detector.findPosition(img, draw = False)
    #print(lmList)
    
    
    # Placing so called logo
    
    logo = cv2.imread("logo.jpg")
    hl, wl, cl = logo.shape
    scale_percentLogo = 10 # percent of original size
    logowidth = int(logo.shape[1] * scale_percentLogo / 100)
    logoheight = int(logo.shape[0] * scale_percentLogo / 100)
    logoDim = (logowidth, logoheight)
      
    resized = cv2.resize(logo, logoDim, interpolation = cv2.INTER_AREA)
    hl, wl, cl = resized.shape
    img[360:hl+360, 0:wl] = resized
    
    
    
    # Initializing Fingers
    
    thumb = True
    index = True 
    middle = True
    ring = True
    pinky = True
    


    if(len(lmList) != 0):
        
################ ---Detecting Fingers--- #################
        
        if lmList[4][1] < lmList[2][1]:
            thumb = False
            
        if lmList[8][2] > lmList[6][2]:
            index = False
            
        if lmList[12][2] > lmList[10][2]:
            middle = False
        
        if lmList[16][2] > lmList[14][2]:
            ring = False
            
        if lmList[20][2] > lmList[18][2]:
            pinky = False
            
######################################### ---Conditions For Emoji--- #########################################
        
        if (index == True and middle == True and thumb == False and ring == False and pinky == False):
            h, w, c = overlayList[0].shape
            img[0:h, 0:w] = overlayList[0]            
            
        elif (index == True and pinky == True and middle == False and ring == False and thumb == False):
            h, w, c = overlayList[1].shape
            img[0:h, 0:w] = overlayList[1]
            
        elif (index == True and thumb == False and middle == False and ring == False and pinky == False):
            h, w, c = overlayList[2].shape
            img[0:h, 0:w] = overlayList[2]
        
        elif (index == True and thumb == True and middle == False and ring == False and pinky == False):
            h, w, c = overlayList[3].shape
            img[0:h, 0:w] = overlayList[3]    
            
        elif (middle == True and thumb == False and index == False and ring == False and pinky == False):
            h, w, c = overlayList[4].shape
            img[0:h, 0:w] = overlayList[4]
        
        elif (thumb == True and index == False and middle == False and ring == False and pinky == False):
            h, w, c = overlayList[5].shape
            img[0:h, 0:w] = overlayList[5]
            
        elif (thumb == True and index == True and middle == True and ring == True and pinky == True):
            h, w, c = overlayList[6].shape
            img[0:h, 0:w] = overlayList[6]
        
        elif (thumb == False and index == False and middle == False and ring == False and pinky == False):
            h, w, c = overlayList[7].shape
            img[0:h, 0:w] = overlayList[7]
            
        elif (thumb == True and index == False and middle == False and ring == False and pinky == True):
            h, w, c = overlayList[8].shape
            img[0:h, 0:w] = overlayList[8]
            
            
##################################################################################################################  
    
    
    #For Displaying FPS on Camera screen        
    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.rectangle(img, (440,18), (590,62), (255,255,253), cv2.FILLED)
    cv2.putText(img, f'FPS: {int(fps)}', (450,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,225),3)
    
    
    
    
    
    
    cv2.imshow("Image", img)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()