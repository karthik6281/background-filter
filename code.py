import cv2
import  cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation

cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)

path1 = "images/2.jpg"
img1 = cv2.imread(path1)
bc1 = cv2.resize(img1, (640, 480),interpolation = cv2.INTER_LINEAR)

path2 = "images/3.jpg"
img2 = cv2.imread(path2)
bc2 = cv2.resize(img2,(640,480),interpolation = cv2.INTER_LINEAR)

imgList = []
imgList.append(bc1)
imgList.append(bc2)

segmentor = SelfiSegmentation()
fps = cvzone.FPS()

i = 0

while True :
    success , img = cap.read()
    imgOut = segmentor.removeBG(img,imgList[i],threshold=0.8)

    imgStacked = cvzone.stackImages([img,imgOut],2,0.8)
    _,imgStacked = fps.update(imgStacked)

    cv2.imshow("image-stacked", imgStacked)
    key = cv2.waitKey(1)
    if key == ord('a'):
        i = 0
    if key == ord('d'):
        i = 1
    if key == ord('q'):
        break