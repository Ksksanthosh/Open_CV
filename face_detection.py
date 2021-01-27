#face detection in image
import cv2 as cv
faceCascade = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
img = cv.imread("Lenna.png")
imgGray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# img=cv.VideoCapture(1)
# img.set(3,640)
# img.set(4,480)


faces=faceCascade.detectMultiScale(img,1.1,4)

for x,y,w,h in faces:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
cv.imshow("CAP",img)
cv.waitKey(0)