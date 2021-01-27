import cv2 as cv
import numpy as np
def stacking (n_c,input_images,widht=300,height=300):
    lenght=len(input_images)
    n_r=int(lenght/n_c)
    # print("lenght",lenght)
    images=[0]*lenght
    f=0
    for i in input_images:
        i=cv.resize(i,(widht,height))
        if(len(i.shape)>2):
            images[f]=i
            # pass
        else:
            images[f] =cv.cvtColor(i,cv.COLOR_GRAY2BGR)
        f=f+1
    #####For end #####

    #Stackin images#
    h=[0]*n_r
    j=0
    i=0
    while(i<lenght):
        h1=np.hstack(images[i:(i+n_c)])
        i=i+n_c
        h[j]=h1
        j=j+1
    vertical=np.vstack(h)
    return (vertical)

def getcountours(img):
    countours,hierarchy=cv.findContours(img,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    maxarea = 2000
    biggest=np.array([])
    for cnt in countours:
        area=cv.contourArea(cnt)
        # print(area) # to print the area of the contours
        # cv.drawContours(imgcopy, cnt, -1, (0, 0, 255), 3)  # '-1' to print all the contours
        if(area>2000):
            # cv.drawContours(imgcopy, cnt, -1, (0, 0, 255), 3)  # '-1' to print all the contours
            # perimeter of the contours
            peri=cv.arcLength(cnt,True)
            # print(peri)
            # corner points of the contour
            approx=cv.approxPolyDP(cnt,0.02*peri,True)
            # print(approx,area,len(approx))
            if area > maxarea :
                # print("in if")
                biggest=approx
                maxarea=area
    cv.drawContours(imgcopy, biggest, -1, (0, 0, 255), 20)
    return biggest


def preprosessing(img):
    imggray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    imgblur=cv.GaussianBlur(imggray,(5,5),1)
    imgcanny=cv.Canny(imgblur,100,100)
    kernel=np.ones((5,5))
    imgdila=cv.dilate(imgcanny,kernel,iterations=2)
    imgthers=cv.erode(imgdila,kernel,iterations=1)
    # final=stacking(3,[img,imggray,imgblur,imgcanny,imgdila,imgthers],350,400)
    return imgthers
def reorder(point):
    point=point.reshape((4,2))
    points_reorder=np.zeros((4,1,2),np.int32)
    sum=point.sum(1)
    # print(point,"\n",sum)
    points_reorder[0]=point[np.argmin(sum)]
    points_reorder[3]=point[np.argmax(sum)]
    diff=np.diff(point,axis=1)
    points_reorder[1]=point[np.argmin(diff)]
    points_reorder[2] = point[np.argmax(diff)]
    # print(point,"\n",points_reorder)
    return points_reorder

def wrap(img,biggest,widthimg,heightimg):
    # print(biggest.shape)
    pt = np.float32(biggest)
    pt2 = np.float32([[0, 0], [widthimg, 0], [0, heightimg], [widthimg, heightimg]])
    matrix = cv.getPerspectiveTransform(pt, pt2)
    imgout = cv.warpPerspective(img, matrix, (widthimg, heightimg))
    return imgout


doc=cv.imread("../res/doc.png")
# img=stacking(1,[doc])
# success,img= cap.read()
# print(doc.shape)
widthimg=300
heightimg=400
doc=cv.resize(doc,(widthimg,heightimg))
imgcopy=doc.copy()

imgthresh=preprosessing(doc)

biggest=getcountours(imgthresh)

mypoints=reorder(biggest)

mypoints=reorder(biggest)
rect_point=mypoints.reshape((4,2))

rectangle=doc.copy()
cv.rectangle(rectangle,(rect_point[0][0],rect_point[0][1]),(rect_point[3][0],rect_point[3][1]),(0,0,0),4)
imgwrapped=wrap(doc,mypoints,widthimg,heightimg)
result=stacking(2,[doc,imgcopy,rectangle,imgwrapped],widthimg,heightimg)
# print(mypoints.shape,mypoints)
cv.imshow("result",result)
while True:
    if cv.waitKey(1) & 0xFF == ord('e'):
        break