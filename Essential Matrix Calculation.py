import cv2 as cv
import numpy as np
import os
from matplotlib import pyplot as plt

# Defining Aruco Dictionary parameters
aruco_dict= cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_1000)
marker_length= 2.0 #Doesn't effect the working
marker_separation= 0.6
arucoparams= cv.aruco.DetectorParameters_create()

#Reading Images
Left_img= cv.imread("HW2/left.jpg")
Right_img= cv.imread("HW2/right.jpg")
#Converting to Grayscale
Left_img= cv.cvtColor(Left_img, cv.COLOR_BGR2GRAY)
Right_img= cv.cvtColor(Right_img, cv.COLOR_BGR2GRAY)
#Detecting Aruco Tags and Extracting corners and ids
corners_L, ids_L, rejected_L= cv.aruco.detectMarkers(Left_img, aruco_dict, parameters= arucoparams)
corners_R, ids_R, rejected_R= cv.aruco.detectMarkers(Right_img, aruco_dict, parameters= arucoparams)

corners_L= np.asarray(corners_L)
ids_L=np.asarray(ids_L)
corners_R= np.asarray(corners_R)
ids_R=np.asarray(ids_R)
#print(ids_L)
#print(corners_L)

#X= sorted(zip(ids_L,corners_L))

#print(X)
#print(X[0][1])


L_I_C= zip(ids_L, corners_L)
#print(L_I_C)
L_I_C= sorted(L_I_C)
corners_L= [ corners for ids,corners in L_I_C ]
corners_L= np.array(corners_L)
#print (type(corners_L))
#corners_L= (corners_L.reshape(32,2))
#print(corners_L)

R_I_C= zip(ids_R, corners_R)
#print(R_I_C)
R_I_C= sorted(R_I_C)
corners_R= [ corners for ids,corners in R_I_C ]
corners_R= np.array(corners_R)
#print (type(corners_R))
#corners_R= (corners_R.reshape(32,2))
#print(corners_R)

corners_L=(corners_L.reshape(32,2))
corners_R=(corners_R.reshape(32,2))
F, mask= cv.findFundamentalMat(corners_L, corners_R, cv.FM_8POINT)

#print(F)

lines1 = cv.computeCorrespondEpilines(corners_R.reshape(-1,1,2), 2,F)
lines2 = cv.computeCorrespondEpilines(corners_L.reshape(-1,1,2), 1,F)
lines1 = lines1.reshape(-1,3)
lines2 = lines2.reshape(-1,3)
#print(lines1)
#print(lines2)

def drawlines(img1,img2,lines,pts1,pts2):
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,3)
        #img1 = cv.circle(img1,tuple(pt1),5,color,-1)
        #img2 = cv.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

img5,img6 = drawlines(Left_img,Right_img,lines1,corners_L,corners_R)
img3,img4 = drawlines(Right_img,Left_img,lines2,corners_R,corners_L)

plt.subplot(121),plt.imshow(img5)
plt.subplot(122),plt.imshow(img3)
plt.show()


Camera_Matrix =[[1.37703226e+03, 0.00000000e+00, 9.89500043e+02],
 [0.00000000e+00 ,1.38159326e+03, 5.89329448e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]]

#print(np.array(Camera_Matrix).shape)

Camera_Matrix= np.array(Camera_Matrix)
#print(type(Camera_Matrix))
essential_matrix= np.matmul(Camera_Matrix.T, np.matmul(F, Camera_Matrix))

R1, R2, t= cv.decomposeEssentialMat(essential_matrix)

print( "Fundamental Matrix is :: ", F)
print("\nR1 matrix is :::",R1)
print("\nR2 matrix is :::",R2)
print("\nt matrix is :::",t)
print("\nEssential Matrix is :: ",essential_matrix)