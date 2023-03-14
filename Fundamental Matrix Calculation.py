import cv2 as cv
import numpy as np
import os

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
# Zipping to sort based on Tag ids
L_I_C= zip(ids_L, corners_L)

L_I_C= sorted(L_I_C)
corners_L= [ corners for ids,corners in L_I_C ]
corners_L= np.array(corners_L)

R_I_C= zip(ids_R, corners_R)

R_I_C= sorted(R_I_C)
corners_R= [ corners for ids,corners in R_I_C ]
corners_R= np.array(corners_R)


corners_L=(corners_L.reshape(32,2))
corners_R=(corners_R.reshape(32,2))
col= np.zeros((32,1))
points1= np.hstack((corners_L, col))
col= np.zeros((32,1))
points2= np.hstack( (corners_R,col))


def Compute_Fundamental_Matrix(X, X_dash):
    
    X= X
    X_dash= X_dash
    
    A=[]


    for i in range(len(X)):

        x= X[i][0]
        y= X[i][1]
        x_dash= X_dash[i][0]
        y_dash= X_dash[i][1]
            
        A.append([x_dash*x,x_dash*y,x_dash,y_dash*x,y_dash* y,y_dash,x,y,1])
    
    _,_,V_T= np.linalg.svd(A, full_matrices=True)
   

    F_hat= V_T.T[:,-1]
    
    F_hat=np.reshape(F_hat,(3,3))
    

    U,S,V_T= np.linalg.svd(F_hat)
    

    S_diagonal= np.diag(S)
   
    
    S_diagonal[-1,-1] = 0
    
    
    
    F_normalised= np.matmul(U, np.matmul(S_diagonal, V_T.T))
    F_normalised[2][2]= round(F_normalised[2][2])  
    return F_normalised


def Compute_fundamental_matrix_normalised(points1, points2):

    centroid1= np.array([np.mean(points1[:,0]), np.mean(points1[:,1])])
    centroid2= np.array([np.mean(points2[:,0]), np.mean(points2[:,1])])

    squared_dist1=[]
    for pt in points1:
        squared_dist1.append( (pt[0] - centroid1[0])**2+ ((pt[0] - centroid1[0])**2 ))
    mean_squared_distance1= np.sqrt(np.mean(squared_dist1))

    pt=0
    squared_dist2=[]
    for pt in points2:
        squared_dist2.append( (pt[0] - centroid2[0])**2+ ((pt[0] - centroid2[0])**2 ))
    mean_squared_distance2= np.sqrt(np.mean(squared_dist2))

    translation1 = np.array([[1., 0., -centroid1[0]],
                             [0., 1., -centroid1[1]],
                             [0., 0., 1.]])
    scaling1 = np.array([[np.sqrt(2.)/mean_squared_distance1, 0., 0.],
                         [0., np.sqrt(2.)/mean_squared_distance1, 0.],
                         [0., 0., 1.0]])
    
    T1 = np.dot(scaling1, translation1)
    normalized_points1 = np.dot(T1, points1.T).T

    translation2 = np.array([[1., 0., -centroid2[0]],
                             [0., 1., -centroid2[1]],
                             [0., 0., 1.]])
    scaling2 = np.array([[np.sqrt(2.)/mean_squared_distance2, 0., 0.],
                         [0., np.sqrt(2.)/mean_squared_distance2, 0.],
                         [0., 0., 1.0]])

    T2 = np.dot(scaling2, translation2)
    normalized_points2 = np.dot(T2, points2.T).T

    F_normalised= Compute_Fundamental_Matrix(normalized_points1, normalized_points2)  

    #print(F_normalised) 

    F = np.dot(T2.T, np.dot(F_normalised, T1))  

    #print(F)
    
    #Checking whether the F matrix is correct or not using T'xFxT= 0 property
    xasxasd= np.matmul(F, points1.T)
    check= np.matmul(points2.T, xasxasd.T)
    #print(check)

    return 

#F= Compute_fundamental_matrix_normalised(points1, points2)
F= Compute_Fundamental_Matrix(points1, points2)
print(F)