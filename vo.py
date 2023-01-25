import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

path = os.listdir("C:/Users/mariam/Desktop/projectuav/undistorted")

intrinsic_matrix = np.array([[1.18162463e+03, 0.00000000e+00, 1.38216566e+03, 0.00000000e+00],
                             [0.00000000e+00, 1.18748291e+03, 8.33757136e+02, 0.00000000e+00],
                             [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])

calibration_matrix = np.array([[1.18162463e+03, 0.00000000e+00, 1.38216566e+03],
                               [0.00000000e+00, 1.18748291e+03, 8.33757136e+02],
                               [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

points = [[1, 1, 1, 1]]
transform = np.eye(4, 4)
def creatTransformationMatrix(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[0][-1] = t[0]
    T[1][-1] = t[1]
    T[2][-1] = t[2]
    return T

# def rotation_angles(matrix):

#     r11, r12, r13 = matrix[0]
#     r21, r22, r23 = matrix[1]
#     r31, r32, r33 = matrix[2]
#     theta1 = np.arctan(r21 / r11)
#     theta2 = np.arctan(-r31 * np.cos(theta1) / r11)
#     theta3 = np.arctan(r32 / r33)
#     theta1 = theta1 * 180 / np.pi
#     theta2 = theta2 * 180 / np.pi
#     theta3 = theta3 * 180 / np.pi

#     return (theta1, theta2, theta3)

# def compute_trajectory(relative_rotation, relative_translation, relative_scale):
#     # Convert the relative rotation from axis-angle representation to a 3x3 rotation matrix
#     rot_mat = np.zeros((3,3))
#     rot_mat[0,0] = np.cos(relative_rotation[2])*relative_scale[0]
#     rot_mat[0,1] = -np.sin(relative_rotation[2])*relative_scale[1]
#     rot_mat[1,0] = np.sin(relative_rotation[2])*relative_scale[0]
#     rot_mat[1,1] = np.cos(relative_rotation[2])*relative_scale[1]
#     rot_mat[2,2] = relative_scale[2]
    
#     # Compute the 3D transformation matrix
#     transform_matrix = np.eye(4)
#     transform_matrix[:3,:3] = rot_mat
#     transform_matrix[0][-1] = relative_translation[0]
#     transform_matrix[1][-1] = relative_translation[1]
#     transform_matrix[2][-1] = relative_translation[2]
    
#     return transform_matrix 
# indexParams = dict(algorithm = 0, trees=5) 
# searchParams = dict(checks=50)
detector = cv2.ORB_create()
bf = cv2.BFMatcher()

P1 = intrinsic_matrix
# T0 = np.eye(4, 4)
for i in tqdm(range(1, len(path) - 2000)):
    img1 = cv2.imread(f"C:/Users/mariam/Desktop/projectuav/undistorted/frame{i-1}.jpg")
    img2 = cv2.imread(f"C:/Users/mariam/Desktop/projectuav/undistorted/frame{i}.jpg")

    kp1, des1 = detector.detectAndCompute(img1, None)
    kp2, des2 = detector.detectAndCompute(img2, None)

    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    pts1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in good_matches])

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)

    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]
    
    E, _ = cv2.findEssentialMat(pts1, pts2, calibration_matrix, method=cv2.RANSAC, prob=0.999, threshold=3.0)
    pts1 = pts1[_.ravel() == 1]
    pts2 = pts2[_.ravel() == 1]
    pointss, R, t, mask = cv2.recoverPose(E, pts1, pts2, calibration_matrix)
    transform = creatTransformationMatrix(R, t) @ transform
    P2 = intrinsic_matrix @ transform 
    
    points_3d1 = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    P1 = P2

    points_3d2 = creatTransformationMatrix(R, t) @ points_3d1

    points_3d1 /= points_3d1[3]
    points_3d2 /= points_3d2[3]

    points_3d1 = points_3d1[:3].T

    points_3d2 = points_3d2[:3].T

    relative_scale = np.mean(np.linalg.norm(points_3d1[:-1] - points_3d1[1:], axis = -1)/np.linalg.norm(points_3d2[:-1] - points_3d2[1:], axis = -1) < 10)
    # t[0] = relative_scale * t[0]
    # t[1] = relative_scale * t[1]
    # t[2] = relative_scale * t[2]
    t = relative_scale * t
    # R[0][0] = relative_scale * R[0][0]
    # R[0][1] = relative_scale * R[0][1]
    # R[0][2] = relative_scale * R[0][2]
    # R[1][0] = relative_scale * R[1][0]
    # R[0][1] = relative_scale * R[0][1]
    # R[0][2] = relative_scale * R[0][2]
    # R[2][0] = relative_scale * R[2][0]
    # R[0][1] = relative_scale * R[0][1]
    # R[0][2] = relative_scale * R[0][2]
    # R = relative_scale * R
    # T0[:3, 3] = relative_scale * T0[:3, 3]
    # point = R @ points[-1].T
    # point[0] = point[0] + t[0]
    # point[1] = point[1] + t[1]
    # point[2] = point[2] + t[2]
    # points.append(point)
    T = creatTransformationMatrix(R, t)
    
    # R = T[:3, :3]
    # t = T[:3, -1]
    point = points[-1] @ T
    points.append(point)
    transform = T @ transform
    


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.set_xlim([-20, 20])
ax.set_ylim([-20, 20])
ax.set_zlim([-25, 25])

x, y, z, h = zip(*points)
ax.scatter(x, y, z)
 
plt.show()
