import sympy as sp
import cv2
import numpy as np
from modules.functions import *

# hyperparameters
# f = 600
f = 700
a= 1
b= 0
dx = 600
dy = 400
distCoefs = [-0.245031, 0.071524, -0.00994978]
pointlist=[np.array([[1.0],[1.0]]),np.array([[0.0],[3.0]]),np.array([[2.0],[3.0]]),np.array([[2.0],[4.0]])]
destpointlist=[np.array([[-0.33333333],[-0.66666667]]),np.array([[-0.33333333],[-2.0]]),np.array([[-1.0],[-2.0]]),np.array([[-1.0],[-2.66666667]])]

H=np.array(
[   [-2, 0, 1],
    [0, -2, 0],
    [0, 0, 3]])

K=np.array(
[   [f, b*f, dx],
    [0, a*f, dy],
    [0, 0, 1]])

R1=np.eye(3)
R2=np.eye(3)
t1=np.array([[0.0],[0.0],[1.0]])
t2=np.array([[0.0],[0.0],[20.0]])
Q_3d = np.array([[1.0],[1.0],[0.0]])

P1 = calc_projection_matrix(K, R1, t1)
P2 = calc_projection_matrix(K, R2, t2)

q1 = project_points_with_Proj_matrix(P=P1, Q=PiInv(Q_3d))
q2 = project_points_with_Proj_matrix(P=P2, Q=PiInv(Q_3d))

q1bar = q1 + np.array([[1.0],[-1.0]])
q2bar = q2 + np.array([[1.0],[-1.0]])

triang = triangulate(np.array([q1bar, q2bar]), np.array([P1, P2])).reshape(4,1) # 4,

def info(q1bar, q2bar, q1proj, q2proj, Qhat, Qori):
    print(f"pixels in camera 1 are {np.linalg.norm(( q1bar - q1proj))}") #pixels
    print(f"pixels in camera 2 are {np.linalg.norm(( q2bar - q2proj))}") #pixels
    q1_dist = np.linalg.norm(q1bar - q1proj)
    q2_dist = np.linalg.norm(q2bar - q2proj)
    print(f'Distance between reprojection and original point for q1: {q1_dist}')
    print(f'Distance between reprojection and original point for q2: {q2_dist}')
    print(f"general error is {np.linalg.norm(Qori - Pi(Qhat))}") # ||Q - Qswigle||^2 error
    print(50*"-")


q1projected = project_points_with_Proj_matrix(P1, triang)
q2projected = project_points_with_Proj_matrix(P2, triang)


Qbar=triangulate_nonlin(np.array([q1bar, q2bar]), np.array([P1, P2]))
print(Pi(Qbar))
print((Qbar))

info(q1bar=q1bar, q2bar=q2bar, q1proj=q1projected, q2proj=q2projected, Qori=Q_3d, Qhat=triang)

q1_reproj = Pi(project_points_with_Proj_matrix(P1, Qbar))
q2_reproj = Pi(project_points_with_Proj_matrix(P2, Qbar))

info(q1bar=q1bar, q2bar=q2bar, q1proj=q1_reproj, q2proj=q2_reproj, Qori=Q_3d, Qhat=Qbar)

