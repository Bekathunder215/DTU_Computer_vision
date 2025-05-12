#----------------------------
# exam 2023
#----------------------------

from function_folder.function_file import *
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

# Question 3
f = 1720
K = np.array([[f, 0, 680], [0, f, 610.0], [0, 0, 1]])
R = cv2.Rodrigues(np.array([-0.1, 0.1, -0.2]))[0]
t = np.array([0.09, 0.05, 0.05]).reshape(3, 1)
P = calc_projection_matrix(K,R,t)
print("Projection matrix:", P.shape)
dpoint = np.array([-0.03, 0.01, 0.59]).reshape(3, 1)
print("3D point:", dpoint.shape)
# Project the 3D point using the projection matrix
projected_point = project_points_with_Proj_matrix(P, PiInv(dpoint))
print("Projected point:", projected_point)

print(50*"-")

# Question 5
# You are given three cameras (1, 2 and 3) that share the same camera matrix
# K and have the following extrinsics. You can copy and paste the following into
# Python:
K = np.array([[900, 0, 1070], [0, 900, 610.0], [0, 0, 1]], float)
R1 = cv2.Rodrigues(np.array([-1.6, 0.3, -2.1]))[0]
t1 = np.array([[0.0], [1.0], [3.0]], float)
R2 = cv2.Rodrigues(np.array([-0.4, -1.3, -1.6]))[0]
t2 = np.array([[0.0], [1.0], [6.0]], float)
R3 = cv2.Rodrigues(np.array([2.5, 1.7, -0.4]))[0]
t3 = np.array([[2.0], [-7.0], [25.0]], float)
# You observe the same point in all three cameras, but with some noise. The
# observed points for cameras 1 to 3 are respectively:
p1 = np.array([[1046.0], [453.0]])
p2 = np.array([[1126.0], [671.0]])
p3 = np.array([[1165.0], [453.0]])
# How far is p2 from the epipolar line in camera 2 corresponding to p1?

F12 = fundamentalmatrix(K=K, R1=R1, t1=t1, R2=R2, t2=t2)
print("Fundamental matrix:\n", F12)
print("Fundamental matrix:", F12.shape)
# Calculate the epipolar line in camera 2 corresponding to p1
epipolar_line = F12 @ PiInv(p1)
print("Epipolar line:\n", epipolar_line)
# Calculate the distance from p2 to the epipolar line
distance = point_line_distance(epipolar_line, p2)
print("Distance from p2 to epipolar line:", distance)

print(50*"-")

# Question 6
# Use all three observations of the point from the previous question to triangulate
# the point with the linear algorithm from the slides. Do not normalize the points
P1 = calc_projection_matrix(K, R1, t1)
P2 = calc_projection_matrix(K, R2, t2)
P3 = calc_projection_matrix(K, R3, t3)

pointlist = np.array([p1, p2, p3])
projlist = np.array([P1, P2, P3])

X_3D = Pi(triangulate(pointlist, projlist))
print("3D point from triangulation:\n", X_3D)
print("3D point from triangulation:", X_3D.shape)

print(50*"-")

# Question 11
q = np.array([2, 4, 3])
line = np.array([1, 2, 2])
# Calculate the distance from point q to line
distance = point_line_distance(line, Pi(q))
print("Distance from point to line:", distance)

print(50*"-")

# Question 12
stuff = np.load("harris.npy", allow_pickle=True)
print("Stuff:", stuff)
gI_x2 = stuff.item()['g*(I_x^2)']
gI_y2 = stuff.item()['g*(I_y^2)']
gI_x_I_y = stuff.item()['g*(I_x I_y)']

k = 0.06
t = 516

def harrisMeasure(gI_x2, gI_y2, gI_x_I_y, k):
    ab = gI_x2 * gI_y2
    anb = gI_y2 + gI_x2
    c2 = gI_x_I_y * gI_x_I_y
    return ab - c2 - k * anb * anb


def cornerDetector(k, tau):
    r = harrisMeasure(gI_x2, gI_y2, gI_x_I_y, k)

    def non_maximum_suppression(r):
        # Pad the array with -inf to handle edge cases
        padded = np.pad(r, pad_width=1, mode="constant", constant_values=-np.inf)

        # Compare each pixel with its left, right, top, and bottom neighbors
        is_max = (
            (r > padded[1:-1, :-2])
            & (r > padded[1:-1, 2:])
            & (r > padded[:-2, 1:-1])
            & (r > padded[2:, 1:-1])
        )

        # Create a suppressed version where only local maxima are kept
        suppressed = np.where(is_max, r, 0)

        return suppressed

    r = non_maximum_suppression(r)

    return np.where(r >= tau, r, r >= tau)


c = cornerDetector(k, 516)
print("Corner detector result:", c)
y = np.nonzero(c)[0]
x = np.nonzero(c)[1]
points = np.array([x, y])
print("Corner points:", points)

print(50*"-")
# question 13
rans = np.load("materials/ransac.npy", allow_pickle=True)
# print("RANSAC data:", rans)
# print("RANSAC data item:", rans.item())
# print("RANSAC data item keys:", rans.item().keys())
points = rans.item()['points'].reshape(100, 2)
x1 = rans.item()['x1']
x2 = rans.item()['x2']
print("x1:", x1) 
print("x2:", x2)
print("Points shape:", points.shape) # (100,2)
destpoints = np.array([x1, x2])
print("Destination points shape:", destpoints.shape) 
tau = 0.2  # Threshold

# Calculate distances of all points to the line
def point_line_distance(p, x1, x2):
    """Compute perpendicular distance from point p to line through x1 and x2."""
    numerator = abs((x2[0] - x1[0]) * (x1[1] - p[1]) - (x1[0] - p[0]) * (x2[1] - x1[1]))
    denominator = math.sqrt((x2[0] - x1[0])**2 + (x2[1] - x1[1])**2)
    return numerator / denominator

def count_inliers(points, x1, x2, tau=0.2):
    """Return number of inliers for line through x1 and x2 given threshold tau."""
    inliers = []
    for p in points:
        distance = point_line_distance(p, x1, x2)
        if distance <= tau:
            inliers.append(p)
    return len(inliers), inliers

num_inliers, inliers = count_inliers(points, x1, x2, tau=0.2)
print("Number of inliers:", num_inliers)
# print("Inlier points:", inliers)

print(50*"-")
# question 14

# The required number of RANSAC trials (to reach confidence (p)) is given by

# [ N ;=; \frac{\ln(1 - p)}{\ln\bigl(1 - w^s\bigr)} ]

# where

# (p=0.95) is the desired success probability,
# (w=\frac{#\text{inliers}}{#\text{matches}}=\frac{103}{404}),
# (s=4) is the size of the random sample needed to estimate a homography.
# Plugging in:

p = 0.95
w = 103/404
s = 4

N = np.log(1 - p) / np.log(1 - w**s)
print(np.ceil(N))   # → 708
# So you need at least 708 iterations to be 95% sure you’ll pick an all-inlier sample once.



# You need the following minimum number of point‐correspondences per RANSAC sample:

# Fundamental matrix
# • 8 points for the (normalized) 8-point algorithm
# • (or 7 if you use the specialized 7-point solver)

# Homography
# • 4 points

# Essential matrix
# • 5 points (the minimal “5-point” algorithm)

# So in your RANSAC loops you’d sample s=8 for F, s=4 for H, and s=5 for E.