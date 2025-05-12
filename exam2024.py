import cv2
import numpy as np

from function_folder.function_file import (
    PiInv,
    calc_projection_matrix,
    hest_with_numpy,
    project_points_with_Proj_matrix,
    projectpoints_with_dist,
    ransac_homography_with_copilot,
    undistortImage,
    warpImage,
)

# QUestion 1
R = cv2.Rodrigues(np.array([0.2, 0.2, -0.1]))[0]
t = np.array([-0.08, 0.01, 0.03]).reshape(-1, 1)
Q3d = np.array([-0.38, 0.1, 1.32]).reshape(-1, 1)

K = np.array([[1400, 0, 750], [0, 1400, 520], [0, 0, 1]])

P = calc_projection_matrix(K=K, R=R, t=t)
point = project_points_with_Proj_matrix(P=P, Q=PiInv(Q3d))
print(point)

# Question2
p1 = np.array(
    [
        [1.45349587e02, -1.12915131e-01, 1.91640565e00, -6.08129962e-01],
        [1.05603820e02, 5.62792554e-02, 1.79040110e00, -2.32182177e-01],
    ]
)
p1hom = PiInv(p1)
p2 = np.array(
    [
        [1.3753556, -1.77072961, 2.94511795, 0.04032374],
        [0.30936653, 0.37172814, 1.44007577, -0.03173825],
    ]
)
p2hom = PiInv(p2)


H = hest_with_numpy(p2, p1)
# H = hest_stolen(p1, p2)
H = H / H[0][0]
print(H)


# QUestion3
# what is the minimum number of point correspondences you need to estimate a Fundamental matrix?
# answer 7

# question 4
K = np.array([[300, 0, 840], [0, 300, 620], [0, 0, 1]], float)
dist = [0.2, 0.01, -0.03]

pointproj = undistortImage(np.ones((800, 1000)), K, dist)
points = projectpoints_with_dist(
    K, np.eye(3), np.zeros((3, 1)), PiInv(np.array([[742.8], [593.5]])), dist
)
print(pointproj.shape)
print(points)


# question9
sift_data = np.load("./materials2024/sift_data.npy", allow_pickle=True).item()
kp1 = sift_data["kp1"]
des1 = sift_data["des1"]
kp2 = sift_data["kp2"]
des2 = sift_data["des2"]

def rootsift(descriptors):
    """
    Applies the RootSIFT normalization to SIFT descriptors
    
    Args:
        descriptors: numpy array of SIFT descriptors
        
    Returns:
        Root-SIFT normalized descriptors
    """
    # L1 normalize
    descriptors = descriptors / (np.sum(descriptors, axis=1, keepdims=True) + 1e-7)
    
    # Square root (Hellinger kernel)
    descriptors = np.sqrt(descriptors)
    
    return descriptors


# asekd to match sift features, use brute force matcher from cv2
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
print(f"Number of matches: {len(matches)}")
print(matches[0].distance)  # distance of the best match
print(matches[0].queryIdx)  # index of the best match in im1
print(matches[0].trainIdx)  # index of the best match in im2


bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)  # Must set crossCheck to False for knnMatch
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
ratio = 0.8  # 0.7-0.8 is typical
for m, n in matches:
    if m.distance < ratio * n.distance:
        good_matches.append(m)

print(f"Number of matches after ratio test: {len(good_matches)}")


#question 10

# camera calibration
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt

def calibrate_camera(image_paths, checkerboard_size=(7, 10)):
    """
    Calibrate camera from checkerboard images.
    
    Args:
        image_paths: List of paths to images containing checkerboard patterns
        checkerboard_size: Size of the checkerboard (internal corners)
        
    Returns:
        ret: RMS reprojection error
        mtx: Camera matrix containing focal lengths (fx, fy) and optical centers (cx, cy)
        dist: Distortion coefficients
        rvecs: Rotation vectors for each image
        tvecs: Translation vectors for each image
    """
    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ... (8,5,0)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Process each image separately before attempting to create the visualization
    successful_images = []
    
    for fname in image_paths:
        # Read image
        img = cv2.imread(fname)
        if img is None:
            print(f"Warning: Could not read image {fname}")
            continue
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find the chessboard corners
        # Let's try with different flags to improve detection
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)
        
        # If found, add object points, image points
        if ret:
            # Refine corner positions for better accuracy
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            successful_images.append((img, fname))
            print(f"Successfully detected checkerboard in {fname}")
        else:
            print(f"Could not find checkerboard pattern in {fname}")
    
    # Check if we have any successful detections
    if not objpoints:
        raise ValueError("No checkerboard patterns found in any of the images. Check the file paths and checkerboard_size parameter.")
    
    # Now create visualization only for successful images
    fig, axes = plt.subplots(1, len(successful_images), figsize=(15, 5))
    if len(successful_images) == 1:
        axes = [axes]
        
    for i, (img, fname) in enumerate(successful_images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        cv2.drawChessboardCorners(img, checkerboard_size, corners, ret)
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f'Image {i+1}')
        axes[i].axis('off')
    
    # plt.tight_layout()
    # plt.show()
    
    # Calibrate camera using the successful images
    if gray is not None:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        return ret, mtx, dist, rvecs, tvecs
    else:
        raise ValueError("Failed to process any images")    
        
# Define paths to the checkerboard images
image_paths = sorted(glob.glob('./materials2024/board*.jpg'))

# Run calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(image_paths)

# Print results
print(f"Camera Matrix:\n{camera_matrix}")
print(f"Focal length (fx, fy): ({camera_matrix[0,0]:.2f}, {camera_matrix[1,1]:.2f})")
print(f"Principal point (cx, cy): ({camera_matrix[0,2]:.2f}, {camera_matrix[1,2]:.2f})")
print(f"Distortion coefficients: {dist_coeffs.ravel()}")
print(f"Reprojection error: {ret:.6f}")

# Additional visualization: undistort an example image
img = cv2.imread(image_paths[0])
h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1, (w,h))

# Undistort
dst = cv2.undistort(img, camera_matrix, dist_coeffs, None, newcameramtx)

# Display result
# plt.figure(figsize=(12, 6))
# plt.subplot(121)
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.title('Original Image')
# plt.axis('off')

# plt.subplot(122)
# plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
# plt.title('Undistorted Image')
# plt.axis('off')
# plt.tight_layout()
# plt.show()


#question 12# 1. Load images
im1 = cv2.imread("./materials2024/im1.jpg")
im2 = cv2.imread("./materials2024/im2.jpg")
im1g = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)

im2g = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)


# SIFT
sift1 = cv2.SIFT.create()
sift2 = cv2.SIFT.create()

# kp1, kp2 = keypoints (interesting points in the images, like corners)
# des1, des2 = descriptors (vector that describes the area around each keypoint)

kp1, des1 = sift1.detectAndCompute(im1g, None)
kp2, des2 = sift2.detectAndCompute(im2g, None)

# kp1[0].pt coords of the first keypoint
print(f"Number of keypoints in im1: {len(kp1)}")
print(f"Number of keypoints in im2: {len(kp2)}")

# asekd to match sift features, use brute force matcher from cv2
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)
print(f"Number of matches: {len(matches)}")
print(matches[0].distance)  # distance of the best match
print(matches[0].queryIdx)  # index of the best match in im1
print(matches[0].trainIdx)  # index of the best match in im2

pts1 = np.array([kp1[m.queryIdx].pt for m in matches]).astype(np.float32)
pts2 = np.array([kp2[m.trainIdx].pt for m in matches]).astype(np.float32)

# RANSAC for homography
H, best_inliers = ransac_homography_with_copilot(pts1, pts2)
print("Estimated Homography:")
print(H/H[0][0])  # Normalize the homography matrix
A = np.linalg.inv(H)
print(f"Estimated Homography inverted: {A/A[0][0]}")

print(f"Found {len(best_inliers)} inliers!")
# best_matches = [matches[i] for i in best_inliers]
# img_inliers = cv2.drawMatches(
#     im1,
#     kp1,
#     im2,
#     kp2,
#     best_matches[:50],
#     None,
#     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
# )
# plt.figure(figsize=(15, 10))
# plt.imshow(img_inliers)
# plt.show()

# #  Warp im1 into the coordinate system of im2
# xRange = (0, im1.shape[1] + im2.shape[1])  # Make it big enough for both images
# yRange = (0, max(im1.shape[0], im2.shape[0]))

# im1Warped, mask1Warped = warpImage(im1, H, xRange, yRange)

# #  Warp im2 using identity matrix (no change)
# identity_H = np.eye(3)
# im2Warped, mask2Warped = warpImage(im2, identity_H, xRange, yRange)
# print(mask1Warped)

# question 15


# Camera parameters
K = np.array([[300, 0, 840], [0, 300, 620.0], [0, 0, 1]], float)
R1 = cv2.Rodrigues(np.array([-2.3, -0.7, 1.0]))[0]
t1 = np.array([0.0, -1.0, 4.0], float)
R2 = cv2.Rodrigues(np.array([-0.6, 0.5, -0.9]))[0]
t2 = np.array([0.0, 0.0, 9.0], float)
R3 = cv2.Rodrigues(np.array([-0.1, 0.9, -1.2]))[0]
t3 = np.array([-1.0, -6.0, 28.0], float)

# Image points
p1 = np.array([853.0, 656.0])
p2 = np.array([814.0, 655.0])
p3 = np.array([798.0, 535.0])

# Step 1: Calculate the fundamental matrix F12
# First, get the projection matrices for both cameras
P1 = K @ np.hstack((R1, t1.reshape(-1, 1)))
P2 = K @ np.hstack((R2, t2.reshape(-1, 1)))

# Calculate the camera centers in world coordinates
# For camera 1
C1 = -np.linalg.inv(R1) @ t1
# For camera 2
C2 = -np.linalg.inv(R2) @ t2

# Calculate the essential matrix
R_rel = R2 @ R1.T
t_rel = t2 - R_rel @ t1
t_rel_cross = np.array([
    [0, -t_rel[2], t_rel[1]],
    [t_rel[2], 0, -t_rel[0]],
    [-t_rel[1], t_rel[0], 0]
])
E = t_rel_cross @ R_rel

# Calculate the fundamental matrix
F12 = np.linalg.inv(K).T @ E @ np.linalg.inv(K)

# Step 2: Find the epipolar line in camera 1 corresponding to p2
p2_homogeneous = np.append(p2, 1)
epipolar_line = F12.T @ p2_homogeneous

# Step 3: Calculate the distance from p1 to this epipolar line
# The formula for the distance from a point (x, y) to a line a*x + b*y + c = 0 is:
# distance = |a*x + b*y + c| / sqrt(a^2 + b^2)
a, b, c = epipolar_line
p1_x, p1_y = p1
distance = np.abs(a * p1_x + b * p1_y + c) / np.sqrt(a**2 + b**2)

print(f"Epipolar line in camera 1: {a:.6f}x + {b:.6f}y + {c:.6f} = 0")
print(f"Distance from p1 to epipolar line: {distance:.6f} pixels")

#question 16

# Camera parameters (given in the problem)
K = np.array([[300, 0, 840], [0, 300, 620.0], [0, 0, 1]], float)
R1 = cv2.Rodrigues(np.array([-2.3, -0.7, 1.0]))[0]
t1 = np.array([0.0, -1.0, 4.0], float)
R2 = cv2.Rodrigues(np.array([-0.6, 0.5, -0.9]))[0]
t2 = np.array([0.0, 0.0, 9.0], float)
R3 = cv2.Rodrigues(np.array([-0.1, 0.9, -1.2]))[0]
t3 = np.array([-1.0, -6.0, 28.0], float)

# Image points
p1 = np.array([853.0, 656.0])
p2 = np.array([814.0, 655.0])
p3 = np.array([798.0, 535.0])

# Create projection matrices for each camera
P1 = K @ np.hstack((R1, t1.reshape(-1, 1)))
P2 = K @ np.hstack((R2, t2.reshape(-1, 1)))
P3 = K @ np.hstack((R3, t3.reshape(-1, 1)))

# Prepare the inputs for triangulation
projection_matrices = np.array([P1, P2, P3])
image_points = np.array([p1, p2, p3]).reshape(-1, 2, 1)

# Use the triangulate_nonlin function to get the optimized 3D point
from scipy import optimize

def compute_reprojection_error(point_3d):
    """Compute the reprojection error for a 3D point across all cameras"""
    point_3d_homogeneous = np.append(point_3d, 1)
    
    total_error = 0
    for i in range(3):  # For each camera
        # Project the 3D point to the image plane
        proj = projection_matrices[i] @ point_3d_homogeneous
        proj_2d = proj[:2] / proj[2]  # Convert to inhomogeneous coordinates
        
        # Calculate the squared reprojection error
        error = np.sum((proj_2d - [p1, p2, p3][i])**2)
        total_error += error
        
    return total_error

# Initial estimate using linear triangulation
def linear_triangulation():
    # Set up the DLT system
    A = np.zeros((6, 4))
    
    for i in range(3):
        x, y = [p1, p2, p3][i]
        P = projection_matrices[i]
        
        A[i*2]   = x * P[2] - P[0]
        A[i*2+1] = y * P[2] - P[1]
    
    # Solve using SVD
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    # Return the point in inhomogeneous coordinates
    return X[:3] / X[3]

# Get initial estimate
initial_point = linear_triangulation()

# Refine using non-linear optimization
result = optimize.minimize(compute_reprojection_error, initial_point)
optimal_point_3d = result.x

print(f"Triangulated 3D point: [{optimal_point_3d[0]:.3f}, {optimal_point_3d[1]:.3f}, {optimal_point_3d[2]:.3f}]")


#question 17

# Camera extrinsic parameters
R = cv2.Rodrigues(np.array([-1.9, 0.1, -0.2]))[0]
t = np.array([-1.7, 1.3, 1.5], float)

# Calculate camera center in world coordinates
# C = -R^T * t
camera_position = -np.linalg.inv(R) @ t

# Since R is a rotation matrix, its inverse equals its transpose
# So the same calculation can be done as:
camera_position_alt = -R.T @ t

print(f"Camera position: ({camera_position[0]:.4f}, {camera_position[1]:.4f}, {camera_position[2]:.4f})")

#question 18
# Given 3D point in camera 3's reference frame
point_camera3 = np.array([[-0.38], [0.1], [1.32]])

# Step 1: Convert from camera 3 coordinates to world coordinates
# X_world = R3.T @ X_camera3 - R3.T @ t3
point_world = R3.T @ point_camera3 - R3.T @ t3.reshape(-1, 1)

# Step 2: Convert from world coordinates to camera 2 coordinates
# X_camera2 = R2 @ X_world + t2
point_camera2 = R2 @ point_world + t2.reshape(-1, 1)

print(f"3D point in camera 2's reference frame: [{point_camera2[0,0]:.4f}, {point_camera2[1,0]:.4f}, {point_camera2[2,0]:.4f}]")


#question 19

# Calculating Required RANSAC Iterations for 90% Confidence
# To calculate the minimum number of iterations required to be 90% confident of finding a model fitted to only inliers in RANSAC, we need to use the standard RANSAC probability formula.

# The formula is: N = log(1-p) / log(1-(1-ε)^s)

# Where:

# N = number of iterations needed
# p = desired probability of success (0.90 in this case)
# ε = outlier ratio in the data
# s = number of samples needed to estimate the model (4 for homography)
# Given information:

# 465 inliers out of 1177 total points
# Outlier ratio ε = (1177-465)/1177 ≈ 0.605 (60.5%)
# s = 4 (we need 4 point correspondences to estimate a homography)
# p = 0.90 (we want 90% confidence)

#question 20
# Determining the Squared Threshold for RANSAC with Known Measurement Noise
# When determining the appropriate squared threshold (τ²) for RANSAC with normally distributed keypoints, we need to use the properties of the chi-squared distribution.

# For keypoint measurement errors that follow a normal distribution with standard deviation σ = 1.4 pixels in both x and y directions:

# The squared Euclidean reprojection error follows a chi-squared distribution with 2 degrees of freedom (one for each dimension x and y)

# For a point to be considered an inlier with 95% confidence, we need to determine the threshold value where: P(error² ≤ τ²) = 0.95

# For a chi-squared distribution with 2 degrees of freedom, the 95th percentile is approximately 5.99

# Since our measurement error has standard deviation σ = 1.4 pixels, we need to scale this value: τ² = 5.99 × σ² = 5.99 × (1.4)² = 5.99 × 1.96 = 11.74

# Therefore, to correctly identify 95% of true inliers with keypoints having standard deviation of 1.4 pixels, you should set your squared threshold τ² to approximately 11.74 square pixels.


#question 21

# Structured Light Surface Reconstruction with Phase Shifting
# To calculate the unwrapped phase angle θ in a pixel from structured light patterns, I'll need to:

# Calculate the wrapped phase for both the primary and secondary patterns
# Use these wrapped phases to determine the unwrapped phase
# Step 1: Calculate wrapped phase angles
# For phase shifting patterns, the wrapped phase can be calculated using:

# Given intensity values
primary = np.array([12, 9, 10, 13, 18, 25, 33, 40, 46, 49, 48, 45, 39, 31, 23, 17])
secondary = np.array([15, 29, 43, 49, 43, 29, 15, 10])

# Calculate wrapped phases using arctangent method (for N-step phase shifting)
def calculate_wrapped_phase(intensities, n_steps):
    # N-step phase shifting formula
    numerator = 0
    denominator = 0
    for i in range(n_steps):
        phase_shift = 2 * np.pi * i / n_steps
        numerator += intensities[i] * np.sin(phase_shift)
        denominator += intensities[i] * np.cos(phase_shift)
    
    # Calculate phase using arctangent
    phase = np.arctan2(-numerator, denominator)
    # Convert to range [0, 2π]
    if phase < 0:
        phase += 2 * np.pi
    return phase

# Get wrapped phases
theta_primary_wrapped = calculate_wrapped_phase(primary, 16)
theta_secondary_wrapped = calculate_wrapped_phase(secondary, 8)

print(f"Primary wrapped phase: {theta_primary_wrapped:.4f} rad")
print(f"Secondary wrapped phase: {theta_secondary_wrapped:.4f} rad")

# Step 2: Unwrap using the two-frequency approach
# With two patterns having different frequencies (40 and 41 periods), we can use them together to unwrap the phase:

# Primary pattern has 40 periods, secondary has 41 periods
primary_periods = 40
secondary_periods = 41

# Calculate the beat frequency
beat_periods = primary_periods * secondary_periods / abs(primary_periods - secondary_periods)
# beat_periods = 40 * 41 / 1 = 1640

# Calculate beat phase = primary - secondary * (primary_periods/secondary_periods)
beat_phase = theta_primary_wrapped - theta_secondary_wrapped * (primary_periods/secondary_periods)
if beat_phase < 0:
    beat_phase += 2 * np.pi

# Calculate the fringe order k
k = round((beat_phase * primary_periods / (2 * np.pi)))

# Calculate unwrapped phase
theta_unwrapped = theta_primary_wrapped + k * 2 * np.pi

print(f"Unwrapped θ: {theta_unwrapped:.4f} rad")
# The unwrapped θ represents the absolute phase in the measurement space,
# which directly correlates to the object's 3D geometry in structured light reconstruction


#question 21 again 
