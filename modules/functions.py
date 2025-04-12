import itertools as it

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from cv2.typing import MatLike
from scipy.spatial.transform import Rotation

f = 700
a = 1
b = 0
dx = 600
dy = 400


def box3d(n: int = 16) -> np.ndarray:
    """
    Generates a set of 3D points representing a box structure.

    Parameters:
        n (int): Number of points per edge.

    Returns:
        np.ndarray: 3D points of shape (3, N), where N is the number of points.
    """
    points = []
    N = tuple(np.linspace(-1, 1, n))
    for i, j in [(-1, -1), (-1, 1), (1, 1), (0, 0)]:
        points.extend(set(it.permutations([(i,) * n, (j,) * n, N])))
    return np.hstack(points) / 2


def PiInv(nparray: np.ndarray) -> np.ndarray:
    """
    Converts points into Homogeneous coordinates.

    Parameters:
        nparray (np.ndarray): Input array of shape (D, N) where D is the dimension.

    Returns:
        np.ndarray: Homogeneous coordinate array of shape (D+1, N).
    """
    ones = np.ones(
        (1, nparray.shape[1])
    )  # make ones array with dim 1 * N, where N is the ammount of points in the array
    # They are column vectors, meaining they have all x's in one array and all y's in another
    return np.vstack((nparray, ones))


def Pi(nparray: np.ndarray) -> np.ndarray:
    """
    Converts points from Homogeneous coordinates to Inhomogeneous.

    Parameters:
        nparray (np.ndarray): Input homogeneous coordinate array of shape (D+1, N).

    Returns:
        np.ndarray: Inhomogeneous coordinate array of shape (D, N).
    """

    """Converts points from Homogeneous coordinates to Inhomogeneous"""
    return (
        nparray[:-1] / nparray[-1]
    )  # Convert from homogeneous to inhomogeneous coordinates


def projectpoints_with_dist(
    K: np.ndarray = np.array([[f, b * f, dx], [0, a * f, dy], [0, 0, 1]]),
    R: np.ndarray = np.array(
        [
            [np.sqrt(0.5), -np.sqrt(0.5), 0.0],
            [np.sqrt(0.5), np.sqrt(0.5), 0.0],
            [0.0, 0.0, 1],
        ]
    ),
    t: np.ndarray = np.array([[0.0], [0.0], [10.0]]),
    Q: np.ndarray = box3d(),
    dist: list[float] = [-0.245031, 0.071524, -0.00994978],
) -> np.ndarray:
    """
    Projects 3D points onto a 2D image plane with radial distortion.

    Args:
        K (np.ndarray): Camera intrinsic matrix.
        R (np.ndarray): Rotation matrix.
        t (np.ndarray): Translation vector.
        Q (np.ndarray): 3D points of shape (3, N).
        dist (list[float]): Radial distortion coefficients.

    Returns:
        np.ndarray: Projected 2D points in homogeneous coordinates.
    """
    # Point = K*Rt*Q
    P = np.concatenate((R, t), axis=1)

    # Q which is the box3d, is not in homogeneous coordinates, which produces an error
    # So we convert Q into Q_h
    # Q_h = np.vstack((Q, np.ones((1, Q.shape[1]))))
    Q_h = PiInv(Q)

    points = P @ Q_h

    # Make inhomogeneous
    points = Pi(points)
    x, y = points[0], points[1]
    r2 = x * x + y * y
    dr = 0
    dr = [(j * (np.pow(r2, i))) for i, j in enumerate(dist, start=1)]
    radial_factor = sum(dr) + 1

    x_distorted = x * radial_factor
    y_distorted = y * radial_factor

    # make the projection homogeneous
    Qd = PiInv(np.vstack((x_distorted, y_distorted)))

    return K @ Qd


def project_points_with_Proj_matrix(
    P: np.ndarray = np.array(
        [[700.0, 0.0, 600.0, 600.0], [0, 700.0, 400.0, 400.0], [0.0, 0.0, 0.1, 0.1]]
    ),
    Q=box3d(),
    dist: list[float] = 0,
) -> np.ndarray:
    """
    Projects 3D points onto a 2D image plane using a projection matrix.

    Args:
        P (np.ndarray): Projection matrix of shape (3, 4).
        Q (np.ndarray): 3D points of shape (3, N).
        dist (float): Distortion parameter (currently unused).

    Returns:
        np.ndarray: Projected 2D points in inhomogeneous coordinates.
    """
    return Pi(P @ Q)


def plot_3d(dpoints: np.ndarray, i: int = 0) -> None:
    """
    Plots 3D points and their 2D projection, with the option to highlight a specific point.

    Args:
        dpoints (np.ndarray): A 3xN array of homogeneous coordinates (3D points).
        i (int, optional): Index of the point to highlight. Default is 0.

    Returns:
        None: Displays the 3D and 2D scatter plots.
    """
    # Normalize projected points (Convert from homogeneous to Cartesian)
    # dpoints /= dpoints[2, :]  # Divide x, y by z to get 2D points

    # Extract 2D points for visualization
    x_proj, y_proj = dpoints[0], dpoints[1]

    x, y, z = box3d()[0], box3d()[1], box3d()[2]

    # Select the index of the point to highlight
    highlight_idx = i  # Change this index to highlight a different point

    # Plot in 3D
    fig = plt.figure(figsize=(10, 5))

    # 3D Plot of original points
    ax = fig.add_subplot(121, projection="3d")
    ax.scatter(x, y, z, c="b", marker="o")
    # Highlight specific point
    ax.scatter(
        x[highlight_idx],
        y[highlight_idx],
        z[highlight_idx],
        c="r",
        marker="o",
        s=100,
        label="Highlighted Point",
    )
    # ax.scatter(x[highlight_idx+16], y[highlight_idx+16], z[highlight_idx+16], c='g', marker='o', s=100, label='Highlighted Point 2')
    # ax.scatter(x[highlight_idx+32], y[highlight_idx+32], z[highlight_idx+32], c='gold', marker='o', s=100, label='Highlighted Point 3')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Original 3D Points")
    ax.legend()

    # 2D Projection Plot
    ax2 = fig.add_subplot(122)
    ax2.scatter(x_proj, y_proj, c="r", marker="x")
    ax2.scatter(
        x_proj[highlight_idx],
        y_proj[highlight_idx],
        c="b",
        s=100,
        label="Highlighted Point",
    )
    # ax2.scatter(x_proj[highlight_idx+16], y_proj[highlight_idx+16], c='g', s=100, label="Highlighted Point 3")
    # ax2.scatter(x_proj[highlight_idx+32], y_proj[highlight_idx+32], c='gold', s=100, label="Highlighted Point 3")
    ax2.set_xlabel("Image X")
    ax2.set_ylabel("Image Y")
    plt.legend()
    plt.gca().invert_yaxis()  # Invert y-axis for correct image representation
    plt.grid()
    ax2.set_title("Projected 2D Points")

    plt.show()


def undistortImage(
    im: np.ndarray,
    K: np.ndarray = np.array([[f, b * f, dx], [0, a * f, dy], [0, 0, 1]]),
    distCoeff: list = [-0.245031, 0.071524, -0.00994978],
) -> np.ndarray:
    """
    Undistorts an image by applying the camera matrix and distortion coefficients.

    Args:
        im (np.ndarray): The input image to be undistorted.
        K (np.ndarray, optional): Camera matrix. Default is a 3x3 matrix based on focal lengths and principal point.
        distCoeff (list, optional): List of distortion coefficients. Default includes radial distortion terms.

    Returns:
        np.ndarray: The undistorted image.
    """
    # Creates a mesh grid
    x, y = np.meshgrid(np.arange(im.shape[1]), np.arange(im.shape[0]))

    # Creates homogeneous coordinates
    p = PiInv(np.stack((x.ravel(), y.ravel())))

    # Normalize pixel coords with inverse of K
    q = np.linalg.inv(K) @ p
    q_h = PiInv(q[:-1])
    # compute radial dist
    q_d = projectpoints_with_dist(Q=q_h)

    x_d = q_d[0].reshape(x.shape).astype(np.float32)
    y_d = q_d[1].reshape(y.shape).astype(np.float32)

    assert (q_d[2] == 1).all(), "You did a mistake somewhere"

    imundist = cv2.remap(im, x_d, y_d, cv2.INTER_LINEAR)

    return imundist


def homography(
    pointlist: list = [np.array([[1.0], [1.0]])],
    H: np.ndarray = np.array([[-2, 0, 1], [0, -2, 0], [0, 0, 3]]),
) -> np.ndarray:
    """
    Applies a homography transformation to a list of points using a given transformation matrix.

    Args:
        pointlist (list, optional): List of 2D points in homogeneous coordinates to be transformed.
        H (np.ndarray, optional): 3x3 homography matrix for transformation. Default is a predefined matrix.

    Returns:
        np.ndarray: Transformed points after applying the homography.
    """
    homlist = [PiInv(i) for i in pointlist]
    return [Pi(H @ i) for i in homlist]


def hest(
    pointlist: list = [
        np.array([[1.0], [1.0]]),
        np.array([[0.0], [3.0]]),
        np.array([[2.0], [3.0]]),
        np.array([[2.0], [4.0]]),
    ],
    dest: list = [
        np.array([[-0.33333333], [-0.66666667]]),
        np.array([[-0.33333333], [-2.0]]),
        np.array([[-1.0], [-2.0]]),
        np.array([[-1.0], [-2.66666667]]),
    ],
    normalize: bool = False,
) -> np.ndarray:
    """
    Estimates the homography matrix using the linear DLT algorithm.

    Args:
        pointlist (list, optional): List of source points in homogeneous coordinates. Default includes four points.
        dest (list, optional): List of destination points in homogeneous coordinates. Default includes four points.
        normalize (bool, optional): Whether to normalize the points before computing the homography. Default is False.

    Returns:
        np.ndarray: The 3x3 estimated homography matrix.
    """
    # assert len(pointlist) == 4 and len(dest) == 4

    # Modify how points are extracted from the lists
    if normalize:
        T1, src_pts = normalize2d(pointlist)
        T2, dst_pts = normalize2d(dest)
    else:
        src_pts = [p.flatten() for p in pointlist]
        dst_pts = [p.flatten() for p in dest]

    B = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        B.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        B.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])

    B = np.array(B)

    # Compute SVD
    U, S, VT = np.linalg.svd(B)
    h = VT[-1, :]  # Smallest singular vector (last row of VT)

    # Reshape into 3x3 homography matrix
    H = h.reshape(3, 3)

    if normalize:
        # Apply the normalization matrices to the estimated homography
        H = np.linalg.inv(T2) @ H @ T1

    return H


def normalize2d(points: list) -> tuple:
    """
    Normalizes a set of 2D points so that their mean is (0,0) and standard deviation is (1,1).

    Args:
        points (list): List of 2D points as np.arrays (column vectors [[x], [y]]).

    Returns:
        tuple: A tuple containing the normalization matrix T and the normalized points.
    """
    # Convert list of column vectors to (N,2) array
    points = np.array([p.flatten() for p in points])

    # Compute mean and standard deviation
    mean = np.mean(points, axis=0)
    std_dev = np.std(points, axis=0)

    # Avoid division by zero in case of a degenerate set of points
    std_dev[std_dev == 0] = 1

    # Construct the normalization matrix T
    T = np.array(
        [
            [1 / std_dev[0], 0, -mean[0] / std_dev[0]],
            [0, 1 / std_dev[1], -mean[1] / std_dev[1]],
            [0, 0, 1],
        ]
    )

    # Convert points to homogeneous coordinates
    points_hom = PiInv(points.T).T  # (N, 3)

    # Apply transformation
    normalized_points_hom = (T @ points_hom.T).T  # Transform and convert back

    # Return normalized points in original shape
    normalized_points = Pi(normalized_points_hom.T).T

    return T, normalized_points


def CrossOp(dvector: np.ndarray) -> np.ndarray:
    """
    Computes the cross-product matrix of a vector for use in matrix multiplication.

    Args:
        dvector (np.ndarray): A 3D vector as a column vector [[x], [y], [z]].

    Returns:
        np.ndarray: A 3x3 cross-product matrix.
    """
    x, y, z = dvector.squeeze()
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def fundamentalmatrix(
    Q: np.ndarray = np.array([[1.0], [0.5], [4.0], [1.0]]),
    K: np.ndarray = np.array([[f, b * f, dx], [0, a * f, dy], [0, 0, 1]]),
    R1: np.ndarray = np.eye(3),
    t1: np.ndarray = np.array([[0.0], [0.0], [0.0]]),
    t2: np.ndarray = np.array([[0.2], [2.0], [1.0]]),
    R2: np.ndarray = Rotation.from_euler("xyz", [0.7, -0.5, 0.8]).as_matrix(),
) -> np.ndarray:
    """
    Computes the fundamental matrix from two camera positions and a set of 3D points.

    Args:
        Q (np.ndarray, optional): A set of 3D points in homogeneous coordinates. Default is a predefined array.
        K (np.ndarray, optional): Camera matrix. Default includes focal lengths and principal point.
        R1 (np.ndarray, optional): Rotation matrix for the first camera. Default is the identity matrix.
        t1 (np.ndarray, optional): Translation vector for the first camera. Default is [0, 0, 0].
        t2 (np.ndarray, optional): Translation vector for the second camera. Default is [0.2, 2.0, 1.0].
        R2 (np.ndarray, optional): Rotation matrix for the second camera. Default is based on specified Euler angles.

    Returns:
        np.ndarray: The computed 3x3 fundamental matrix.
    """
    # hyperparameters
    f = 1000
    a = 1
    b = 0
    dx = 300
    dy = 200

    # calculate relative transformations
    # relative rotation from cam1 to cam2
    RR = R2 @ (R1.T)
    # relative translation from cam1 to cam2
    TT = t2 - RR @ t1
    # Essential matrix
    E = CrossOp(TT) @ RR
    # fundamental matrix
    F = np.linalg.inv(K.T) @ E @ np.linalg.inv(K)
    return F


def point_line_distance(L: np.ndarray, p2: np.ndarray) -> float:
    """
    Computes the perpendicular distance from a point to a line in homogeneous coordinates.

    Parameters:
    - L: Epipolar line coefficients [a, b, c] (1D array or list of length 3).
    - p2: Point (x2, y2) in the second image.

    Returns:
    - Distance (float).
    """
    a, b, c = L
    x2, y2 = p2
    return abs(a * x2 + b * y2 + c) / np.sqrt(a**2 + b**2)


def triangulate(pointlist: np.ndarray, Projlist: np.ndarray) -> np.ndarray:
    """
    Triangulates 3D points from multiple camera projections using broadcasting.

    Args:
        pointlist (np.ndarray): Array of 2D points from different camera views, with shape (n, 2, 1).
        Projlist (np.ndarray): Array of 3x4 camera projection matrices, with shape (n, 3, 4).

    Returns:
        np.ndarray: The triangulated 3D points in homogeneous coordinates, with shape (n, 3).
    """

    # Ensure correct input shapes
    # Each point should have shape (n, 2, 1)
    assert pointlist.shape[1:] == (2, 1)
    assert Projlist.shape[1:] == (3, 4)  # Each projection matrix is (3, 4)

    # Extract x, y from the pointlist (broadcastable shape)
    x = pointlist[:, 0:1, :]  # Shape (n, 1, 1)
    y = pointlist[:, 1:2, :]  # Shape (n, 1, 1)
    n = x.shape[0]
    # Extract rows of projection matrices
    P0 = Projlist[:, 0:1, :]  # Shape (n, 1, 4)
    P1 = Projlist[:, 1:2, :]  # Shape (n, 1, 4)
    P2 = Projlist[:, 2:3, :]  # Shape (n, 1, 4)

    # Construct A using broadcasting
    A = np.concatenate([x * P2 - P0, y * P2 - P1], axis=1)  # Shape (n, 2, 4)

    # Reshape A into a batch form for SVD
    A = A.reshape(n * 2, 4)  # Shape (2n, 4), ensuring it's valid for SVD

    # Solve AX = 0 using SVD
    _, _, Vt = np.linalg.svd(A)

    # Extract last row of Vt (smallest singular value vector)
    X_homogeneous = Vt[-1]  # Shape (4, n)

    # Convert from homogeneous to inhomogeneous coordinates
    X = X_homogeneous.T  # Shape (n, 3)

    return X


def calc_projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Calculates the camera projection matrix from the camera matrix, rotation matrix, and translation vector.

    Args:
        K (np.ndarray): Camera intrinsic matrix (3x3).
        R (np.ndarray): Rotation matrix (3x3).
        t (np.ndarray): Translation vector (3x1).

    Returns:
        np.ndarray: The resulting 3x4 camera projection matrix.
    """
    return K @ (np.concat((R, t), axis=1))


def pest(Q: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Computes the projection matrix from 3D points (Q) and their corresponding 2D projections (q) using the Direct linear Transforms method.

    Args:
        Q (np.ndarray): Array of 3D points in homogeneous coordinates, with shape (n, 4).
        q (np.ndarray): Array of 2D points in homogeneous coordinates, with shape (n, 2).

    Returns:
        np.ndarray: The estimated 3x4 projection matrix.
    """
    B = np.array([])
    for i in q.T:
        B = np.vstack(np.kron(Q, CrossOp(i)))
    # Compute SVD
    _, _, VT = np.linalg.svd(B)
    h = VT[-1, :]  # Smallest singular vector (last row of VT) 12, 1
    # Reshape into 3x4 projection matrix
    return h.reshape(3, 4)


def get_manual_checkerboard_points(n: int, m: int) -> np.ndarray:
    """
    Generates a 3D point set representing a checkerboard pattern in the z = 0 plane.

    Each point Q_ij is defined as:
        Q_ij = [
            [i - (n - 1) / 2],
            [j - (m - 1) / 2],
            [0]
        ]
    where i ∈ {0, ..., n-1} and j ∈ {0, ..., m-1}.

    The function returns a 3 × (n * m) matrix, where:
        - The first row contains the x-coordinates.
        - The second row contains the y-coordinates.
        - The third row contains only zeros (z = 0 plane).

    Parameters:
    -----------
    n : int
        Number of points along the x-axis (rows of the checkerboard).
    m : int
        Number of points along the y-axis (columns of the checkerboard).

    Returns:
    --------
    np.ndarray
        A 3 × (n * m) matrix containing the 3D coordinates of the checkerboard points.

    Example:
    --------
    >>> get_manual_checkerboard_points(3, 3)
    array([[-1.,  0.,  1., -1.,  0.,  1., -1.,  0.,  1.],
           [-1., -1., -1.,  0.,  0.,  0.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    """
    Q = np.zeros((3, n * m))
    index = 0
    for i in range(n):
        for j in range(m):
            Q[:, index] = [i - (n - 1) / 2, j - (m - 1) / 2, 0]
            index += 1

    return Q


def triangulate_nonlin(pointlist: np.ndarray, Projlist: np.ndarray) -> np.ndarray:
    """
    Triangulates 3D points from multiple camera projections using nonlinear optimization.

    Args:
        pointlist (np.ndarray): Array of 2D points from different camera views, with shape (n, 2, 1).
        Projlist (np.ndarray): Array of 3x4 camera projection matrices, with shape (n, 3, 4).

    Returns:
        np.ndarray: The optimized 3D points in homogeneous coordinates, with shape (n, 4).
    """

    def compute_residuals(Q):
        Qh = Q.reshape(4, 1)  # 4,1
        # array = np.array((N,))
        array = []
        for qi, Pii in zip(pointlist, Projlist):  # qi 2,1  Pii 3,4  Pi(Pii@Qh) 2,1
            a = Pi(Pii @ Qh) - qi  # a.shape 2,1
            array = np.hstack((array, a.flatten()))  # make it into 2,
        return array

    x0 = triangulate(pointlist, Projlist).flatten()  # (4,)
    point = scipy.optimize.least_squares(compute_residuals, x0)
    optimized = point.x
    return optimized.reshape(-1, 1)


def get_checkerboard_points(n: int, m: int) -> np.ndarray:
    """
    Generates a 3D point set representing a checkerboard pattern in the z = 0 plane.

    Each point Q_ij is defined as:
        Q_ij = [
            [i - (n - 1) / 2],
            [j - (m - 1) / 2],
            [0]
        ]
    where i ∈ {0, ..., n-1} and j ∈ {0, ..., m-1}.

    The function returns a 3 × (n * m) matrix, where:
        - The first row contains the x-coordinates.
        - The second row contains the y-coordinates.
        - The third row contains only zeros (z = 0 plane).

    Parameters:
    -----------
    n : int
        Number of points along the x-axis (rows of the checkerboard).
    m : int
        Number of points along the y-axis (columns of the checkerboard).

    Returns:
    --------
    np.ndarray
        A 3 × (n * m) matrix containing the 3D coordinates of the checkerboard points.

    Example:
    --------
    >>> get_checkerboard_points(3, 3)
    array([[-1.,  0.,  1., -1.,  0.,  1., -1.,  0.,  1.],
           [-1., -1., -1.,  0.,  0.,  0.,  1.,  1.,  1.],
           [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
    """
    Q = np.zeros((3, n * m))
    index = 0
    for i in range(n):
        for j in range(m):
            Q[:, index] = [i - (n - 1) / 2, j - (m - 1) / 2, 0]
            index += 1

    return Q


def gausian1DKernel(sigma: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a 1D Gaussian kernel and its derivative for a given sigma.

    Args:
        sigma (int): Standard deviation of the Gaussian.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            g  — 1D Gaussian kernel of shape (N, 1), normalized to sum to 1.
            gd — Derivative of the Gaussian kernel with respect to x.
    """
    """To start with, create the function g, gd = gaussian1DKernel(sigma), where:
    g is the 1D Gaussian kernel,
    gd is the derivative of g,
    sigma is the Gaussian width.
    In this function, you have a choice:
    What length should the Gaussian kernel have?
    What is the error in the truncation when setting the length to sigma, 2·sigma, or 6·sigma?
    No matter which length you end up choosing you should normalize g such that it sums to 1."""

    if sigma == 0:
        return np.zeros((0, 0)), np.zeros((0, 0))

    x = np.arange(-sigma, sigma + 1)
    g = (1 / np.sqrt(2 * (sigma) ** 2)) * np.exp((-(x**2)) / (2 * (sigma**2)))
    g /= np.sum(g)
    gd = np.gradient(g)
    return g.reshape(-1, 1), gd.reshape(-1, 1)


def gaussianSmoothing(
    im: MatLike, sigma: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Gaussian smoothing and compute image derivatives.

    Args:
        im (MatLike): Input grayscale image.
        sigma (float): Standard deviation of the Gaussian kernel.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            I  — Smoothed image.
            Ix — Smoothed derivative along x-axis.
            Iy — Smoothed derivative along y-axis.
    """
    """Now create the function I, Ix, Iy = gaussianSmoothing(im, sigma), where I is the Gaussian
    smoothed image of im, and Ix and Iy are the smoothed derivatives of the image im. The im is the
    original image and sigma is the width of the Gaussian.
    Remember to convert the image im to a single channel (greyscale) and floating point, so you are
    able to do the convolutions.
    Using the g, gd = gaussian1DKernel(sigma) function, how would you do 2D smoothing?
    Tip: Using a 1D kernel in one direction e.g. x is independent of kernels in the other directions.
    What happens if sigma = 0? What should the function return if it supported that?
    Use the smoothing function on your test image. Do the resulting images look correct?"""
    g, gd = gausian1DKernel(sigma)
    I = cv2.sepFilter2D(im, -1, g.T, g)
    Ix = cv2.sepFilter2D(im, -1, gd.T, g)
    Iy = cv2.sepFilter2D(im, -1, g.T, gd)
    return I, Ix, Iy


def structureTensor(im, sigma, epsilon):
    """Now create the function C = structureTensor(im, sigma, epsilon) where
    C(x, y) = gϵ *Ix2(x, y) gϵ *Ix(x, y)Iy(x, y)
    gϵ *Ix(x, y)Iy(x, y) gϵ *Iy2(x, y) (1)
    and gϵ *. . . is the convolution of a new Gaussian kernel with width epsilon.
    Use the structure tensor function on your test image. Do the resulting images still look correct?
    We use two Gaussian widths in this function: sigma and epsilon. The first one sigma is used
    to calculate the derivatives and the second one to calculate the structure tensor. Do we need to
    use both? If you don’t know the answer, start on the next exercise and return to this question
    afterwards."""
    I, Ix, Iy = gaussianSmoothing(im, sigma)
    ge, gde = gausian1DKernel(epsilon)
    IxIy = cv2.filter2D(Ix * Iy, -1, ge * ge.T)
    Ix2 = cv2.filter2D(Ix * Ix, -1, ge * ge.T)
    Iy2 = cv2.filter2D(Iy * Iy, -1, ge * ge.T)
    return np.hstack((np.vstack((Ix2, IxIy)), np.vstack((IxIy, Iy2))))


def harrisMeasure(im, sigma, epsilon, k):
    """Create the function r = harrisMeasure(im, sigma, epsilon, k) where
    r(x, y) = a·b − c2 − k(a + b)2
    , where     C(x, y) = a c
                          c b
    Now return to the question from last exercise.
    Tip: What happens to r if you set epsilon = 0? Take a look at Equations 1 to 3. Why is it
    essential that epsilon ̸= 0?
    Use the harrisMeasure function on your test image. Recall that k = 0.06 is a typical choice of k.
    Are there large values near the corners?"""
    I, Ix, Iy = gaussianSmoothing(im, sigma)
    ge, gde = gausian1DKernel(epsilon)
    IxIy = cv2.filter2D(Ix * Iy, -1, ge * ge.T)
    Ix2 = cv2.filter2D(Ix * Ix, -1, ge * ge.T)
    Iy2 = cv2.filter2D(Iy * Iy, -1, ge * ge.T)
    ab = Ix2 * Iy2
    anb = Ix2 + Iy2
    c2 = IxIy * IxIy
    return ab - c2 - k * anb * anb


def cornerDetector(im, sigma, epsilon, k, tau):
    """Finally, create the function c = cornerDetector(im, sigma, epsilon, k, tau) where c is a
    list of points where r is the local maximum and larger than some relative threshold i.e.
    r(x, y) > tau.
    To get local maxima, you should implement non-maximum suppression, see the slides or Sec. 4.3.1
    in the LN. Non-maximum suppression ensures that r(x, y) > r(x ±1, y) and r(x, y) > r(x, y ±1).
    Once you have performed non-maximum suppression you can find the coordinates of the points
    using np.where.
    Use the corner detector on your test image. Does it find all the corners, or too many corners?"""
    r = harrisMeasure(im, sigma, epsilon, k)

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


def scaleSpaced(im: MatLike, sigma: int, n: int) -> list[np.ndarray]:
    """
    Generate a scale-space of Gaussian-blurred images.

    Args:
        im (MatLike): Input grayscale image.
        sigma (float): Base standard deviation of the Gaussian.
        n (int): Number of scales.

    Returns:
        List[np.ndarray]: List of blurred images at different scales.
    """
    images = []
    for i in range(n):
        sigma_i = sigma**i
        blurred, _, _ = gaussianSmoothing(im, sigma_i)
        images.append(blurred)
    return images


def differenceOfGaussians(im: MatLike, sigma: int, n: int) -> list[np.ndarray]:
    """
    Compute Difference of Gaussians (DoG) across a scale-space.

    Args:
        im (MatLike): Input grayscale image.
        sigma (float): Base standard deviation of the Gaussian.
        n (int): Number of scales.

    Returns:
        List[np.ndarray]: List of DoG images.
    """
    images = scaleSpaced(im, sigma, n)
    dog_images = []
    for i in range(1, len(images)):
        dog_images.append(cv2.subtract(images[i], images[i - 1]))
    return dog_images


def detectBlobs(
    im: MatLike, sigma: int, n: int, tau: float
) -> list[tuple[int, int, float]]:
    """
    Detect blobs in an image using Difference of Gaussians and non-maximum suppression.

    Args:
        im (MatLike): Input grayscale image.
        sigma (float): Base standard deviation for scale space.
        n (int): Number of scales.
        tau (float): Threshold for blob detection.

    Returns:
        List[Tuple[int, int, float]]: Detected blobs as (x, y, radius).
    """
    dog = differenceOfGaussians(im, sigma, n)

    def non_maximum_suppression(r):
        # Pad the array with -inf to handle edge cases
        padded = np.pad(r, ((1, 1), (1, 1)), mode="constant", constant_values=-np.inf)
        center = padded[1:-1, 1:-1]

        # Compare each pixel with its left, right, top, and bottom neighbors
        is_max = (
            (center > padded[0:-2, 0:-2])  # top-left
            & (center > padded[0:-2, 1:-1])  # top
            & (center > padded[0:-2, 2:])  # top-right
            & (center > padded[1:-1, 0:-2])  # left
            & (center > padded[1:-1, 2:])  # right
            & (center > padded[2:, 0:-2])  # bottom-left
            & (center > padded[2:, 1:-1])  # bottom
            & (center > padded[2:, 2:])  # bottom-right
        )

        return is_max

    dog_abs = [np.abs(d) for d in dog]

    dog_max = [cv2.dilate(d, np.ones((3, 3))) for d in dog_abs]

    blobs = []
    for i in range(1, len(dog) - 1):
        current = dog_abs[i]
        prev = dog_abs[i - 1]
        next1 = dog_abs[i + 1]

        # get the spacial max
        is_spatial_max = non_maximum_suppression(current)

        # scale nms compare with same pixel in prev and next scale
        is_scale_max = (current > prev) & (current > next1)

        # apply threshold
        cond = is_spatial_max & is_scale_max & (current > tau)

        y, x = np.where(cond)
        sigma_i = sigma**i
        for j in range(len(x)):
            blobs.append((x[j], y[j], sigma_i * np.sqrt(2)))
    return blobs


def ransac_homography(
    pts1: np.ndarray, pts2: np.ndarray, num_iterations: int = 200, sigma: float = 3.0
) -> tuple[np.ndarray, NumPyArrayFloat32]:
    """
    Estimates the best homography matrix between two sets of points using RANSAC.

    Args:
        pts1 (np.ndarray): Source points of shape (N, 2).
        pts2 (np.ndarray): Destination points of shape (N, 2).
        num_iterations (int, optional): Number of RANSAC iterations. Defaults to 200.
        sigma (float, optional): Standard deviation for inlier thresholding. Defaults to 3.0.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - best_H (np.ndarray): Estimated 3x3 homography matrix.
            - best_inliers (np.ndarray): Indices of inlier points used for the final model.
    """
    best_inliers = []
    best_H = None

    threshold = 5.99 * sigma**2  # chi-squared threshold for 2 DOF (approx)

    for i in range(num_iterations):
        # 1. Randomly select 4 points
        idx = random.sample(range(len(pts1)), 4)
        src_pts = pts1[idx]
        dst_pts = pts2[idx]

        # 2. Compute Homography
        H, _ = cv2.findHomography(
            src_pts, dst_pts, method=0
        )  # 0 means no RANSAC inside

        if H is None:
            continue

        # 3. Transform all pts1 using H
        pts1_homog = np.hstack(
            [pts1, np.ones((pts1.shape[0], 1))]
        )  # make homogeneous coords
        pts2_proj = (H @ pts1_homog.T).T

        # Normalize projected points
        pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2, np.newaxis]

        # 4. Compute error
        errors = np.sum((pts2_proj - pts2) ** 2, axis=1)  # squared distance

        # 5. Determine inliers
        inliers = np.where(errors < threshold)[0]

        # 6. Keep best model
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H

    return (best_H, best_inliers)


def warpImage(
    im: np.ndarray, H: np.ndarray, xRange: tuple[int, int], yRange: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Warps an input image into a new coordinate frame defined by a homography and output ranges.

    Args:
        im (np.ndarray): Input image.
        H (np.ndarray): 3x3 homography matrix.
        xRange (tuple[int, int]): Output x-coordinate range (min_x, max_x).
        yRange (tuple[int, int]): Output y-coordinate range (min_y, max_y).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - imWarp (np.ndarray): Warped image.
            - maskWarp (np.ndarray): Warped binary mask indicating valid pixels.
    """
    T = np.eye(3)
    T[:2, 2] = [-xRange[0], -yRange[0]]
    H = T @ H
    outSize = (xRange[1] - xRange[0], yRange[1] - yRange[0])
    mask = np.ones(im.shape[:2], dtype=np.uint8) * 255
    imWarp = cv2.warpPerspective(im, H, outSize)
    maskWarp = cv2.warpPerspective(mask, H, outSize)
    return imWarp, maskWarp
