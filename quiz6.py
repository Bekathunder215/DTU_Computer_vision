# %%
import cv2
import numpy as np


# %%
def gausian1DKernel(sigma):
    """To start with, create the function g, gd = gaussian1DKernel(sigma), where:
    g is the 1D Gaussian kernel,
    gd is the derivative of g,
    sigma is the Gaussian width.
    In this function, you have a choice:
    What length should the Gaussian kernel have?
    What is the error in the truncation when setting the length to sigma, 2·sigma, or 6·sigma?
    No matter which length you end up choosing you should normalize g such that it sums to 1."""

    x = np.arange(-sigma, sigma + 1)
    g = (1 / np.sqrt(2 * (sigma) ** 2)) * np.exp((-(x**2)) / (2 * (sigma**2)))
    g /= np.sum(g)
    gd = np.gradient(g)
    return g.reshape(-1, 1), gd.reshape(-1, 1)


def gaussianSmoothing(im, sigma):
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
    if type(im) == np.ndarray:
        im_gray = im.astype(float)
    else:
        im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(float)
    I = cv2.sepFilter2D(im_gray, -1, g.T, g)
    Ix = cv2.sepFilter2D(im_gray, -1, gd.T, g)
    Iy = cv2.sepFilter2D(im_gray, -1, g.T, gd)
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
    # r = harrisMeasure(im, sigma, epsilon, k)

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

    r = non_maximum_suppression(im)

    return np.where(r >= tau, r, r >= tau)


cv2.imshow("bleepbloop", cornerDetector(im, sigma, epsilon, k, tau))
cv2.waitKey(0)
# %%
im = np.load("harris.npy", allow_pickle=True).item()
print(im)
print(im.keys())
# %%
sigma = 5
epsilon = 4
k = 0.06
r = (
    im["g*(I_x^2)"] * im["g*(I_y^2)"]
    - im["g*(I_x I_y)"] * im["g*(I_x I_y)"]
    - k * (im["g*(I_x^2)"] + im["g*(I_y^2)"]) ** 2
)
c = cornerDetector(r, sigma, epsilon, k, 516)
y = np.nonzero(c)[0]
x = np.nonzero(c)[1]
points = np.array([x, y])
print(points)

# %%
