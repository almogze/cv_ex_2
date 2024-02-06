import cv2
import numpy as np
import harris_corner_detector as hcd
from matplotlib import pyplot as plt


# Implement the following functions:

# Input: two vectors
# Output distance (scalar) between two patches
def SSD(patch_descr_1, patch_descr_2):
    return np.diag((patch_descr_1 - patch_descr_2) @ (patch_descr_1 - patch_descr_2).T)


# Input: two vectors
# Output normalized cross correlation  (scalar) between two patches
def NCC(patch_descr_1, patch_descr_2):
    return np.diag(patch_descr_1 @ patch_descr_2.T) / (
            np.linalg.norm(np.diag(patch_descr_1)) * np.linalg.norm(np.diag(patch_descr_2)))


# Output a descriptor vector
# im is an image, p is a pixel, size is the patch size.
# You can use the histogram function of open cv or numpy
def patch_from_im(im, p, size):
    im_padded = np.pad(im, pad_width=size // 2, mode='constant', constant_values=0)
    p_pad = p + size // 2
    patch = im_padded[p_pad[0] - size // 2:p_pad[0] + size // 2 + 1,
            p_pad[1] - size // 2:p_pad[1] + size // 2 + 1].flatten()
    return patch


# A histogram (30 bins) of the pixels' grey level.
def hist_patch_im(im, p, size):
    histogram = cv2.calcHist([patch_from_im(im, p, size)], [0], None, [30], [0, 255]).flatten()
    return histogram / np.linalg.norm(histogram)


# A vector with the pixels' strength gradient.
def gradient(im, p, size):
    Ix, Iy = hcd.Grad_xy(im)
    I_xy_strength = np.sqrt(Ix ** 2 + Iy ** 2)
    I_xy_strength_pad = np.pad(I_xy_strength, pad_width=size // 2, mode='constant', constant_values=0)
    p_pad = p + size // 2
    grad = I_xy_strength_pad[p_pad[0] - size // 2 - 1:p_pad[0] + size // 2,
           p_pad[1] - size // 2:p_pad[1] + size // 2 + 1].flatten()
    return np.round(grad).astype(np.uint8)


def hist_gradient(im, p, size):
    grad_histogram = cv2.calcHist([gradient(im, p, size)], [0], None, [30], [0, 255]).flatten()
    return grad_histogram / np.linalg.norm(grad_histogram)


def get_top_n_points(matrix, n):
    # Create an array of tuples [value, x_index, y_index]
    points = [(matrix[i, j], i, j) for i in range(matrix.shape[0]) for j in range(matrix.shape[1]) if
              0 < i < matrix.shape[0] - 1 and 0 < j < matrix.shape[1] - 1]

    # Sort the array based on the values in descending order
    sorted_points = sorted(points, key=lambda x: x[0], reverse=True)

    # Get the top n points
    top_n_points = sorted_points[:n]

    return top_n_points


# get n corners from both images
def get_corners(n, image1, image2):
    h_points_matrix1 = hcd.H_corner(image1, sigma_smooth=1, sigma_neighb=1, k=5, th=100, density_size=3, display=1)
    h_points_matrix2 = hcd.H_corner(image2, sigma_smooth=1, sigma_neighb=1, k=5, th=100, density_size=3, display=1)
    corners1 = get_top_n_points(h_points_matrix1, n)
    corners2 = get_top_n_points(h_points_matrix2, n)
    corners1 = np.array([(p[1], p[2]) for p in corners1])
    corners2 = np.array([(p[1], p[2]) for p in corners2])
    return corners1, corners2


def match_patterns(image1, image2, patch_size=50, metric=SSD, patch_descriptor=hist_patch_im, amount_of_edges=200,
                   y_th=None, display=0):
    n, m = image1.shape
    images = cv2.hconcat([image1, image2])
    imC = np.dstack([images, images, images])
    corners1, corners2 = get_corners(amount_of_edges, image1, image2)
    matching_points = []

    for p in corners1:
        # get the corners that have approximately the same y value as the current corner
        relevant_im2_corners = corners2[p[0] - y_th <= corners2[:, 0]] if y_th is not None else corners2
        relevant_im2_corners = relevant_im2_corners[relevant_im2_corners[:, 0] <= p[0] + y_th] if y_th is not None else corners2
        if relevant_im2_corners.size == 0:
            continue
        # get the patch of the current corner and duplicate it to the amount of edges
        patches_1 = np.tile(patch_descriptor(image1, p, patch_size), (len(relevant_im2_corners), 1))
        # get the patches of the second image
        patches_2 = np.array([patch_descriptor(image2, q, patch_size) for q in relevant_im2_corners])
        # calculate the metric between the current patch and all the patches of the second image
        metric_result = np.array(metric(patches_1, patches_2))
        # get the index of the minimum/maximum value for the second image up to a threshold
        index = np.argmin(metric_result) if metric == SSD else np.argmax(metric_result)
        q = relevant_im2_corners[index]
        cv2.line(imC, (p[1], p[0]), (q[1] + m, q[0]), (255, 0, 0), 1)

        matching_points.append((p, q))

    if display:
        plt.imshow(imC)
        plt.show()

    return matching_points


# section b

# img1 = cv2.imread('images/view0.tif', cv2.IMREAD_GRAYSCALE)
# img2 = cv2.imread('images/view6.tif', cv2.IMREAD_GRAYSCALE)

# match_patterns(img1, img2, patch_size=21, metric=SSD, patch_descriptor=hist_patch_im, amount_of_edges=100, y_th=50,
#                display=1)
