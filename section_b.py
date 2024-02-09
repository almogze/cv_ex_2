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


def gradient_opt(I_xy_strength_pad, p, size):
    p_pad = p + size // 2
    grad = I_xy_strength_pad[p_pad[0] - size // 2 - 1:p_pad[0] + size // 2,
           p_pad[1] - size // 2:p_pad[1] + size // 2 + 1].flatten()
    return np.round(grad).astype(np.uint8)


def hist_gradient_opt(I_xy_strength_pad, p, size):
    grad_histogram = cv2.calcHist([gradient_opt(I_xy_strength_pad, p, size)], [0], None, [30], [0, 255]).flatten()
    return grad_histogram / np.linalg.norm(grad_histogram)


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
    h_points_matrix1 = hcd.H_corner(image1, sigma_smooth=1, sigma_neighb=1, k=5, th=100, density_size=3, display=0)
    h_points_matrix2 = hcd.H_corner(image2, sigma_smooth=1, sigma_neighb=1, k=5, th=100, density_size=3, display=0)
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

    for i, p in enumerate(corners1):
        # get the corners that have approximately the same y value as the current corner
        relevant_im2_corners = corners2[p[0] - y_th <= corners2[:, 0]] if y_th is not None else corners2
        relevant_im2_corners = relevant_im2_corners[
            relevant_im2_corners[:, 0] <= p[0] + y_th] if y_th is not None else corners2
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


def match_patterns_with_grad(image1, image2, patch_size=50, metric=SSD, patch_descriptor=hist_gradient_opt, amount_of_edges=200,
                   y_th=None, display=0):
    n, m = image1.shape
    images = cv2.hconcat([image1, image2])
    imC = np.dstack([images, images, images])
    corners1, corners2 = get_corners(amount_of_edges, image1, image2)
    Ix_1, Iy_1 = hcd.Grad_xy(image1)
    I_xy_strength_1 = np.sqrt(Ix_1 ** 2 + Iy_1 ** 2)
    I_xy_strength_pad_1 = np.pad(I_xy_strength_1, pad_width=patch_size // 2, mode='constant', constant_values=0)

    Ix_2, Iy_2 = hcd.Grad_xy(image2)
    I_xy_strength_2 = np.sqrt(Ix_2 ** 2 + Iy_2 ** 2)
    I_xy_strength_pad_2 = np.pad(I_xy_strength_2, pad_width=patch_size // 2, mode='constant', constant_values=0)

    matching_points = []
    mismatching_points = []

    for i, p in enumerate(corners1):
        # get the corners that have approximately the same y value as the current corner
        relevant_im2_corners = corners2[p[0] - y_th <= corners2[:, 0]] if y_th is not None else corners2
        relevant_im2_corners = relevant_im2_corners[
            relevant_im2_corners[:, 0] <= p[0] + y_th] if y_th is not None else corners2
        if relevant_im2_corners.size == 0:
            continue
        # get the patch of the current corner and duplicate it to the amount of edges
        patches_1 = np.tile(patch_descriptor(I_xy_strength_pad_1, p, patch_size), (len(relevant_im2_corners), 1))
        # get the patches of the second image
        patches_2 = np.array([patch_descriptor(I_xy_strength_pad_2, q, patch_size) for q in relevant_im2_corners])
        # calculate the metric between the current patch and all the patches of the second image
        metric_result = np.array(metric(patches_1, patches_2))
        # get the index of the minimum/maximum value for the second image up to a threshold
        first_index = np.argmin(metric_result) if metric == SSD else np.argmax(metric_result)
        first_match = metric_result[first_index]
        metric_result[first_index] = np.iinfo(np.int32).max if metric == SSD else -np.iinfo(np.int32).max
        second_index = np.argmin(metric_result) if metric == SSD else np.argmax(metric_result)
        second_match = metric_result[second_index]
        q = relevant_im2_corners[first_index]

        # check if the ratio between the best and the second best match is greater than 0.8
        if first_match / second_match > 0.8:
            mismatching_points.append((p, q))
            continue

        matching_points.append((p, q))

    if display:
        plt.imshow(imC)
        plt.show()

    return matching_points, mismatching_points


# section b

img1 = cv2.imread('images/view0.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('images/view6.tif', cv2.IMREAD_GRAYSCALE)

# match_patterns(img1, img2, patch_size=21, metric=SSD, patch_descriptor=hist_patch_im, amount_of_edges=500, y_th=None,
#                display=1)


# Study the ratio for matching between the best and the second best match
# You can use NCC or SSD on the desriptor of your choice.
# You can use without (2) or with (3) the y-coordinate constraint
# Present examples that demonstrate the effectiveness of using the ratio.

def match_patterns_with_ratio(image1, image2, patch_size=50, metric=SSD, patch_descriptor=hist_patch_im,
                              amount_of_edges=200,
                              y_th=None, display=0):
    n, m = image1.shape
    images = cv2.hconcat([image1, image2])
    imC = np.dstack([images, images, images])
    corners1, corners2 = get_corners(amount_of_edges, image1, image2)
    matching_points = []
    mismatching_points = []

    for p in corners1:
        # get the corners that have approximately the same y value as the current corner
        relevant_im2_corners = corners2[p[0] - y_th <= corners2[:, 0]] if y_th is not None else corners2
        relevant_im2_corners = relevant_im2_corners[
            relevant_im2_corners[:, 0] <= p[0] + y_th] if y_th is not None else corners2
        if relevant_im2_corners.size == 0:
            continue
        # get the patch of the current corner and duplicate it to the amount of edges
        patches_1 = np.tile(patch_descriptor(image1, p, patch_size), (len(relevant_im2_corners), 1))
        # get the patches of the second image
        patches_2 = np.array([patch_descriptor(image2, q, patch_size) for q in relevant_im2_corners])
        # calculate the metric between the current patch and all the patches of the second image
        metric_result = np.array(metric(patches_1, patches_2))
        # get the index of the minimum/maximum value for the second image up to a threshold
        first_index = np.argmin(metric_result) if metric == SSD else np.argmax(metric_result)
        first_match = metric_result[first_index]
        metric_result[first_index] = np.iinfo(np.uint32).max if metric == SSD else -np.iinfo(np.uint32).max
        second_index = np.argmin(metric_result) if metric == SSD else np.argmax(metric_result)
        second_match = metric_result[second_index]
        q = relevant_im2_corners[first_index]

        # check if the ratio between the best and the second-best match is greater than 0.8
        if (metric == SSD and (second_match == 0 or first_match / second_match > 0.8)) or (
                metric == NCC and (second_match == 0 or first_match / second_match < 1.028)):
            mismatching_points.append((p, q))
            continue

        q = relevant_im2_corners[first_index]
        cv2.line(imC, (p[1], p[0]), (q[1] + m, q[0]), (255, 0, 0), 1)

        matching_points.append((p, q))

    if display:
        plt.imshow(imC)
        plt.show()

    return matching_points, mismatching_points


# match_patterns_with_ratio(img1, img2, patch_size=21, metric=SSD, patch_descriptor=hist_patch_im, amount_of_edges=500,
#                           y_th=None,
#                           display=1)

# Study the differences between the different descriptors and also the use of SSD or NCC.
# Present examples that demonstrate the effectiveness of using the different descriptors and the different metrics.

# match_patterns_with_grad(img1, img2, patch_size=31, metric=SSD, patch_descriptor=hist_gradient_opt, amount_of_edges=400, y_th=None, display=1)
# match_patterns_with_grad(img1, img2, patch_size=31, metric=NCC, patch_descriptor=hist_gradient_opt, amount_of_edges=400, y_th=None, display=1)

# match_patterns_with_grad(img1, img2, patch_size=31, metric=SSD, patch_descriptor=gradient_opt, amount_of_edges=400, y_th=None, display=1)
# match_patterns_with_grad(img1, img2, patch_size=31, metric=NCC, patch_descriptor=gradient_opt, amount_of_edges=400, y_th=None, display=1)

# match_patterns(img1, img2, patch_size=31, metric=SSD, patch_descriptor=hist_patch_im, amount_of_edges=400, y_th=None, display=1)
# match_patterns(img1, img2, patch_size=31, metric=NCC, patch_descriptor=hist_patch_im, amount_of_edges=400, y_th=None, display=1)

# match_patterns(img1, img2, patch_size=31, metric=SSD, patch_descriptor=patch_from_im, amount_of_edges=400, y_th=None, display=1)
# match_patterns(img1, img2, patch_size=31, metric=NCC, patch_descriptor=patch_from_im, amount_of_edges=400, y_th=None, display=1)

# Identify incorrect pairs of matched points.
# Mark and display for cases (2), (3), and (4) a pair of incorrectly matched points.
# Answer
# a. In which of the 3 cases are there more incorrect matches?
# b. What may be the reason for the incorrect matches?

def claus6():
    # Mark and display for cases (2)
    _, mismatch2 = match_patterns_with_ratio(img1, img2, patch_size=21, metric=SSD, patch_descriptor=hist_patch_im, amount_of_edges=500, y_th=None, display=1)
    # Mark and display for cases (3)
    _, mismatch3 = match_patterns_with_ratio(img1, img2, patch_size=21, metric=SSD, patch_descriptor=hist_patch_im, amount_of_edges=500, y_th=10, display=1)
    # Mark and display for cases (4)
    _, mismatch4_1 = match_patterns_with_ratio(img1, img2, patch_size=21, metric=SSD, patch_descriptor=patch_from_im, amount_of_edges=500, y_th=None, display=1)
    _, mismatch4_2 = match_patterns_with_ratio(img1, img2, patch_size=21, metric=NCC, patch_descriptor=hist_patch_im, amount_of_edges=500, y_th=None, display=1)
    _, mismatch4_3 = match_patterns_with_grad(img1, img2, patch_size=21, metric=SSD, patch_descriptor=hist_gradient_opt, amount_of_edges=500, y_th=None, display=1)

    n, m = img1.shape
    images = cv2.hconcat([img1, img2])
    image = np.dstack([images, images, images])

    p2, q2 = mismatch2[0]
    p3, q3 = mismatch3[0]
    p4_1, q4_1 = mismatch4_1[0]
    p4_2, q4_2 = mismatch4_2[0]
    p4_3, q4_3 = mismatch4_3[0]

    cv2.line(image, (p2[1], p2[0]), (q2[1] + m, q2[0]), (255, 0, 0), 1)
    cv2.line(image, (p3[1], p3[0]), (q3[1] + m, q3[0]), (0, 255, 0), 1)
    cv2.line(image, (p4_1[1], p4_1[0]), (q4_1[1] + m, q4_1[0]), (255, 255, 55), 1)
    cv2.line(image, (p4_2[1], p4_2[0]), (q4_2[1] + m, q4_2[0]), (0, 0, 255), 1)
    cv2.line(image, (p4_3[1], p4_3[0]), (q4_3[1] + m, q4_3[0]), (0, 255, 255), 1)

    plt.imshow(image)
    plt.show()

    print(f'Case 2 number of mismatch points: {len(mismatch2)}. SSD + no y-coordinate constraint + histogram descriptor')
    print(f'Case 3 number of mismatch points: {len(mismatch3)}. SSD + y-coordinate constraint of 10 pixles + histogram descriptor')
    print(f'Case 4 SSD number of mismatch points: {len(mismatch4_1)}. SSD + no y-coordinate constraint + Patch descriptor')
    print(f'Case 4 NCC number of mismatch points: {len(mismatch4_2)}. NCC + no y-coordinate constraint + histogram descriptor')
    print(f'Case 4 gradient histogram number of mismatch points: {len(mismatch4_3)}. SSD + no y-coordinate constraint + gradient histogram descriptor')


claus6()