import cv2
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy.linalg import null_space


# img = cv2.imread('.\images\Sudoku.PNG', cv2.IMREAD_GRAYSCALE)
# edges = cv2.Canny(img, 250, 500, 5)
# nonzero_indices = np.nonzero(edges)
# l_points = np.array(list(zip(nonzero_indices[0], nonzero_indices[1])))


# def plot_graph(img, title=""):
#     plt.imshow(img, extent=[-90, 90, -r_max, r_max], aspect="auto")
#     plt.title(title)
#     plt.show()


def get_edges_indices(edges):
    nonzero_indices = np.nonzero(edges)
    l_points = np.array(list(zip(nonzero_indices[0], nonzero_indices[1])))
    return l_points


def create_synthetic_img():
    lines_image = np.zeros((100, 100))
    lines_image[20:60, 70] = 1
    lines_image[80, 10:50] = 1
    z = draw_line(1, -15)
    lines_image[z] = 1
    return lines_image


def draw_line(m, c):
    x = np.linspace(0, 99, 100)  # array of x values
    y = np.linspace(0, 99, 100)  # array of y values
    x, y = np.meshgrid(x, y)  # Create a grid of points
    z = np.where((y - (m * x + c)) == 0)
    return z


# img = create_synthetic_img()
# plt.imshow(img)
# plt.show()
# l_points = get_edges_indices(img)


# Input: a set of edge points (or corners), and the resolution of the distance and angles.
# output: the Hough matrix (H) containing votes for lines represented by r and θ.
def H_matrix(img, L_points, resolution_r, resolution_ang):
    n, m = img.shape
    r_max = np.ceil(np.sqrt(n ** 2 + m ** 2))

    r_space = np.arange(-r_max, r_max, resolution_r)
    theta_space = np.deg2rad(np.arange(-90, 90, resolution_ang))

    h_matrix = np.zeros((len(r_space), len(theta_space)))

    cosines = np.cos(theta_space)
    sines = np.sin(theta_space)

    for [y, x] in L_points:
        rs = x * cosines + y * sines
        for t_index, r in enumerate(rs):
            r_index = np.argmin(np.abs(r_space - r))
            h_matrix[r_index, t_index] += 1

    return h_matrix, r_space, theta_space


# resolution_r = 1
# resolution_ang = 1
#
# h, r_space, theta_space = H_matrix(img, l_points, resolution_r, resolution_ang)
#
# r_max = np.max(r_space)
#
# plt.imshow(h, extent=[-90, 90, r_max, -r_max], aspect="auto")
# plt.xlabel("theta")
# plt.ylabel("r")
# plt.title("accumulation matrix")
# plt.show()


# Input: The Hough matrix $H$, and a threshold for the number of minimal points on the line.
# output a list of triplets:  $(r, \Theta, num_points)$ where
# num_points is the number of points on that line.

def list_lines(H, th):
    # th - number of minimal points on the line
    H_reduced = H > th
    indices = np.where(H_reduced > 0)
    triplets = np.column_stack((indices[0], indices[1], H[indices])).astype(np.uint32)
    return triplets


# th = 38
# list_of_lines = list_lines(h, th)


# Display the detected lines in red - overlaid the original image
# Note: one way to do is, is to add the red lines to the image, and then display it
def display_lines(img, list_lines, r_space, theta_space):
    imC = np.dstack([img, img, img])  # a gray level image that is saved as a color image
    for (r_ind, t_ind, _) in list_lines:
        rho = r_space[r_ind]
        theta = theta_space[t_ind]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * a)
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * a)

        cv2.line(imC, (x1, y1), (x2, y2), (255, 0, 0), thickness=1)

    plt.imshow(imC)
    plt.show()


# display_lines(img, list_of_lines, r_space, theta_space)


# Now use the above functions to implement
def straight_lines(image_file, res_r, res_orient, min_number_points, display):
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 250, 500, 5)
    l_points = get_edges_indices(edges)
    h, r_space, theta_space = H_matrix(img, l_points, res_r, res_orient)
    list_of_lines = list_lines(h, min_number_points)
    if display:
        display_lines(img, list_of_lines, r_space, theta_space)
    return list_of_lines, r_space, theta_space


# straight_lines('images/Sudoku.PNG', 1, 1, 500, True)
# straight_lines('images/Crosswalk.jpg', 1, 15, 80, True)
straight_lines('images/linesOnTheRoadGray.jpg', 1, 8, 100, True)
