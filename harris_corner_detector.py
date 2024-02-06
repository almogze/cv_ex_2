import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d


def plot_graph(img, title=""):
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.show()


def plot_graphs(img1, img2, img1_title="", img2_title=""):
    figure, axis = plt.subplots(1, 2, figsize=(10, 5))
    axis[0].imshow(img1)
    axis[0].set_title(img1_title)
    axis[1].imshow(img2)
    axis[1].set_title(img2_title)
    plt.show()


def plot_overlay(img, title, dots):
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title(title)
    plt.scatter(dots[:, 0], dots[:, 1], color="red", s=0.5)
    plt.show()


def plot_overlays(img, dots1, dots2, title1="", title2=""):
    figure, axis = plt.subplots(1, 2, figsize=(10, 5))
    axis[0].imshow(img)
    axis[0].set_title(title1)
    axis[0].scatter(dots1[:, 0], dots1[:, 1], color="red", s=0.5)
    axis[1].imshow(img)
    axis[1].set_title(title2)
    axis[1].scatter(dots2[:, 0], dots2[:, 1], color="red", s=0.5)
    plt.show()


def create_grid(sig=1):
    x = np.linspace(-sig, sig, 2 * sig + 1)  # array of x values
    y = np.linspace(-sig, sig, 2 * sig + 1)  # array of y values

    return np.meshgrid(x, y)  # Create a grid of points


def crop(img, pad):
    m, n = np.shape(img)

    return img[pad:m - pad, pad:n - pad]  # Removing the given pad from the image


def Gaussian(sig=1):
    x, y = create_grid(sig)

    return (1 / (2 * np.pi * sig ** 2)) * np.exp(-(np.square(x) + np.square(y)) / (2 * sig ** 2))


def gaussian_derivative_x(x, y, sig=1):
    return - (x / (2 * np.pi * sig ** 4)) * np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2))


def gaussian_derivative_y(x, y, sig=1):
    return - (y / (2 * np.pi * sig ** 4)) * np.exp(-(x ** 2 + y ** 2) / (2 * sig ** 2))


def Deriv_Gauss_xy(sig=1):
    x, y = create_grid(sig)

    div_x = gaussian_derivative_x(x, y, sig)
    div_y = gaussian_derivative_y(x, y, sig)

    return div_x, div_y


def Grad_xy(img, sig=1):
    # 1)
    G_dx, G_dy = Deriv_Gauss_xy(sig)

    Ix_pad = convolve2d(img, G_dx)
    Iy_pad = convolve2d(img, G_dy)

    pad = np.floor(len(G_dx) / 2).astype(int)

    return crop(Ix_pad, pad), crop(Iy_pad, pad)


def d_q(det, trace, k=1):
    return det + k * (trace ** 2)  # Calc the D(q) using the given matrix det and trace


def local_maxima(img, denisty_size):
    m, n = img.shape
    img_copy = img.copy()

    for y in range(0, m, denisty_size):
        for x in range(0, n, denisty_size):
            sub_img = img_copy[y: y + denisty_size, x: x + denisty_size]
            sub_img[sub_img < np.max(sub_img)] = 0

    return img_copy  # A new img with only the maximal value in each window defined by density size


def interesting_points(mat):
    x, y = np.where(mat > 0)
    return np.array(list(zip(y, x)))


def normalize(img):
    min_val = np.min(img)
    max_val = np.max(img)

    return 255 * (img - min_val) / (
                max_val - min_val)  # A normalized version of the input image, rescaled to the range [0, 255]


def H_corner(img, sigma_smooth=1, sigma_neighb=1, k=10, th=400, density_size=10, display=0):
    # 2)
    Ix, Iy = Grad_xy(img, sigma_smooth)

    # 3)
    Ix_square = Ix ** 2
    Iy_square = Iy ** 2
    IxIy = Ix * Iy

    # 4)
    G = Gaussian(sigma_neighb)
    C11 = crop(convolve2d(Ix_square, G), sigma_neighb)
    C12 = C21 = crop(convolve2d(IxIy, G), sigma_neighb)
    C22 = crop(convolve2d(Iy_square, G), sigma_neighb)

    # 5)
    det = C11 * C22 - C12 * C21
    trace = C11 + C22

    D = d_q(det, trace, k)

    # 6)
    D_th = D.copy()
    D_th[D_th < th] = 0

    # 7)
    D_maxima = local_maxima(D_th, density_size)

    detected_corners = interesting_points(D_maxima)

    if display == 1:
        # plot_graph(img, title="The original image")
        # plot_graphs(img1=Ix, img1_title="The derivatives of Ix", img2=Iy, img2_title="The derivatives of Iy")
        # plot_graphs(img1=normalize(D), img1_title="normalized D", img2=normalize(D_th) > 0,
        #             img2_title="normelized D_th")
        plot_overlay(img, "detected_corners", detected_corners)

    # return detected_corners
    return D_maxima
