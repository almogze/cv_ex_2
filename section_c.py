import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cbook, cm
from matplotlib.colors import LightSource

import section_b as b

img1 = cv2.imread('images/view1.tif', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('images/view5.tif', cv2.IMREAD_GRAYSCALE)


def get_dense_matching(im1, im2, d_range=(20, 120), display=0):
    matching_points = b.match_patterns(im1, im2, patch_size=20, metric=b.NCC, patch_descriptor=b.hist_patch_im,
                                       amount_of_edges=4000, y_th=1, display=display)

    # filter matching_points by d_range
    filtered_matching_points = []
    for (p, q) in matching_points:
        if d_range[0] <= p[1] - q[1] <= d_range[1]:
            filtered_matching_points.append((p, q))

    return filtered_matching_points


def compute_disparity(im1, im2, d_range=(20, 120), display=0):
    matching_points = get_dense_matching(im1, im2, d_range, display)
    disparity = np.zeros(im1.shape)
    for (p, q) in matching_points:
        disparity[p[0], p[1]] = p[1] - q[1]

    if display:
        plt.imshow(disparity, cmap='gray')
        plt.show()

    return disparity


def compute_depth_map(im1, im2, d_range=(20, 120), display=0):
    disparity = compute_disparity(im1, im2, d_range, display)

    positive = disparity > 0
    depth_map = np.zeros(im1.shape)
    depth_map[positive] = 160 / (disparity[positive] + 100)

    if display:
        plt.imshow(depth_map, cmap='gray')
        plt.show()

    return depth_map


def compute_3D_points(im1, im2, d_range=(20, 120), display=0):
    z = compute_depth_map(im1, im2, d_range, display)
    x, y = np.meshgrid(np.arange(im1.shape[1]), np.arange(im1.shape[0]))

    # Present the object given by X,Y,Z using a 3D plot.
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(x, y, z)

    # fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
    #
    # ls = LightSource(20, 45)
    # # To use a custom hillshading mode, override the built-in shading and pass
    # # in the rgb colors of the shaded surface calculated from "shade".
    # rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    # surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
    #                        linewidth=0, antialiased=False, shade=False)
    #
    # plt.show()

    fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))

    ls = LightSource(azdeg=270, altdeg=45)
    rgb = ls.shade(z, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft',
                   norm=plt.Normalize(vmin=z.min(), vmax=z.max()))
    surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=rgb,
                           linewidth=0, antialiased=False, shade=False)

    plt.show()


compute_3D_points(img1, img2, d_range=(20, 120), display=1)
