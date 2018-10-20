import cv2
import numpy as np


def hue_integral_bins(image):
    hues = cv2.split(image)[0]
    height = len(hues)
    width = len(hues[0])

    bins = []
    for i in range(0, 18):
        bins.append(np.zeros((height, width)))

    bin_range = 10
    for y in range(0, height):
        for x in range(0, width):
            hue = hues[y][x]
            if hue == 180:
                hue_bin = 18 - 1
            else:
                hue_bin = int(hue / bin_range)
            bins[hue_bin][y][x] = 1

    integral_bins = []
    for i in range(0, 18):
        integral_bins.append(cv2.integral(bins[i]))
    return integral_bins


def calculate_corners(x, y, width, height):
    left_top = (x, y)
    left_bot = (x, y + height - 1)
    right_top = (x + width - 1, y)
    right_bot = (x + width - 1, y + height - 1)
    return left_top, left_bot, right_top, right_bot


def map_to_integral_corners(img_corners):
    left_top = img_corners[0]
    left_bot = img_corners[1][0], img_corners[1][1] + 1
    right_top = img_corners[2][0] + 1, img_corners[2][1]
    right_bot = (img_corners[3][0] + 1, img_corners[3][1] + 1)
    return left_top, left_bot, right_top, right_bot


def histogram_from_integral(integral_bins, img_corners):
    left_top, left_bot, right_top, right_bot = map_to_integral_corners(img_corners)

    histogram = np.zeros(len(integral_bins))
    for i in range(0, len(integral_bins)):
        value = integral_bins[i][right_bot[1]][right_bot[0]]
        value = value + integral_bins[i][left_top[1]][left_top[0]]
        value = value - integral_bins[i][right_top[1]][right_top[0]]
        value = value - integral_bins[i][left_bot[1]][left_bot[0]]
        histogram[i] = value

    return histogram
