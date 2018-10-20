import numpy as np
import integral_images as ii
from pyemd import emd


class Patches:
    bin = 18
    distance_matrix = np.zeros((bin, bin))

    base = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 8, 7, 6, 5, 4, 3, 2, 1])
    base_index = 0
    for i in range(0, bin):
        index_copy = base_index
        for j in range(0, bin):
            distance_matrix[i][j] = base[index_copy % 18]
            index_copy = index_copy + 1
        base_index = base_index - 1

    def __init__(self, integral_bins, start_x, start_y, width, height, split=(10, 10), q=0.4):
        if q > 1 or q < 0:
            raise ValueError("q must be in range [0, 1]. Was ", q)
        self.q = q
        self.start_x = start_x
        self.start_y = start_y
        self.patch_width = int(width/split[0])
        self.patch_height = int(height/split[1])
        self.patches_histograms = []

        for i in range(0, split[0]):
            for j in range(0, split[1]):
                x = self.start_x + i * self.patch_width
                y = self.start_y + j * self.patch_height
                p_width = self.patch_width
                p_height = self.patch_height
                if i == split[0] - 1:
                    p_width = width - (i * self.patch_width)
                    p_height = height - (j * self.patch_height)
                corners = ii.calculate_corners(x, y, p_width, p_height)
                int_hist = ii.histogram_from_integral(integral_bins, corners)
                self.patches_histograms.append(int_hist)

    def distance(self, other_patches):
        distances = []
        for i in range(0, len(self.patches_histograms)):
            d = emd(self.patches_histograms[i], other_patches.patches_histograms[i], self.distance_matrix)
            distances.append(d)
        distances = sorted(distances)
        return distances[int(self.q * len(distances))]
