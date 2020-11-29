"""
Execution of the algorithm to perform Region Grow
"""

import numpy as np
import region_grow.classifiers as cfs
import region_grow.functions as func


class Region_Grow:
    """
    It allows to calculate the connected component from the initial points.

    Parameters
    --------------
    pixels_indexes: numpy.ndarray
        Indexes (X,Y) of each of the seed pixels
    img_array: numpy.ndarray
        Pixels of the raster read
    classifier: Classifier
        Sorting method to decide which neighboring pixels to add

    """

    def __init__(
        self,
        pixels_indexes: np.ndarray,
        img_array: np.ndarray,
        classifier: cfs.Classifier,
    ):
        self.pixels_indexes = pixels_indexes
        self.img_array = img_array
        self.classifier = classifier

    """
    Compute the growth of the region using each of the seed pixels given
    
    Return
    --------------
    pixels_group: set
        A Set with the array index (X_Index, Y_Index) for each of the seed pixels given
        
    """

    def grow(self):
        pixels_queue = set([tuple(i) for i in self.pixels_indexes])
        pixels_group = set()
        while len(pixels_queue) > 0:
            pixel = pixels_queue.pop()
            new_neighborhood = func.check_hood(
                pixel_cords=pixel,
                data_array=self.img_array,
                classifer=self.classifier,
                seen_pixels=pixels_queue,
                confirmed_pixels=pixels_group,
            )
            if len(new_neighborhood) > 0:
                pixels_queue = pixels_queue | set(new_neighborhood)
            pixels_group.add(pixel)

        self.pixels_queue = pixels_queue
        self.pixels_group = pixels_group
        return pixels_group
