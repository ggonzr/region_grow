"""
Ejecucion del algoritmo para realizar Region Grow
"""

import numpy as np
import region_grow.classifiers as cfs
import region_grow.functions as func


class Region_Grow:
    """
    Permite calcular la componente conectada a partir de los puntos
    iniciales.

    Parameters
    --------------
    pixels_indexes: numpy.ndarray
        Indices (X,Y) de cada uno de los pixeles semilla
    img_array: numpy.ndarray
        Pixeles del raster leido
    classifier: Classifier
        Metodo de clasificacion para decidir que pixeles vecinos agregar

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
    Computa el crecimiento de la region dado cada uno de los pixeles semilla
    
    Return
    --------------
    pixels_group: set
        Conjunto con tuplas (X_Index, Y_Index) en el arreglo de lectura del raster (img_array)
        de cada uno de los pixeles que pertenecen a la componente conectada
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
