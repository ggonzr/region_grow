"""
Classes with the definition of the different classifiers for
build the connected components for the Region Grow algorithm

"""

import math
import logging
import numpy as np
import pandas as pd
from scipy.stats import t, sem
from abc import ABC, abstractmethod

# Clase abstracta para definir la interfaz de un clasificador
class Classifier(ABC):
    """
    Define the class that will provide the method of
    classification for the new pixels given the seed

    Parameters
    --------------
    pixels_df: pandas.DataFrame
        Dataframe with each of the spectral reflectances to be used
        for the pixels associated with the input points

    """

    @abstractmethod
    def __init__(self, pixels_df: pd.DataFrame):
        pass

    @abstractmethod
    def fit(self):
        """
        It defines the way in which the classification model
        will be trained.

        """
        pass

    @abstractmethod
    def predict(self, pixel: np.ndarray):
        """
        Predicts whether a given pixel can be added or not
        to the connected component under construction
        
        Parameters
        --------------
        pixel: numpy.ndarray
            Vector with each of the reflectance values
            spectral or indexes that make up the pixel
        
        Return
        --------------
        True/False: bool
            If the pixel meets the conditions
            proposals to add it or not to the component
            connected.    

        """
        pass


# Clasificador de pixeles mediante intervalo de confianza
class ConfidenceInterval(Classifier):
    """
    Allows to make a classification of the pixels
    by using a confidence interval
    for each of the spectral bands or indices

    Parameters
    --------------
    pixels_df: pandas.DataFrame
        Dataframe with each of the spectral reflectances or
        indexes to be used for the pixels associated with the seed
        points
    confidence_lvl: float
        Confidence level at which the interval is built
        confidence_lvl in [0, 1)

    """

    def __init__(self, pixels_df, confidence_lvl=0.99):
        super()
        self.pixels_df = pixels_df
        self.confidence_lvl = confidence_lvl
        self.intervals = self.fit()

    def fit(self):
        """
        Build a confidence interval for each of
        the bands or indexes to use them as a classification
        threshold
        
        """
        response = {}
        columns = list(self.pixels_df.columns)
        for column in columns:
            data_column = self.pixels_df[column]
            degrees_freedom = data_column.size - 1
            mean = np.mean(data_column)
            standard_error = sem(data_column)
            confidence_interval = t.interval(
                self.confidence_lvl, degrees_freedom, mean, standard_error
            )
            response[column] = confidence_interval
        return response

    def predict(self, pixel):
        """
        It makes the prediction according to the values of 
        confidence interval

        Parameters
        --------------
        pixel: numpy.ndarray
            Vector with each of the reflectance values
            spectral or indexes that make up the pixel
            
        Return
        --------------
        True/False: bool
            If the pixel meets the conditions
            proposals to add it or not to the component
            connected.

        """

        for banda, ic in self.intervals.items():
            indice = int(banda.split(" ")[1]) - 1
            media_indice = np.mean(pixel[indice])
            if not (ic[0] <= media_indice <= ic[1]):
                # print(f'Banda: {banda} - Media: {media_indice} - Intervalo: {ic}')
                return False
        return True


# Clasificador con distancia euclideana
class EuclideanDistance(Classifier):
    """
    Allows to make a classification of the pixels
    calculating the sum of the euclidean distance differences
    between the i-band of the pixel against the
    the i-band of seed pixels

    Parameters
    --------------
    pixels_df: pandas.DataFrame
        Dataframe with each of the spectral reflectances or
        indexes to be used for the pixels associated with the points
        of entry

    """

    def __init__(self, pixels_df):
        super()
        self.pixels_df = pixels_df
        self.bands_mean = self.fit()
        self.threshold = self.threshold()

    def fit(self):
        """
        Calculate the average of each of the bands present
        in the seed pixel array (DataFrame)
        
        Return
        --------------
        bands_mean: numpy.ndarray
            Average of each of the bands of the
            dataframe being the first position the average of band 1
            
        """
        description = self.pixels_df.describe()
        numpy_mean = description.loc[["mean"]].to_numpy()
        return numpy_mean

    def threshold(self):
        """
        Calculate the average of each of the bands present
        in the seed pixel array (DataFrame)
        
        Return
        --------------
        threshold: float
            Acceptable difference threshold to consider a pixel as
            member of the connected component. It is calculated from the
            seed points.
            
        """
        max_threshold = -100
        numpy_df = self.pixels_df.to_numpy()
        bands_mean = self.bands_mean
        for row in range(numpy_df.shape[0]):
            seed_pixel = numpy_df[row, :]
            difference = seed_pixel - bands_mean
            # print(f'Initial Median: {np.dot(difference, np.transpose(difference))}')
            distance = math.sqrt(np.dot(difference, np.transpose(difference)))
            # print(f'Square Root: {distance} \n')
            max_threshold = max(max_threshold, distance)
        return max_threshold

    def predict(self, pixel):
        """
        Predicts whether a given pixel can be added or not
        to the connected component under construction
        
        Parameters
        --------------
        pixel: numpy.ndarray
            Vector with each of the reflectance values
            spectral or indexes that make up the pixel
        
        Return
        --------------
        True/False: bool
            If the pixel meets the conditions
            proposals to add it or not to the component
            connected.

        """
        substract = pixel - self.bands_mean
        difference = math.sqrt(np.dot(substract, np.transpose(substract)))
        # print(f'Threshold: {self.threshold}')
        # print(f'Diferencia: {difference}')
        # print('\n')
        return difference <= self.threshold


# Clasificador con distancia euclideana con reajustado y sesgo
class EuclideanDistanceReFit(Classifier):
    """
    Allows to make a classification of the pixels
    calculating the sum of the distance differences
    euclidean between the i-band of the pixel against the
    the i-band of the seed pixels

    The difference with the standard implementation is that, as
    that finds new pixels, retrains and adjusts the content of the
    bands with the values of these new pixels

    Parameters
    --------------
    pixels_df: pandas.DataFrame
        Dataframe with each of the spectral reflectances or
        indexes to be used for the pixels associated with the points
        of entry
    refit_threshold: int
        Number of extra pixels to be added before retraining
        the classifier

    """

    def __init__(self, pixels_df: pd.DataFrame, refit_threshold: int = 3):
        super()
        self.pixels_df = pixels_df
        self.refit_threshold = refit_threshold
        self.new_pixels = 0
        self.bands_mean = self.fit()
        self.threshold = self.compute_threshold()

    def fit(self):
        """
        Calculate the average of each of the bands present
        in the seed pixel array (DataFrame)
        
        Return
        --------------
        bands_mean: numpy.ndarray
            Average of each of the bands of the
            dataframe being the first position the average of band 1
            
        """
        description = self.pixels_df.describe()
        numpy_mean = description.loc[["mean"]].to_numpy()
        return numpy_mean

    def compute_threshold(self):
        """
        Calculates the threshold to determine if a pixel should be
        grouped or not within the polygon to be generated
        
        Return
        --------------
        threshold: float
            Acceptable difference threshold to consider a pixel as
            member of the connected component. It is calculated from the
            seed points.

        """
        max_threshold = -100
        numpy_df = self.pixels_df.to_numpy()
        bands_mean = self.bands_mean
        for row in range(numpy_df.shape[0]):
            seed_pixel = numpy_df[row, :]
            difference = seed_pixel - bands_mean
            # print(f'Initial Median: {np.dot(difference, np.transpose(difference))}')
            distance = math.sqrt(np.dot(difference, np.transpose(difference)))
            # print(f'Square Root: {distance} \n')
            max_threshold = max(max_threshold, distance)
        return max_threshold

    def refit(self, pixel):
        """
        Allows to retrain the algorithm with the classified points
        in order to improve accuracy in finding similar pixels
        and bias it to achieve this goal

        pixel: numpy.ndarray
            Vector with each of the reflectance values
            spectra or indexes that make up the new pixel to be added to the
            connected.

        """
        self.new_pixels += 1

        # Agregar al Dataframe el nuevo dato
        pixel_reshaped = pixel.reshape((1, (pixel.shape[0])))
        new_row = pd.DataFrame(pixel_reshaped)
        new_row.columns = self.pixels_df.columns
        self.pixels_df = self.pixels_df.append(new_row, ignore_index=True)

        # Re entrenar de nuevo el clasificador
        if self.new_pixels % self.refit_threshold == 0:
            self.bands_mean = self.fit()
            # Si se recomputa el umbral con el maximo, toma
            # 1/3 de la imagen, ese resultado no es util.
            # Si se hace con media, tambien da problemas
            # detalles arriba. Sin recalcular el umbral, da resultados
            # aceptables.
            #
            # self.threshold = self.compute_threshold()

    def predict(self, pixel):
        """
        Determines if the pixel should be added to the polygon 
        making the prediction according to the difference
        of the band with the average of the seed pixels for that band.

        Parameters
        --------------
        pixel: numpy.ndarray
            Spectral reflectance of the pixel to be analyzed

        Return
        --------------
        True/False: bool
            If the pixel meets the conditions
            proposals to add it or not to the component
            connected.
            
        """
        substract = pixel - self.bands_mean
        difference = math.sqrt(np.dot(substract, np.transpose(substract)))
        is_true = True if difference <= self.threshold else False
        if is_true:
            self.refit(pixel=pixel)
        return is_true


# Clasificador de pixeles mediante umbral definido por el usuario
class BandThreshold(Classifier):
    """
    Allows to classify all pixels similar to a given one
    The threshold of the bands will be +- a percentage defined by the user
    The pixel will be classified if and only if its reflectance for each spectral band
    is in the margin calculated from the threshold.

    WARNING: This classification method is designed to perform the process
    of Region Grow with only one pixel, in case more than one pixel is provided
    The user will be notified that the process will only use the first one found.

    Parameters
    --------------
    pixels_df: pandas.DataFrame
        Dataframe with each of the spectral reflectances or
        indexes to be used for the pixels associated with the points
        of entry
    band_threshold: float
        Similarity threshold to be calculated for each band
        band_threshold in [0, 1)

    """

    def __init__(self, pixels_df, band_threshold=0.10):
        super()
        self.pixels_df = pixels_df
        self.band_threshold = band_threshold
        self.intervals = self.fit()

    def fit(self):
        """
        Allows training in the method of classification and calculation
        the acceptance threshold for each of the bands

        """
        # Seleccion del primer pixel
        if len(self.pixels_df) > 1:
            warning_message = (
                "[BandThreshold - Classifier] ATENCION: "
                "El conjunto de datos de pixeles semilla tiene mas de 1 registro, "
                "solo se utilizara el primero. "
                "\n"
                "Por favor considere ejecutar de nuevo este algoritmo pasando en el .csv "
                "unicamente las coordenadas del punto de interes"
                "\n"
            )
            logging.warning(warning_message)
        # Seleccionar el primer pixel
        seed_pixel = self.pixels_df.iloc[0, :].to_numpy()
        threshold_intervals = []
        for band in seed_pixel:
            th = band * self.band_threshold
            threshold_intervals.append((band - th, band + th))
        return threshold_intervals

    def predict(self, pixel):
        """
        Determines if the pixel should be added to the polygon 
        making the prediction according to the threshold values
        for each of the bands

        Parameters
        --------------
        pixel: numpy.ndarray
            Spectral reflectance of the pixel to be analyzed

        Return
        --------------
        True/False: bool
            If the pixel meets the conditions
            proposals to add it or not to the component
            connected.
            
        """
        for band in range(len(self.intervals)):
            low_boundary, upper_boundary = self.intervals[band]
            if not (low_boundary <= pixel[band] <= upper_boundary):
                return False
        return True


# Seleccionar un clasificador en especifico
def select_classifier(classifier_tag: str) -> Classifier:
    """
    Allows you to select a given sorter
    its abbreviation, to create the connected component

    Parameters
    --------------
    classifier_tag: str
        Tag that defines each of the registered classifiers

        ED: Euclidean Distance - EuclideanDistance
            Calculates the euclidean distance between the band of the pixel to be analyzed and
            the average of the accepted threshold
        CI: Confidence Interval over the average - ConfidenceInterval
            It performs a confidence interval as a way to determine the
            acceptance threshold
        EDR: Euclidean distance that recomputes the average of the bands
            after X-pixels new added to the connected component
            EuclideanDistanceReFit.
        BD: Threshold for a single seed pixel band - BandThreshold
            A +- threshold is defined to define the accepted range per band
            
    Return
    --------------
    rg_classifier: Classifier
        Reference to the class of the sorter to be instantiated

    """
    switcher = {
        "ED": EuclideanDistance,
        "CI": ConfidenceInterval,
        "EDR": EuclideanDistanceReFit,
        "BD": BandThreshold,
    }
    return switcher.get(classifier_tag)


# Seleccionar un clasificador en especifico para metodos con
# balanceo
def select_balanced_classifier(classifier_tag: str) -> Classifier:
    """
    Allows you to select a given sorter
    its abbreviation, to create the connected component

    Parameters
    --------------
    classifier_tag: str
        Tag that defines each of the registered classifiers  

        EDR: Euclidean distance that recomputes the average of the bands
            after X-pixels new added to the connected component
            EuclideanDistanceReFit.
        BD: Threshold for a single seed pixel band - BandThreshold
            A +- threshold is defined to define the accepted range per band

    Return
    --------------
    rg_classifier: Classifier
        Reference to the class of the sorter to be instantiated

    """
    switcher = {
        "EDR": EuclideanDistanceReFit,
        "BD": BandThreshold,
    }
    return switcher.get(classifier_tag)

