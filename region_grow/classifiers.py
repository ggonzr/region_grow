"""
Clases con la definicion de los diferentes clasificadores para
construir las componentes conectadas para el proceso de Region Grow
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
    Define la clase que suministrará el metodo de
    clasificación para los nuevos pixeles dada la semilla

    Parameters
    --------------
    pixels_df: pandas.DataFrame
        Dataframe con cada una de las reflectancias espectrales o
        indices a utilizar para los pixeles asociados a los puntos
        de entrada
    """

    @abstractmethod
    def __init__(self, pixels_df: pd.DataFrame):
        pass

    """
    Define la forma en la cual el modelo de clasificación
    sera entrenado.
    """

    @abstractmethod
    def fit(self):
        pass

    """
    Predice si un pixel dado se puede agregar o no
    a la componente conectada en construcción
    
    Parameters
    --------------
    pixel: numpy.ndarray
        Vector con cada uno de los valores de las reflectancias
        espectrales o indices que componen el pixel
    
    Return
    --------------
    True/False: bool
        En caso de que el pixel cumpla con las condiciones
        propuestas para agregarlo o no a la componente
        conectada.    
    """

    @abstractmethod
    def predict(self, pixel: np.ndarray):
        pass


# Clasificador de pixeles mediante intervalo de confianza
class ConfidenceInterval(Classifier):
    """
    Permite realizar una clasificación de los pixeles
    mediante el uso de un intervalo de confianza
    para cada una de las bandas espectrales o indices

    Parameters
    --------------
    pixels_df: pandas.DataFrame
        Dataframe con cada una de las reflectancias espectrales o
        indices a utilizar para los pixeles asociados a los puntos
        de entrada
    confidence_lvl: float
        Nivel de confianza en la cual se construye el intervalo
        confidence_lvl in [0, 1)
    """

    def __init__(self, pixels_df, confidence_lvl=0.99):
        super()
        self.pixels_df = pixels_df
        self.confidence_lvl = confidence_lvl
        self.intervals = self.fit()

    """
    Construye un intervalo de confianza para cada una de
    las bandas o indices para usarlo como clasificación
    """

    def fit(self):
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

    """
    Realiza la predicción de acuerdo a los valores del 
    intervalo de confianza
    """

    def predict(self, pixel):
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
    Permite realizar una clasificación de los pixeles
    calculando la suma de las diferencias de la distancia
    euclidea entre la banda i del pixel contra la media de
    la banda i de los pixeles semilla

    Parameters
    --------------
    pixels_df: pandas.DataFrame
        Dataframe con cada una de las reflectancias espectrales o
        indices a utilizar para los pixeles asociados a los puntos
        de entrada
    """

    def __init__(self, pixels_df):
        super()
        self.pixels_df = pixels_df
        self.bands_mean = self.fit()
        self.threshold = self.threshold()

    """
    Calcula la media de cada una de las bandas presentes
    en el conjunto de pixeles semilla (DataFrame)
    
    Return
    --------------
    bands_mean: numpy.ndarray
        Arreglo con la media de cada una de las bandas del
        dataframe siendo la primera posición la media de la banda 1
    """

    def fit(self):
        description = self.pixels_df.describe()
        numpy_mean = description.loc[["mean"]].to_numpy()
        return numpy_mean

    """
    Calcula la media de cada una de las bandas presentes
    en el conjunto de pixeles semilla (DataFrame)
    
    Return
    --------------
    threshold: float
        Umbral de diferencia aceptable para considerar a un pixel como
        miembro de la componente conectada. Se calcula a partir de los
        puntos semilla.
    """

    def threshold(self):
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

    """
    Predice si un pixel dado se puede agregar o no
    a la componente conectada en construcción
    
    Parameters
    --------------
    pixel: numpy.ndarray
        Vector con cada uno de los valores de las reflectancias
        espectrales o indices que componen el pixel
    
    Return
    --------------
    True/False: bool
        En caso de que el pixel cumpla con las condiciones
        propuestas para agregarlo o no a la componente
        conectada.    
    """

    def predict(self, pixel):
        substract = pixel - self.bands_mean
        difference = math.sqrt(np.dot(substract, np.transpose(substract)))
        # print(f'Threshold: {self.threshold}')
        # print(f'Diferencia: {difference}')
        # print('\n')
        return difference <= self.threshold


# Clasificador con distancia euclideana con reajustado y sesgo
class EuclideanDistanceReFit(Classifier):
    """
    Permite realizar una clasificación de los pixeles
    calculando la suma de las diferencias de la distancia
    euclidea entre la banda i del pixel contra la media de
    la banda i de los pixeles semilla.

    La diferencia con la implementacion estandar es que, a medida
    que encuentra nuevos pixeles, re-entrena y ajusta el contenido de las
    bandas con los valores de estos nuevos pixeles

    Parameters
    --------------
    pixels_df: pandas.DataFrame
        Dataframe con cada una de las reflectancias espectrales o
        indices a utilizar para los pixeles asociados a los puntos
        de entrada
    refit_threshold: int
        Numero de pixeles extra a agregar antes de volver a re-entrenar
        el clasificador
    """

    def __init__(self, pixels_df: pd.DataFrame, refit_threshold: int = 3):
        super()
        self.pixels_df = pixels_df
        self.refit_threshold = refit_threshold
        self.new_pixels = 0
        self.bands_mean = self.fit()
        self.threshold = self.compute_threshold()

    """
    Calcula la media de cada una de las bandas presentes
    en el conjunto de pixeles semilla (DataFrame)
    
    Return
    --------------
    bands_mean: numpy.ndarray
        Arreglo con la media de cada una de las bandas del
        dataframe siendo la primera posición la media de la banda 1
    """

    def fit(self):
        description = self.pixels_df.describe()
        numpy_mean = description.loc[["mean"]].to_numpy()
        return numpy_mean

    """
    Calcula la media de cada una de las bandas presentes
    en el conjunto de pixeles semilla (DataFrame)
    
    Return
    --------------
    threshold: float
        Umbral de diferencia aceptable para considerar a un pixel como
        miembro de la componente conectada. Se calcula a partir de los
        puntos semilla.
    """

    """
    A continuacion se deja una version para computar el umbral con
    la media, el resultado de la ejecucion, sobre el lote de test
    (Lote 1 SIA-ASG-51300) no toma ni la mitad de los pixeles de la ejecucion
    estandar - Solo toma 11 - por si las moscas, dejamos su implementacion aqui

    def compute_threshold(self):
        max_threshold = 0
        numpy_df = self.pixels_df.to_numpy()
        bands_mean = self.bands_mean
        for row in range(numpy_df.shape[0]):
            seed_pixel = numpy_df[row, :]
            difference = seed_pixel - bands_mean
            # print(f'Initial Median: {np.dot(difference, np.transpose(difference))}')
            distance = math.sqrt(np.dot(difference, np.transpose(difference)))
            # print(f'Square Root: {distance} \n')
            max_threshold += distance
        rsp = 0 if numpy_df.shape[0] == 0 else max_threshold / numpy_df.shape[0]
        return rsp
    """

    def compute_threshold(self):
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
        Permite re-entrenar el algoritmo con los puntos clasificados
        con el fin de mejorar la precisión para encontrar pixeles similares
        con caña y sesgarlo para lograr este objetivo

        pixel: numpy.ndarray
            Vector con cada uno de los valores de las reflectancias
            espectrales o indices que componen el nuevo pixel que se va a agregar a la componente
            conectada.
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

    """
    Predice si un pixel dado se puede agregar o no
    a la componente conectada en construcción
    
    Parameters
    --------------
    pixel: numpy.ndarray
        Vector con cada uno de los valores de las reflectancias
        espectrales o indices que componen el pixel
    
    Return
    --------------
    True/False: bool
        En caso de que el pixel cumpla con las condiciones
        propuestas para agregarlo o no a la componente
        conectada.    
    """

    def predict(self, pixel):
        substract = pixel - self.bands_mean
        difference = math.sqrt(np.dot(substract, np.transpose(substract)))
        is_true = True if difference <= self.threshold else False
        if is_true:
            self.refit(pixel=pixel)
        return is_true

    def test_df(self):
        return self.pixels_df


# Clasificador de pixeles mediante umbral definido por el usuario
class BandThreshold(Classifier):
    """
    Permite clasificar todos los pixeles similares a uno dado
    El umbral de las bandas sera +- un porcentaje definido por el usuario
    Se clasificara el pixel si y solo si su reflectancia por cada banda espectral
    se encuentra en el margen calculado a partir del umbral.

    ATENCION: Este metodo de clasificación está pensado para realizar el proceso
    de Region Grow con solo un pixel, en caso de que se brinde mas de un pixel
    semilla se avisará al usuario que el proceso solo utilizará el primero que encuentre.

    Parameters
    --------------
    pixels_df: pandas.DataFrame
        Dataframe con cada una de las reflectancias espectrales o
        indices a utilizar para los pixeles asociados a los puntos
        de entrada
    band_threshold: float
        Umbral de similitud a calcular por cada banda
        band_threshold in [0, 1)
    """

    def __init__(self, pixels_df, band_threshold=0.10):
        super()
        self.pixels_df = pixels_df
        self.band_threshold = band_threshold
        self.intervals = self.fit()

    def fit(self):
        """
        Permite entrenar el metodo de clasificacion y calcular
        el umbral de aceptación por cada una de las bandas.
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

    """
    Realiza la predicción de acuerdo a los valores del umbral
    para cada una de las bandas
    """

    def predict(self, pixel):
        for band in range(len(self.intervals)):
            low_boundary, upper_boundary = self.intervals[band]
            if not (low_boundary <= pixel[band] <= upper_boundary):
                return False
        return True


# Seleccionar un clasificador en especifico
def select_classifier(classifier_tag: str) -> Classifier:
    """
    Permite seleccionar un clasificador dado
    su abreviación, para crear la componente conectada

    Parameters
    --------------
    classifier_tag: str
        Tag que define cada uno de los clasificadores registrados
        ED: Distancia Euclideana - EuclideanDistance
        CI: Intervalo de Confianza sobre la media - ConfidenceInterval
        EDR: Distancia euclideana que recomputa la media de las bandas
             despues de X-pixeles nuevos agregados a la componente conectada
             EuclideanDistanceReFit.
        BD: Umbral para una banda de un único pixel semilla - BandThreshold
            Se define un umbral +- para definir el rango aceptado por banda
    Return
    --------------
    rg_classifier: Classifier
        Referencia a la clase del clasificador a instanciar
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
    Permite seleccionar un clasificador dado
    su abreviación, para crear la componente conectada

    Parameters
    --------------
    classifier_tag: str
        Tag que define cada uno de los clasificadores registrados        
        EDR: Distancia euclideana que recomputa la media de las bandas
             despues de X-pixeles nuevos agregados a la componente conectada
             EuclideanDistanceReFit.
        BD: Umbral para una banda de un único pixel semilla - BandThreshold
            Se define un umbral +- para definir el rango aceptado por banda
    Return
    --------------
    rg_classifier: Classifier
        Referencia a la clase del clasificador a instanciar
    """
    switcher = {
        "EDR": EuclideanDistanceReFit,
        "BD": BandThreshold,
    }
    return switcher.get(classifier_tag)

