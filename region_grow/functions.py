"""
Functions to be used to perform the Region Growing process

"""

# Librerias
import logging
import rasterio as rio
import numpy as np
import pandas as pd
import geopandas as gpd
import region_grow.region as rg
from shapely.geometry import Polygon
from region_grow.classifiers import Classifier, select_balanced_classifier

# from region_grow.region import Region_Grow

# Cargar un raster .tif
def cargar_raster(ruta_archivo: str):
    """
    Loads a raster of a satellite image in .tif format and restructures the image in 1-D
    to be processed later.

    Parameters
    --------------
    file_path: str
        Image location route

    Return
    --------------
    img_1d: ndarray
        Image with height and width dimensions restructured into one dimension R^2 => R
    img_array: ndarray
        Image with original dimensions
    metadata: rasterio.meta
        Metadata of the loaded raster

    """
    with rio.open(ruta_archivo) as raster:
        logging.info("[Cargar] Metadatos de la imagen en raster:", raster.meta)

        # Cambiar la distribucion de la imagen, leerla en un arreglo Numpy
        img_array = np.empty(
            (raster.height, raster.width, raster.count), raster.meta["dtype"]
        )
        for band in range(img_array.shape[2]):
            img_array[:, :, band] = raster.read(band + 1)

        logging.info("[Cargar] Dimension de la imagen cargada:", img_array.shape)

        # Estirar las dimensiones X,Y de la imagen en un solo vector para cada una de las bandas espectrales
        img_1d = img_array[:, :, : img_array.shape[2]].reshape(
            (img_array.shape[0] * img_array.shape[1], img_array.shape[2])
        )
        logging.info(
            "[Cargar] Dimensiones de apilado de la imagen en 1-D", img_1d.shape
        )

        return (img_1d, img_array, raster.meta)


# Consultar las posiciones en el arreglo leido del raster según las cordenadas del pixel
def append_xy_index(df: pd.DataFrame, raster_path: str) -> pd.DataFrame:
    """
    Given a Dataframe (pandas), it allows to add two new
    columns indicating the corresponding indexes in the arrangement
    to the pixel

    Parameters
    --------------
    df: pandas.DataFrame
        Dataframe with the information of the coordinates of the points
    raster_path: str
        Complete raster file path with spectral information

    Return
    --------------
    df_rsp: pandas.DataFrame
        Dataframe with the pixel coordinates expressed in its position
        X, Y in the data arrangement.

    """
    with rio.open(raster_path) as raster:
        df[["X_Index", "Y_Index"]] = df.apply(
            lambda row: raster.index(row["LONGITUD"], row["LATITUD"]),
            axis=1,
            result_type="expand",
        )

    return df


# Realizar el recorrido para 8-vecinos de un pixel en busca de nuevos pixeles aptos
# para la componente conectada
def check_hood(
    pixel_cords: tuple,
    data_array: np.ndarray,
    classifer: Classifier,
    seen_pixels: set,
    confirmed_pixels: set,
) -> list:
    """
    It makes a route on the 8 neighbors around a given pixel
    its X,Y indexes in the data arrangement

    Parameters
    --------------
    pixel_cords: (int, int)
        X,Y coordinates of the pixel in the data array
    data_array: numpy.ndarray
        Arrangement with all one pixel spectral information
    classifier: Classifier
        Classifier that allows us to determine that if the pixel
        is suitable for adding or not to the polygon under construction
    seen_pixels: set
        Set with the pixel tuples (X_Index, Y_Index)
        that form that have been added for later
        confirmation
    confirmed_pixels: set
        Set with the pixels in which we have secured
        belonging to the connected component

    Return
    --------------
    new_points: list
        List with the new coordinates (x, y) of the points
        neighbors who meet the condition to be added to the
        polygon

    """
    new_points = []
    for x in range(pixel_cords[0] - 1, pixel_cords[0] + 2):
        for y in range(pixel_cords[1] - 1, pixel_cords[1] + 2):
            tuple_cord = (x, y)
            try:
                if (
                    x > 0
                    and y > 0
                    and tuple_cord not in seen_pixels
                    and tuple_cord not in confirmed_pixels
                ):
                    if tuple_cord == pixel_cords:
                        # print(f'Son el mismo - tuple_cord: {tuple_cord} - pixel_cord: {pixel_cords}')
                        continue
                    else:
                        pixel_data = data_array[x, y, :]
                        if classifer.predict(pixel_data):
                            new_points.append(tuple_cord)
            except IndexError:
                continue
    return new_points


# Creacion del poligono de respuesta
def create_polygon(pixels_selected: set, raster_path: str) -> gpd.GeoDataFrame:
    """
    It allows to transform each of the indexes of the
    pixel data in coordinates for further processing
    the answer polygon

    Parameters
    --------------
    pixels_selected: set
        Set with the pixels selected for the
        Connected component
    raster_path: str
        Route to the raster of origin

    Return
    --------------
    polygon: geopands.GeoDataFrame
        Polygon generated from the points

    """
    with rio.open(raster_path) as raster:
        pixels_cords = []
        for x, y in pixels_selected:
            cord = raster.xy(x, y)
            pixels_cords.append(cord)
        new_polygon_geometry = Polygon(pixels_cords)
        polygon_raw = gpd.GeoDataFrame(
            index=[0], crs=raster.meta["crs"], geometry=[new_polygon_geometry]
        ).unary_union.convex_hull
        new_polygon = gpd.GeoDataFrame(
            index=[0], crs=raster.meta["crs"], geometry=[polygon_raw]
        )
        return new_polygon


def grow_bd_region(
    pixels_indexes: np.ndarray,
    pixels_df: pd.DataFrame,
    img_array: np.ndarray,
    raster_path: str,
    polygon_area: float,
    steps: int = 4,
):
    """
    Generates the polygon using the threshold selection algorithm, the given area    
    and minimizes the difference between the area given by the user and the area covered by
    the generated polygon

    Parameters
    --------------
    pixels_indexes: numpy.ndarray
        Arrangement with each of the coordinates (X,Y) of the seed points
    pixels_df: pandas.DataFrame
        Dataframe with the spectral reflectance of the seed points
    img_array: numpy.ndarray
        Satellite image with which the algorithm is processed
    raster_path: str
        Path to .tif file with raster information
    polygon_area: float
        Reference area of the polygon to be generated in hectares (ha)    
    steps: int
        Maximum number of iterations that the algorithm will perform for 
        calculate a polygon with the smallest difference between given approximate value
        and the calculated

    Return
    --------------
    pixels_selected_rsp: set
        Set with the indexes (X,Y) of the selected pixels
    created_polygon: geopands.GeoDataFrame
        Polygon generated from the points

    """

    threshold = 0.1
    min_polygon_area = 999999
    pixels_selected_rsp = None
    created_polygon_rsp = None
    for step in range(steps):
        classifier_class = select_balanced_classifier(classifier_tag="BD")
        classifier = classifier_class(pixels_df, threshold)
        classifier_rg = rg.Region_Grow(
            pixels_indexes=pixels_indexes, img_array=img_array, classifier=classifier
        )
        pixels_selected = classifier_rg.grow()
        created_polygon = create_polygon(
            pixels_selected=pixels_selected, raster_path=raster_path
        )

        # Calcular el area con respecto al poligono esperado
        metric_polygon = created_polygon.to_crs(epsg=32618)
        hectares_area = float(metric_polygon.area) / 10000
        min_check_area = abs(hectares_area - polygon_area)
        if min_check_area <= min_polygon_area:
            min_polygon_area = min_check_area
            pixels_selected_rsp = pixels_selected
            created_polygon_rsp = created_polygon
            threshold += 0.5
        else:
            break
    return (pixels_selected_rsp, created_polygon_rsp)


def grow_edr_region(
    pixels_indexes: np.ndarray,
    pixels_df: pd.DataFrame,
    img_array: np.ndarray,
    raster_path: str,
    polygon_area: float,
    steps: int = 4,
):
    """
    It generates the polygon using the Euclidean distance algorithm by balancing the average,
    the given area and minimizes the difference between the area given by the user and the area covered by
    the generated polygon.

    Parameters
    --------------
    pixels_indexes: numpy.ndarray
        Arrangement with each of the coordinates (X,Y) of the seed points
    pixels_df: pandas.DataFrame
        Dataframe with the spectral reflectance of the seed points
    img_array: numpy.ndarray
        Satellite image with which the algorithm is processed
    raster_path: str
        Path to .tif file with raster information
    polygon_area: float
        Reference area of the polygon to be generated in hectares (ha)    
    steps: int
        Maximum number of iterations the algorithm will perform for 
        calculate a polygon with the smallest difference between given approximate value
        and the calculated

    Return
    --------------
    pixels_selected_rsp: set
        Set with the indexes (X,Y) of the selected pixels
    created_polygon_en: geopands.GeoDataFrame
        Polygon generated from the points

    """

    threshold = 3
    min_polygon_area = 999999
    pixels_selected_rsp = None
    created_polygon_rsp = None
    for step in range(steps):
        classifier_class = select_balanced_classifier(classifier_tag="EDR")
        classifier = classifier_class(pixels_df, threshold)
        classifier_rg = rg.Region_Grow(
            pixels_indexes=pixels_indexes, img_array=img_array, classifier=classifier
        )
        pixels_selected = classifier_rg.grow()
        created_polygon = create_polygon(
            pixels_selected=pixels_selected, raster_path=raster_path
        )

        # Calcular el area con respecto al poligono esperado
        metric_polygon = created_polygon.to_crs(epsg=32618)
        hectares_area = float(metric_polygon.area) / 10000
        min_check_area = abs(hectares_area - polygon_area)
        if min_check_area <= min_polygon_area:
            min_polygon_area = min_check_area
            pixels_selected_rsp = pixels_selected
            created_polygon_rsp = created_polygon
            threshold += 1
    return (pixels_selected_rsp, created_polygon_rsp)


def grow_balanced_region(
    classifier_tag: str,
    pixels_indexes: np.ndarray,
    pixels_df: pd.DataFrame,
    img_array: np.ndarray,
    raster_path: str,
    polygon_area: float,
    steps: int = 4,
):
    """
    Generates the polygon using a growth algorithm that minimizes the difference 
    between the area given by the user and the area covered by the generated polygon.

    Parameters
    --------------
    classifier_tag: str
        Type of algorithm to be used
    pixels_indexes: numpy.ndarray
        Arrangement with each of the coordinates (X,Y) of the seed points
    pixels_df: pandas.DataFrame
        Dataframe with the spectral reflectance of the seed points
    img_array: numpy.ndarray
        Satellite image with which the algorithm is processed
    raster_path: str
        Path to .tif file with raster information
    polygon_area: float
        Reference area of the polygon to be generated in hectares (ha)    
    steps: int
        Maximum number of iterations that the algorithm will perform for 
        calculate a polygon with the smallest difference between given approximate value
        and the calculated

    Return
    --------------
    pixels_selected_rsp: set
        Set with the indexes (X,Y) of the selected pixels
    created_polygon: geopands.GeoDataFrame
        Polygon generated from the points

    """
    pixels_selected = None
    created_polygon = None

    if classifier_tag == "BD":
        pixels_selected, created_polygon = grow_bd_region(
            pixels_indexes=pixels_indexes,
            pixels_df=pixels_df,
            img_array=img_array,
            raster_path=raster_path,
            polygon_area=polygon_area,
            steps=steps,
        )
    elif classifier_tag == "EDR":
        pixels_selected, created_polygon = grow_edr_region(
            pixels_indexes=pixels_indexes,
            pixels_df=pixels_df,
            img_array=img_array,
            raster_path=raster_path,
            polygon_area=polygon_area,
            steps=steps,
        )
    else:
        raise NotImplementedError(f"El clasificador {classifier_tag} no existe.")

    return (pixels_selected, created_polygon)
