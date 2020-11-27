"""
Funciones a utilizar para realizar el proceso de Region Growing
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
    Carga un raster de una imagen satelital en formato .tif y reestructura la imagen en 1-D
    para ser procesada posteriormente por un algoritmo de aprendizaje automatico.

    Parameters
    --------------
    ruta_archivo: str
        Ruta de localizacion de la imagen

    Return
    --------------
    img_1d: ndarray
        Imagen con dimensiones de alto y ancho re-estructuradas en una sola dimensión R^2 => R
    img_array: ndarray
        Imagen con dimensiones originales
    metadata: rasterio.meta
        Metadatos del raster cargado
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
def append_xy_index(df: pd.DataFrame, raster_path: str):
    """
    Dado un Dataframe (pandas), permite agregar dos nuevas
    columnas que indican los indices en el arreglo que corresponden
    al pixel

    Parameters
    --------------
    df: pandas.DataFrame
        Dataframe con la información de las cordenadas de los puntos
    raster_path: str
        Ruta completa del archivo raster con la informacion espectral

    Return
    --------------
    df_rsp: pandas.DataFrame
        Dataframe con los indices X,Y del pixel que se ubica
        en las coordenadas dadas.
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
):
    """
    Realiza un recorrido sobre los 8 vecinos alrededor de un pixel dadas
    sus indices X,Y en el arreglo de datos

    Parameters
    --------------
    pixel_cords: (int, int)
        Coordenadas X,Y del pixel en el arreglo de datos
    data_array: numpy.ndarray
        Arreglo con todas la información espectral de un pixel
    classifier: Clasification
        Clasificador que nos permite determinar que si el pixel
        es apto para agregar o no al poligono en construccion
    seen_pixels: set
        Conjunto con las tuplas de pixeles (X_Index, Y_Index)
        que forman que se han agregado para su posterior
        confirmacion
    confirmed_pixels: set
        Conjunto con los pixeles en los cuales hemos asegurado
        que pertenecen a la componente conectada

    Return
    --------------
    new_points: list
        Lista con la nuevas coordenadas (x, y) de los puntos
        vecinos que cumplan la condicion para ser agregados al
        poligono
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
def create_polygon(pixels_selected: set, raster_path: str):
    """
    Permite transformar cada una de los indices del arreglo de
    datos del pixel en cordenadas para posteriormente elaborar
    el poligono respuesta

    Parameters
    --------------
    pixels_selected: set
        Conjunto con los pixeles seleccionados para la
        componente conectada
    raster_path: str
        Ruta al raster de origen

    Return
    --------------
    polygon: geopandas.GeoDataFrame
        Poligono generado a partir de los puntos
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
    classifier_tag: str,
    pixels_indexes: np.ndarray,
    pixels_df: pd.DataFrame,
    img_array: np.ndarray,
    raster_path: str,
    polygon_area: float,
    steps: int = 4,
):
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
    classifier_tag: str,
    pixels_indexes: np.ndarray,
    pixels_df: pd.DataFrame,
    img_array: np.ndarray,
    raster_path: str,
    polygon_area: float,
    steps: int = 4,
):
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
    pixels_indexes: numpy.ndarray
        Indices (X,Y) de cada uno de los pixeles semilla
    img_array: numpy.ndarray
        Pixeles del raster leido
    classifier_tag: str
        Indicador del clasificador a renderizar
    raster_path: str
        Ruta del raster de carga
    """
    pixels_selected = None
    created_polygon = None

    if classifier_tag == "BD":
        pixels_selected, created_polygon = grow_bd_region(
            classifier_tag=classifier_tag,
            pixels_indexes=pixels_indexes,
            pixels_df=pixels_df,
            img_array=img_array,
            raster_path=raster_path,
            polygon_area=polygon_area,
            steps=steps,
        )
    elif classifier_tag == "EDR":
        pixels_selected, created_polygon = grow_edr_region(
            classifier_tag=classifier_tag,
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
