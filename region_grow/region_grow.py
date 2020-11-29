#!/usr/bin/env python
# coding: utf-8

# Librerias
import sys
import argparse
import tkinter
import logging
import tkinter.filedialog as fl
import numpy as np
import pandas as pd

# Modulos propios
from . import classifiers as classifiers
from . import functions as functions
from . import region as rg


def archivo_cargar(files: list):
    """
    Permite seleccionar el archivo que va a ser cargado

    Parameters
    --------------
    files: list
        Tupla con el tipo de archivo y su extension

    Return
    --------------
    ruta: str
        Ruta absoluta del archivo a cargar
    """
    root.withdraw()
    filename = fl.askopenfilename(filetypes=files, defaultextension=files)
    return filename


def archivo_guardar(files: list):
    """
    Permite seleccionar el archivo qe va a ser guardado

    Parameters
    --------------
    files: list
        Tupla con el tipo de archivo y su extension


    Return
    --------------
    ruta: str
        Ruta absoluta del nuevo archivo que se va a generar
    """
    root.withdraw()
    filename = fl.asksaveasfilename(filetypes=files, defaultextension=files)
    return filename


def execute(
    points_path: str, raster_path: str, shape_path: str, classifier_tag: str = "ED"
):
    """
    Ejecuta el proceso para computar el crecimiento de
    la region

    Parameters
    --------------
    points_path: str
        Ruta al archivo .csv con las cordenadas de los puntos
    raster_path: str
        Ruta al archivo .tif con la informacion del raster
    shape_path: str
        Ruta para guardar el archivo .shp con los poligonos
    classifier_tag: str
        Tipo de clasificador para utilizar en el proceso
    """

    puntos_csv = points_path
    raster_all_bands = raster_path

    # Cargar los datos
    # ----------------------------------------
    puntos_cords = pd.read_csv(puntos_csv, sep=";", decimal=",")

    # Carga del raster
    # ----------------------------------------
    img_1d, img_array, raster_metadata = functions.cargar_raster(raster_all_bands)

    # Buscar las coordenadas X,Y en el arreglo para cada tupla de coordenadas
    # ----------------------------------------
    puntos_cords = functions.append_xy_index(puntos_cords, raster_all_bands)

    # Sacar los indices del Dataframe para iterar sobre los datos
    # ----------------------------------------
    pixels_indexes = puntos_cords[["X_Index", "Y_Index"]].to_numpy(copy=True)
    pixels_data = np.empty(
        (len(pixels_indexes), raster_metadata["count"]), raster_metadata["dtype"]
    )

    for idx in range(len(pixels_indexes)):
        x_index, y_index = pixels_indexes[idx]
        pixels_data[idx, :] = img_array[x_index, y_index, :]

    # Crear el Dataframe y asignar columnas
    # ----------------------------------------
    columns_name = [f"Banda {idx + 1}" for idx in range(raster_metadata["count"])]
    pixels_df = pd.DataFrame(pixels_data, columns=columns_name)

    # Creamos el clasificador y la cola de puntos semilla sobre los cuales
    # generaremos el poligono (componente conectada)
    # ----------------------------------------
    classifier_class = classifiers.select_classifier(classifier_tag=classifier_tag)

    if classifier_class is None:
        raise NotImplementedError(f"El clasificador {classifier_tag} no existe.")

    classifier = classifier_class(pixels_df)
    classifier_rg = rg.Region_Grow(
        pixels_indexes=pixels_indexes, img_array=img_array, classifier=classifier
    )
    pixels_selected = classifier_rg.grow()

    created_polygon = functions.create_polygon(
        pixels_selected=pixels_selected, raster_path=raster_all_bands
    )

    # Ruta de almacenamiento
    created_polygon.to_file(filename=shape_path, driver="ESRI Shapefile")


def execute_with_area(
    points_path: str,
    raster_path: str,
    shape_path: str,
    classifier_tag: str = "BD",
    steps: int = 4,
):
    """
    Ejecuta el proceso para computar el crecimiento de
    la region

    Parameters
    --------------
    points_path: str
        Ruta al archivo .csv con las cordenadas de los puntos
    raster_path: str
        Ruta al archivo .tif con la informacion del raster
    shape_path: str
        Ruta para guardar el archivo .shp con los poligonos
    classifier_tag: str
        Tipo de clasificador para utilizar en el proceso
    """

    puntos_csv = points_path
    raster_all_bands = raster_path

    # Cargar los datos
    # ----------------------------------------
    puntos_cords = pd.read_csv(puntos_csv, sep=";", decimal=",")

    # Carga del raster
    # ----------------------------------------
    img_1d, img_array, raster_metadata = functions.cargar_raster(raster_all_bands)

    # Buscar las coordenadas X,Y en el arreglo para cada tupla de coordenadas
    # ----------------------------------------
    puntos_cords = functions.append_xy_index(puntos_cords, raster_all_bands)

    # Sacar los indices del Dataframe para iterar sobre los datos
    # ----------------------------------------
    pixels_indexes = puntos_cords[["X_Index", "Y_Index"]].to_numpy(copy=True)
    pixels_data = np.empty(
        (len(pixels_indexes), raster_metadata["count"]), raster_metadata["dtype"]
    )

    for idx in range(len(pixels_indexes)):
        x_index, y_index = pixels_indexes[idx]
        pixels_data[idx, :] = img_array[x_index, y_index, :]

    # Crear el Dataframe y asignar columnas
    # ----------------------------------------
    columns_name = [f"Banda {idx + 1}" for idx in range(raster_metadata["count"])]
    pixels_df = pd.DataFrame(pixels_data, columns=columns_name)

    # Creamos el clasificador y la cola de puntos semilla sobre los cuales
    # generaremos el poligono (componente conectada)
    # ----------------------------------------
    pixels_selected, created_polygon = functions.grow_balanced_region(
        classifier_tag=classifier_tag,
        pixels_indexes=pixels_indexes,
        pixels_df=pixels_df,
        img_array=img_array,
        raster_path=raster_path,
        polygon_area=puntos_cords["HECTAREAS"][0],
        steps=steps,
    )
    # Ruta de almacenamiento
    created_polygon.to_file(filename=shape_path, driver="ESRI Shapefile")


if __name__ == "__main__":
    # Logger
    logging.basicConfig(
        format="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p"
    )

    # Ejecucion por CLI
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Crear un poligono utilizando Region Growing a partir de unos puntos iniciales"
        )
        parser.add_argument(
            "points_path",
            metavar="points_path",
            type=str,
            help="Ruta al archivo .csv con las cordenadas de los puntos, según el formato establecido",
        )
        parser.add_argument(
            "raster_path",
            metavar="raster_path",
            type=str,
            help="Ruta al archivo del raster en formato .tif",
        )
        parser.add_argument(
            "shape_path",
            metavar="shape_path",
            type=str,
            help="Ruta de almacenamiento del archivo con la informacion del poligono",
        )
        parser.add_argument(
            "--classifier",
            default="ED",
            dest="classifier_tag",
            type=str,
            help="Clasificador a utilizar para realizar el proceso de region growing",
        )
        args = parser.parse_args()
        execute(
            points_path=args.points_path,
            raster_path=args.raster_path,
            shape_path=args.shape_path,
            classifier_tag=args.classifier_tag,
        )
        logging.info(
            f"El archivo Shapefile se ha creado con exito en: {args.shape_path}"
        )
    # Ejecucion por GUI
    else:
        root = tkinter.Tk()  # GUI para Python
        points_path = archivo_cargar([("CSV", "*.csv")])
        raster_path = archivo_cargar([("GeoTIFF", "*.tif")])
        shape_path = archivo_guardar([("Shapefile", "*.shp")])

        # Computar el proceso
        execute(points_path=points_path, raster_path=raster_path, shape_path=shape_path)
        logging.info(f"El archivo Shapefile se ha creado con exito en: {shape_path}")
