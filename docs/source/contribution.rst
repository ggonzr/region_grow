Contributing
=====================================================

Create your own classifier method
------------------------------------------------------

If you want to experiment with new algorithms to grow a region, you only need to write 
your own class which extends the Classifier abstract class and implement methods **fit()**, **predict()**.

Then, depeding if you want to generate the polygon using and aproximate area or not, you have to incorporate
a tag to instanciate this classifier for region grow execution. If you want to use the area fill the 
function ** grow_balanced_region() ** with a new case. For instance:

.. code-block:: ipython3

    def grow_balanced_region(
        classifier_tag: str,
        pixels_indexes: np.ndarray,
        pixels_df: pd.DataFrame,
        img_array: np.ndarray,
        raster_path: str,
        polygon_area: float,
        steps: int = 4,
    ):
        if classifier_tag == "EDR":
            pixels_selected, created_polygon = grow_edr_region(
                classifier_tag=classifier_tag,
                pixels_indexes=pixels_indexes,
                pixels_df=pixels_df,
                img_array=img_array,
                raster_path=raster_path,
                polygon_area=polygon_area,
                steps=steps,
            )
        # .....
        elif classifier_tag == "<YOUR_NEW_ALGO>":
            pixels_selected, created_polygon = grow_new_algo_region()

Finally use the new tag when you call the **execute_with_area(classifier_tag="<YOUR_NEW_ALGO>")** function

.. note:: 
    This process will use the region grow class to create the polygon locally. If you want use a global method **please override the region grow class** and **check_hood()** function

On the other hand, if you want to create the polygon without knowing an aproximate area, you only need to
add the tag to the **selected_classifier()** function and use it when calling the **execute()** function
