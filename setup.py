from setuptools import setup, find_packages

with open("README.md") as readme_file:
    README = readme_file.read()

setup_args = dict(
    name="region_grow",
    version="1.0.2",
    description="Creates a new polygon locally given a set of points using a region grow algorithm approach and a satellite image in GeoTIFF format. The polygon output format is ESRI Shapefile",
    long_description_content_type="text/markdown",
    long_description=README,
    license="MIT",
    packages=find_packages(),
    author="Geovanny Andres Gonzalez",
    author_email="ga.gonzalezr@uniandes.edu.co",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    keywords="raster region grow",
    url="https://github.com/ggonzr/region_grow",
    python_requires=">=3.7",
)

install_requires = ["Shapely", "numpy", "pandas", "scipy", "rasterio", "geopandas"]

if __name__ == "__main__":
    setup(**setup_args, install_requires=install_requires)
