# RGB-IR Field Image Process

This project allows you to calculate different indices of different areas from a field image.  

## Prerequisites

### Software

What's been listed below are the software I used specifically in this project. You may use other compatible software preferred.
Some related installation and deployment instrunctions will not be included in next section. Please see official websites provided below for related tutorial.  

```
Anaconda
Jupyter Notebook(JN)
```

### Python Packages

```
conda install -c anaconda pillow matplotlib ipywidgets pandas xlrd
conda install -c conda-forge opencv scikit-image gdal spectral piexif tqdm notebook jupyter_contrib_nbextensions 
```

#### For cell collapsing in JN page, create a terminal page in JN and run

```
jupyter contrib nbextension install --user
jupyter nbextension enable codefolding/main
```

## How To Use

### File Explained

* [_Image_Processing_1-Set_Plots.ipynb_](Image_Processing_1-Set_Plots.ipynb) lets you set plot locations, name plots(auto from a file or by hand) and adding notes to plots.
* MTP stands for **M**anual **T**ie **P**oint. Currently if you want to apply plot masks from one flight to another, you need to set MTPs for both the image of reference and the image to be calibrated by [_Image_Processing_X-Set_MTP.ipynb_](Image_Processing_X-Set_MTP.ipynb) and then recalculate plot locations for the image to be calibrated by [_Image_Processing_X-Plot_Loc_Transform.ipynb_](Image_Processing_X-Plot_Loc_Transform.ipynb).
* After having correct plot location(either from [_Image_Processing_1_](Image_Processing_1-Set_Plots.ipynb) or [_Image_Processing_X_](Image_Processing_X-Set_MTP.ipynb)), you then use [_Image_Processing_2-Calculate_Indices.ipynb_](Image_Processing_2-Calculate_Indices.ipynb) to generate indices.

### Data Directory Explained

All data used/generated should be stored to [_data_](data/).
* Field images used as input should be put to [_data/field_image_](data/field_image/).
* To load plot name automatically from _.csv_ file, the files need to be stored to [_data/plot_name_csv_](data/plot_name_csv).
* Plot locations from [_Image_Processing_1-Set_Plots.ipynb_](Image_Processing_1-Set_Plots.ipynb) should be put to [_data/plot_location_](data/plot_location/).
* MTP files from [_Image_Processing_X-Set_MTP.ipynb_](Image_Processing_X-Set_MTP.ipynb) shoude be put to [_data/mtp_](data/mtp/).
* Indices of plots (in the form of a table) from [_Image_Processing_2-Calculate_Indices.ipynb_](Image_Processing_2-Calculate_Indices.ipynb) should be put to [_data/final_result_](data/final_result).

## Authors

* **Jiahe Li (LJ)** - *Initial Work* - [GitHub-LJ](https://github.com/LJ-JiaheLi)

## License

This project is lecensed under the MIT License - see the [LICENSE](LICENSE) file for details
