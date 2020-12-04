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
conda install -c anaconda pillow matplotlib ipywidgets pandas
conda install -c conda-forge mplcursors opencv scikit-image gdal spectral notebook jupyter_contrib_nbextensions piexif tqdm
```

#### For cell collapsing in JN page, create a terminal page in JN and run

```
jupyter contrib nbextension install --user
jupyter nbextension enable codefolding/main
```

## How To Use

### Output File Locations

All data used/generated should be stored to [_output_](output/) folder.
* Field images used as input should be put to [_output/field_image_](output/field_image/)
* Plot locations from [_Image Process 1.ipynb_](Image_Process_1-Set_Plots_V2.ipynb) should be put to [_output/plot_location_](output/plot_location/)
* Indices of plots (in the form of a table) from [_Image Process 2.ipynb_](Image_Process_2-Calculate_Indices.ipynb) should be put to [_output/final_result_](output/final_result)


### Run code

Comments in the first line of each cell will tell you what this cell does.  

**Step 1**
Run _Image Process 1_, set plot locations

**Step 2**
Run _Image Process 2_, calculate indices

## Authors

* **Jiahe Li (LJ)** - *Initial Work* - [GitHub-LJ](https://github.com/LJ-JiaheLi)

## License

This project is lecensed under the MIT License - see the [LICENSE](LICENSE) file for details
