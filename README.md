## Overview
This repository contains all the code required to generate figures for Kinsky, Orlin et al. (2024)

## Setup
 - This code utilizes several functions from https://github.com/nkinsky/FearReinstatement and https://github.com/diba-lab/NeuroPy. Clone the `FearReinstatement` and `NeuroPy` repos in the same parent directory as the `Eraser` repo.
 - Install a conda/mamba environment using `mamba env create -f eraser.yml`. You may need to add a few modules manually, e.g. `mamba install -c conda-forge notebook` to use Jupyter Notebook.
 - Download processed neural and behavioral data for each mouse (a permanent repository is being set up).
    - Update `SessionDirectories.csv` file with the location of individual mouse data on your computer.
    - Update `get_comp_name` function in `eraser_reference.py` with locations of data folder and directory to plot things into 
 - Download group data
   - Update location of group data in `subjects.py` file
   - (You may have to add this path to a few notebooks directly as well)
 - Run notebooks in `Notebooks` folder to generate figure panels.
