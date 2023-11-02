---
title: 3.1 Software installation
layout: default
parent: 3. Software
nav_order: 1
---
# Software installation

### Install the correct environment
1. Install conda from <a href="https://conda.io/projects/conda/en/latest/user-guide/install/index.html">conda webpage</a>
2. Create environment
    - Using Conda
        1. Create environment: `conda env create --file StageEnvironment.yml`
    - Using Mamba *(faster, recommended)*
        1. Install Mamba from <a href="https://mamba.readthedocs.io/en/latest/installation.html">Mamba webpage</a>
        2. Create environment: `mamba env update -n macroscope --file StageEnvironment.yml`

3. Activate the environment: `conda activate macroscope`

4. Install the BASLER package. To get the BASLER package running on Windows do the following:
    1. Install pylon (software from BASLER) including all interfaces
    2. Get the pypylon package from <a href="https://github.com/basler/pypylon">GitHub</a> 
    3. Use pip to install the package using the proper wheel (in my case pypylon-1.5.4-cp37-cp37m-win_amd64.whl)
    4. Run the following command using the comandline within the package folder:<br> `pip install pypylon-1.5.4-cp37-cp37m-win_amd64.whl`
