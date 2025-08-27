---
title: 4.1 Software installation
layout: default
parent: 4. Software
nav_order: 1
---
# Software installation

Supported Operating Systems
- Windows 10, Windows 11
- Ubuntu 16.04+
- macOS Sonoma+

1. Create and activate a python environment using **venv** or **Conda**.
    - Using **venv**

        GlowTracker requires python version of at least 3.10. If you do not have one, please download from [[Python]](https://www.python.org/downloads/).
        1. Create the environment
            ```bash
            python -m venv glowtracker
            ```
        2. Activate the environment
            ```bash
            source glowtracker/Scripts/activate
            ```
    - Using **Conda**
        1. Install **Conda** [[Link]](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
        2. Use `conda` to create a new python environment named `glowtracker`.
            ```bash
            conda create -n glowtracker "python>=3.10,<3.12"
            ```
        3. Activate the `glowtracker` environment 
            ```bash
            conda activate glowtracker
            ```
2. Install the `glowtracker` package.
    - Using `pip` to install from PyPI repository:
        ```bash
        python -m pip install glowtracker
        ```
    - Or clone and run the package locally.
        1. Clone the pository
            ```bash
            git clone https://github.com/scholz-lab/GlowTracker.git
            ```
        2. Update the conda environment to download the dependencies
            ```bash
            cd Glowtracker;
            conda env update --file glowtracker/StageEnvironment.yml --prune
            ```

3. Install the **BASLER** pylon software and runtime library [[Link]](https://www.baslerweb.com/en/software/pylon/)
    - pylon Camera Software Suite
    - pylon runtime library

4. (Optional) Install **Zaber Launcher** for inspecting and updating stage firmware [[Link]](https://software.zaber.com/zaber-launcher/download)

5. After finished installation, the software can be started in serveral ways
    - If you have installed via pip
        ```bash
        python -m glowtracker
        ```
        or simply
        ```bash
        glowtracker
        ```
    - If you have installed by cloning the package and running them locally
        ```bash
        python glowtracker/__main__.py
        ```