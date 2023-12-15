---
title: 4.1 Software installation
layout: default
parent: 4. Software
nav_order: 1
---
# Software installation

### Install the correct environment
1. Install **Conda** [[Link]](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
2. Create an environment
    - Using **Mamba** (faster, recommended)
        1. Install Mamba from [[Link]](https://mamba.readthedocs.io/en/latest/installation.html)
        2. Create environment: 
            ```bash 
            mamba env update -n macroscope --file StageEnvironment.yml
            ```
    - Using **Conda**
        ```bash 
        conda env create --file StageEnvironment.yml
        ```

3. Activate the environment: 
    ```bash
    conda activate macroscope
    ```

4. Install the **BASLER** package
    1. Install pylon software from BASLER [[Link]](https://www.baslerweb.com/en/software/pylon/)
        - pylon Camera Software Suite
        - pylon runtime library

5. (Optional) Install **Zaber Launcher** for inspecting and updating stage firmware [[Link]](https://software.zaber.com/zaber-launcher/download)

6. Once the installation is finished, the software can be started by
    ```bash
    python GlowTracker.py
    ```
