---
title: 4.1 Software installation
layout: default
parent: 4. Software
nav_order: 1
---
# Software installation

Supported Operating Systems
- Windows 10, Windows 11
- Ubuntu 16.04 or newer
- macOS Sonoma or newer

1. Create a Python environment using **uv** (recommend) or **venv**.
    - Using **uv** (Recommend)
        1. Install uv [[Link]](https://docs.astral.sh/uv/getting-started/installation/).
        2. Create a virtual environment
            ```bash
            uv venv glowtracker.venv
            ```
    - Using **venv**
        1. Create the environment
            ```bash
            python -m venv glowtracker.venv
            ```

2. Activate the environment
    ```bash
    source glowtracker.venv/Scripts/activate
    ```

3. Install GlowTracker
    You can choose to either install GlowTracker from a distributed Python package from PyPI or clone the git repository and run them locally.

    - Using `pip` to install from PyPI repository:
        ```bash
        uv pip install glowtracker
        ```
    - Or clone and run the package locally.
        1. Clone the pository
            ```bash
            git clone https://github.com/scholz-lab/GlowTracker.git
            ```
        2. Update the conda environment to download the dependencies
            ```bash
            cd Glowtracker;
            uv pip install -r pyproject.toml;
            ```


4. Install the **BASLER** pylon software and runtime library [[Link]](https://www.baslerweb.com/en/software/pylon/)
    - pylon Camera Software Suite
    - pylon runtime library

5. (Optional) Install **Zaber Launcher** for inspecting and updating stage firmware [[Link]](https://software.zaber.com/zaber-launcher/download)

6. After finished installation, the software can be started in serveral ways
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