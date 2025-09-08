# GlowTracker

<div style="display: flex; justify-content: center; align-items: center;">
    <table style="width: 80%; border: none;">
        <colgroup>
            <col style="width: 30%;">
        </colgroup>
        <tr>
            <td>
                <img src="glowtracker/images/glowtracker_logo.png" alt="GlowTracker Logo" display="block">
            </td>
            <td style="text-align: left; vertical-align: top;">   
                GlowTracker is a microscope tracking application that has the capability of tracking a small animal in bright-field, single or dual epi-fluorescence imaging. The application interface provides controls over linear Zaber stage movement and Basler camera properties. Please visit the documentation website on how to build the setup from scratch and how to operate the software at <a href="https://scholz-lab.github.io/GlowTracker/">https://scholz-lab.github.io/GlowTracker/</a>.
            </td>
        </tr>
    </table>
</div>

## Getting started
### Software Setup
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

5. After finished installation, the software can be started in several ways
    - If you have installed it via pip
        ```bash
        python -m glowtracker
        ```
        or simply
        ```bash
        glowtracker
        ```
    - If you have installed it by cloning the package and running them locally
        ```bash
        python glowtracker/__main__.py
        ```

### Device Setup
#### Stage
In **Settings > Stage > Stage serial port**, specify the connection port name to your Stage. In Windows, this is usually `port = COM3`. And `/dev/ttyUSB0` for Linux.

#### Camera
In **Settings > Camera > Default camera settings**, specify the path to your pylon default camera setting. This is a `.pfs` file that can be obtained from the [pylon Viewer](https://www.baslerweb.com/en/software/pylon/pylon-viewer/) software that you have downloaded.

## GUI overview
<img alt="annotated GUI" src="glowtracker/images/gui_annotation.png" width="1250">

## Code overview

The application is based on the Kivy framework which connects to the microscope hardware.
The GUI functionality is implemented mostly in the Kivy file, whereas device functionality is relayed to specific modules.

## Known issues

- On Linux systems, accessing serial ports needs to be allowed for the user running the GUI. In Ubuntu and similar systems, the user has to be added to the group 'dialout'.

## Supported Operating Systems
- Windows 10, Windows 11
- Ubuntu 16.04 or newer
- macOS Sonoma or newer
