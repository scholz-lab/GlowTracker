from setuptools import setup

setup(
    name='glowtracker',
    version='1.0.0',
    description='GlowTracker is a macroscope tracking application that has the capability of tracking a small animal in bright field, single or dual epi-fluorescence imaging.',
    url='https://github.com/scholz-lab/GlowTracker',
    author='Monika Scholz',
    author_email='monika.scholz@mpinb.mpg.de',
    packages=['glowtracker'],
    install_requires=[
        "kivy>=2.2.1",
        "numpy>=1.25.1",
        "matplotlib>=3.7.2",
        "zaber-motion>=4.2.0",
        "pypylon>=2.2.0",
        "opencv-python>=4.7.0",
        "scipy>=1.11.1",
        "scikit-image>=0.21.0",
        "itk-elastix",
    ],
    keywords=[
        'python',
        'research',
        'macroscope',
        'tracking',
        'Zaber',
        'Basler',
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'Programming Language :: Python'
        'Operating System :: POSIX :: Linux',
        'Operating System :: Microsoft :: Windows',
    ],
)
