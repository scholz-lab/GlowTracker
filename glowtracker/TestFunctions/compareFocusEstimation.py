from typing import List
import cv2
import numpy as np
import os
from PIL import Image
import pandas as pd
from datetime import datetime
import re

FOCUS_MODE_DICT = {
    0: 'Variance of Laplace',
    1: 'Tenengrad',
    2: "Brenner's",
    3: 'Energy of Laplacian',
    4: 'Modified Laplacian',
    5: 'Sum of High-Frequency DCT Coefficients'
}

def estimateFocus(image: np.ndarray, mode: int) -> float:
    
    estimatedFocus = 0

    if mode == 0:

        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        estimatedFocus = laplacian.var()

    elif mode == 1:
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        gradient_magnitude = gx**2 + gy**2
        estimatedFocus = np.mean(gradient_magnitude)

    elif mode == 2:
        shifted = np.roll(image, -2, axis=1)
        diff = (image - shifted)**2
        estimatedFocus = np.sum(diff)
    
    elif mode == 3:
        lap = cv2.Laplacian(image, cv2.CV_64F)
        estimatedFocus = np.sum(np.abs(lap))
    
    elif mode == 4:
        mlap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
        estimatedFocus = np.sum(np.abs(mlap))
    
    elif mode == 5:
        resized = cv2.resize(image, (32, 32))  # Small for fast DCT
        dct = cv2.dct(np.float32(resized))
        hf_coeffs = dct[8:, 8:]  # Keep only high-freq block
        estimatedFocus = np.sum(np.abs(hf_coeffs))

    return float(estimatedFocus)



def read_tiff_images_chronologically(directory_path):
    arrays = []

    # List all TIFF files
    file_names = [
        f for f in os.listdir(directory_path)
        if f.lower().endswith('.tiff') or f.lower().endswith('.tif')
    ]

    # Function to extract datetime from filename
    def extract_datetime(filename):
        match = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d+)", filename)
        if match:
            timestamp_str = match.group(1)
            try:
                # Remove microseconds part (if any), or split as needed
                base_time, micro = timestamp_str.rsplit('-', 1)
                dt = datetime.strptime(base_time, "%Y-%m-%d-%H-%M-%S")
                micro = int(micro)
                return dt.replace(microsecond=micro)
            except Exception as e:
                print(f"Error parsing datetime in {filename}: {e}")
        return datetime.min  # fallback: send to start of sort

    # Sort files by extracted datetime
    sorted_files = sorted(file_names, key=extract_datetime)

    # Load images as NumPy arrays
    for file_name in sorted_files:
        full_path = os.path.join(directory_path, file_name)
        try:
            with Image.open(full_path) as img:
                arrays.append(np.array(img))
        except Exception as e:
            print(f"Failed to load {file_name}: {e}")

    return arrays


def read_arrays_from_directory(directory_path: str):
    arrays = []
    
    # Get all file names and sort them numerically
    file_names = os.listdir(directory_path)
    sorted_files = sorted(
        file_names,
        key=lambda name: int(os.path.splitext(name)[0])
    )

    # Read each file and convert to numpy array
    for file_name in sorted_files:

        full_path = os.path.join(directory_path, file_name)
        
        try:
            with Image.open(full_path) as img:
                array = np.array(img)
                arrays.append(array)
                
        except Exception as e:
            print(f"Failed to load {file_name}: {e}")
    
    return arrays


if __name__ == '__main__':

    imageFolderPath = 'C:/Workspace/GlowTracker/glowtracker/record/numba9'

    # images = read_arrays_from_directory(imageFolderPath)
    images = read_tiff_images_chronologically(imageFolderPath)

    # Create a DataFrame filled with zeros
    df = pd.DataFrame(0, index= range(len(images)), columns= FOCUS_MODE_DICT.items())
    
    for colIndex, colName in enumerate(df.columns):

        print(f'Processin: {FOCUS_MODE_DICT[colIndex]}')

        for rowIndex, image in enumerate(images):

            estimatedFocus = estimateFocus(image, mode= colIndex)

            df.iloc[rowIndex, colIndex] = estimatedFocus
            
    df.to_csv("estimatedFocus.csv", index=False)
    