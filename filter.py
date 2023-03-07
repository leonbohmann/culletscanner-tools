descr = """Use this to filter debug output from CulletScanner.
Debug images often contain nonsense, this helps to automatically analyze debug images.

Inputs:
    -i      Input Directory containing all debug output from CS.
    -o      Output Directory. This is where the results are put.

What it does:
    1.      Sort the input directory for the cullet scanner output ID (first 4 digits in file name)
    2.      Group all files starting with output ID into groups
    3.      Of each group, find first bitmap file and try to extract pane
    4.      Check, if the pane area (in pixels) is approximately 4000x4000
    5.      Save all files in that group in output folder and place cropped bitmaps into output-folder/cropped."""

import argparse
from argparse import RawTextHelpFormatter
import os
from itertools import groupby
import tempfile
import cv2
import matplotlib.pyplot as plt
import shutil

from imageOperations import *
# Script for converting CulletScanner Images
#
# 1. Input:     Image Source folder containing all debug images
# 2. Analyze:   Group images by debug ID (start of file name)
# 3. ID:        OCR of Image to find unique specimen ID and apply that ID to the whole group
# 4. Crop:      Crop the image and save the file with the found ID parameters (thickness, residual stress, id)
# 

def get_unique_file_id(file):
    """Get the first 4 numbers of cullet scanner file name.
    Format is: [ID] [Date] ([img_type]).[ext]

    Args:
        file (str): Full file path.

    Returns:
        str: File identifier.
    """
    return os.path.basename(file)[:4]

  
def raiseIfWrongArea(area):
    """Check found pane pixel area.

    Args:
        area (float): Pixel Area of pane.

    Raises:
        Exception: Raises exception, if area is either too large or too small!
    """
    if area < 15_000_000 or area > 17_000_000:
        raise Exception(f"Area {area} can't be a pane!")


#
## FILTER PROCEDURE
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    
    parser.description = descr
    parser.add_argument("-i", "--input-dir", help="Input directory with unsorted scan output image files.", type=str, default=r"a:\Nextcloud\Forschung\Scans")    
    parser.add_argument("-o", "--output-dir", help="Directory to place the filtered images in.", type=str, default=r"filtered")    

    args = parser.parse_args()
    
    root_dir = args.input_dir
    filtered_dir = args.output_dir
    destImg = os.path.join(filtered_dir, "cropped")
    
    # create output dir
    if not os.path.exists(filtered_dir):
        os.makedirs(filtered_dir)
        
    # create cropped img dir
    if not os.path.exists(destImg):
        os.makedirs(destImg)
    
    # analyze only these extensions (used to filter out unwanted files)
    extensions = [".bmp", ".zip"]
    # f = os.listdir(root_dir)
    # find all files in root dir
    files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and any(f.endswith(ext) for ext in extensions)]
    
    # sort files for file name
    sorted_files = sorted(files, key=get_unique_file_id)
    # group files for file id
    grouped_files = {k: list(g) for k, g in groupby(sorted_files, key=get_unique_file_id)}

    for key, group in grouped_files.items():        
        # find first file in group that has bmp extension
        img0_path = next(file for file in group if file.endswith(".bmp"))
        # load image and perform OCR to find specimen Identifier ([thickness].[residual_stress].[boundary].[ID])
        img0 = cv2.imread(img0_path)        
       
        try:
            maxArea, img0 = crop_perspective(img0)
            raiseIfWrongArea(maxArea)

            print(f"Series {key} has visible pane. Area={maxArea}")

            for file in group:       
                filename =  os.path.basename(file)    
                dest = os.path.join(filtered_dir, filename)
                shutil.copy2(file, dest)
                
                
                if filename.endswith(".bmp"):
                    im = cv2.imread(file)
                    _,im = crop_perspective(im)
                    im = rotate_img(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    cv2.imwrite(os.path.join(destImg, filename), im )
        except:
            pass
        
      