import argparse
import os
from itertools import groupby
import tempfile
from easyocr import Reader
import cv2
import matplotlib.pyplot as plt

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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i", "--input-dir", help="Input directory with unsorted scan output image files.", type=str, default=r"d:\Forschung\Glasbruch\Versuche\Code_v2\test_imgs")    

    args = parser.parse_args()
    
    root_dir = args.input_dir
    
    # creating ocr reader
    reader = Reader(['en'],gpu = True) # load once only in memory.
    
    # create output dir
    if not os.path.exists("out"):
        os.makedirs("out")  
    
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
        img0 = crop_perspective(img0)
        img0 = rotate_img(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)        


        x, y, w, h = img0.shape[0]-200, 0, 200, 100  # x, y, width, height
        roi = img0[y:y+h, x:x+w]
        roi = prepare_ocr(roi)
        plt.figure()
        plt.imshow(roi, cmap = 'gray', interpolation = 'bicubic')
        plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        plt.show()
        
        r_easy_ocr=reader.readtext(roi,detail=0)
        print(r_easy_ocr)
        # results = pytesseract.image_to_data(roi, output_type=Output.DICT)
        
        # for i in range(0, len(results["text"])):
        #     x = results["left"][i]
        #     y = results["top"][i]

        #     w = results["width"][i]
        #     h = results["height"][i]

        #     text = results["text"][i]
        #     conf = int(results["conf"][i])

        #     text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        #     cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv2.putText(roi, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)
        
        # cv2.imshow("results", roi)
        # specimen_name = f"{thickness}.{residual_stress}.{specimen_id}"
        # print(f"Copy specimen: {specimen_name}")
        
        # for file in group:            
        #     _, ext = os.path.splitext(file)
        #     cpy_path = ""
            
        #     # image files have "green", "blue" or "transmission"
        #     if( ext == ".bmp" ):
        #         type = get_img_type(file)
        #         cpy_path = f"out/{specimen_name}_p-{type.lower()}{ext}"
        #     else:
        #         cpy_path = f"out/{specimen_name}_d{ext}"

            
        #     shutil.copy2(file, cpy_path)
            
        # # increment specimen id
        # specimen_id += 1
        
    