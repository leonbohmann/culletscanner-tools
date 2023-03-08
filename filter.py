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
import cv2
import matplotlib.pyplot as plt
import shutil
import re
from paddleocr import PaddleOCR
from tqdm import tqdm

from imageOperations import *
# Script for converting CulletScanner Images
#
# 1. Input:     Image Source folder containing all debug images
# 2. Analyze:   Group images by debug ID (start of file name)
# 3. ID:        OCR of Image to find unique specimen ID and apply that ID to the whole group
# 4. Crop:      Crop the image and save the file with the found ID parameters (thickness, residual stress, id)
# 

residual_stresses = {
    
}

def get_unique_file_id(file):
    """Get the first 4 numbers of cullet scanner file name.
    Format is: [ID] [Date] ([img_type]).[ext]

    Args:
        file (str): Full file path.

    Returns:
        str: File identifier.
    """
    return os.path.basename(file)[:4]

def get_img_type(file):
    """Extracts an image type of input file name.

    Args:
        file (str): File name or file path.

    Returns:
        str: Type of Image (BrightPolarizedGreen, Blue, Transmission)
    """
    name = os.path.basename(file)
    match = re.search(r"\[(.*)\]", name)
    if match:
        return match.group(1)
    
    return "unknown"
  
def raiseIfWrongArea(area):
    """Check found pane pixel area.

    Args:
        area (float): Pixel Area of pane.

    Raises:
        Exception: Raises exception, if area is either too large or too small!
    """
    if area < 15_000_000 or area > 17_000_000:
        raise CropException(f"Area {area} can't be a pane!")

def plot(img):
    plt.figure()
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([])
    plt.yticks([])
    plt.show()

def get_series_box(img0, strength = 1):
    """Perform image manipulation on image to find regions of interest (roi).

    Args:
        img0 (Image): Image with possible ROIs.
        strength (int, optional): Strength of image manipulation. Defaults to 1.

    Returns:
        List<x,y,w,h,>: List with ROIs.
    """
    # Convert the image to grayscale
    im = img0.copy()
    gray = to_gray(img0)
    
    thresh = gray
    #plot(gray)
    # _, thresh = cv2.threshold(thresh, 180, 255, cv2.THRESH_BINARY)
    # thresh = cv2.GaussianBlur(thresh, (3,3), 0)
    _, thresh = cv2.threshold(thresh, 130-strength*5, 255, cv2.THRESH_BINARY)
    # ret, thresh = cv2.threshold(thresh, 60, 255, cv2.THRESH_BINARY)
    #thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)        
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # thresh = cv2.erode(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=strength)        
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    textboxes = []
    
    
    # Iterate through contours
    for cnt in contours:
        # Find the bounding rectangle around the contour
        x, y, w, h = cv2.boundingRect(cnt)
        textboxes.append((x, y, w, h))
        cv2.rectangle(im, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.rectangle(thresh, (x,y), (x+w,y+h), (0,0,0), 2)
    
    # plot(thresh)
    # plot(im)
    
    
    return textboxes

def validate_possible_series_id(id: str) -> str | None:
    """Validate the found ID for the series. Should take action if the ID matches an entry in the
    series matrix!

    Args:
        id (str): The ID found by the OCR process.

    Returns:
        str | None: A valid ID or None, if nothing matched.
    """
    if len(id) > 0 and re.match(r"(\d+)\.(\d+)\.(\d+)", id):
        return id
    
    return None

def ocr_on_crop(img, reader: PaddleOCR):
    """Perform an OCR Operation on a specific part of the source img.

    Args:
        img (Image): The image part to analyze.
        reader (PaddleOCR): Reader for OCRing.

    Returns:
        (bool, string): (Valid text found?, Found Text)
    """
    text = reader.ocr(img, cls = False, det=False, )[0]
    # text = reader.readtext(roi, detail=0, allowlist = "0123456789.", width_ths = 10, add_margin = 10)
    # plot(img)
    if len(text) > 0:
        # print(text)
        # plot(img)
        pass
        
    text = validate_possible_series_id(text[0][0])
        
    if text != None:
        return (True, text)
    
    return (False, "")

def get_series_id(img0, reader: PaddleOCR, showRoi = False, strength = 1, maxStrength = 15):
    """Uses an OCR Reader to analyze a given image of a glass pane.

    Args:
        img0 (Image): Image of the full glass pane.
        reader (easyOCR.Reader): Reader.
        showRoi (bool, optional): Display the ROI. Defaults to False.
        strength (int, optional): Strength of image manipulation. Gets increased to maxStrength when current 
                                    strength didn't yield a result. Defaults to 1.
        maxStrength(int, optional): Maximum strength. Defines the upper bounds of image manipulation strength. Defaults to 15.

    Returns:
        str: Found OCR string.
    """
    
    # find mark
    # x0, y0, w0, h0 = 0, 0, img0.shape[1], img0.shape[0]  # full image
    x0, y0, w0, h0 = 500, 5, 500, 100  # upper right corner
    roi = img0[y0:y0+h0, x0:x0+w0]
    
    # # try the whole picture first
    # (valid, text) = ocr_on_crop(prepare_ocr(roi, 2, strength), reader)
    # if valid:
    #     return text
    
    minSize = (40,40)
    maxSize = (400,400)
    
    possible_textboxes = get_series_box(roi, strength)
    found_texts = []
    
    for box in possible_textboxes:
        x, y, w, h = box
        roi = img0[y+y0:y+y0+h, x+x0:x+x0+w]
        
        
        if roi.shape[0] < minSize[0] or roi.shape[0] > maxSize[0] \
            or roi.shape[1] < minSize[1] or roi.shape[1] > maxSize[1]: continue        

        # roi = optimizeImgForPerspectiveCrop(roi)
        roi = prepare_ocr(roi, 2, strength)
        
        # perform ocr on roi
        (valid, text) = ocr_on_crop(roi, reader)
        if valid: 
            return text
        
        
     
    if strength < maxStrength:
        return get_series_id(img0, reader, showRoi, strength + 1)
    else:
        return "unknown"
    # if showRoi:
    #     plt.figure()
    #     plt.imshow(roi, cmap = 'gray', interpolation = 'bicubic')
    #     plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #     plt.show()
    
    # return reader.readtext(roi,detail=0)        

#
## FILTER PROCEDURE
#
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    
    parser.description = descr
    parser.add_argument("-i", "--input-dir", help="Input directory with unsorted scan output image files.", type=str, default=r"a:\Nextcloud\Forschung\Scans")    
    parser.add_argument("-o", "--output-dir", help="Directory to place the filtered images in.", type=str, default=r"filtered")    
    parser.add_argument("-r", "--rotate", help="Rotate images.", type=bool, default=True)    

    args = parser.parse_args()
    
    root_dir = args.input_dir
    filtered_dir = args.output_dir
    cropped_renamed_dir = os.path.join(filtered_dir, "cropped")
    rotate = args.rotate
    
    # rotate = False
    
    # create output dir
    if not os.path.exists(filtered_dir):
        os.makedirs(filtered_dir)
        
    # create cropped img dir
    if not os.path.exists(cropped_renamed_dir):
        os.makedirs(cropped_renamed_dir)
    
    # create ocr reader
    # load once only in memory
    reader = PaddleOCR(lang='en', show_log=False, use_mp=True, use_gpu = True, ) 
    
    # analyze only these extensions (used to filter out unwanted files)
    extensions = [".bmp", ".zip"]
    # f = os.listdir(root_dir)
    # find all files in root dir
    files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isfile(os.path.join(root_dir, f)) and any(f.endswith(ext) for ext in extensions)]
    
    # sort files for file name
    sorted_files = sorted(files, key=get_unique_file_id)
    # group files for file id
    grouped_files = {k: list(g) for k, g in groupby(sorted_files, key=get_unique_file_id)}

    for key, group in tqdm(grouped_files.items()):

                
        # get bitmaps from group
        img0_path = next(file for file in group if file.endswith(".bmp") and "Blue" in file)
        # load image and perform OCR to find specimen Identifier ([thickness].[residual_stress].[boundary].[ID])
        img0 = cv2.imread(img0_path)


        try:
            maxArea, img0 = crop_perspective(img0)
            if rotate:
                img0 = rotate_img(img0, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            raiseIfWrongArea(maxArea)

            # print(f"Series {key} has visible pane. Area={maxArea}")
            
            # this combines the three images
            # imgs = [cv2.imread(file) for file in group if file.endswith(".bmp")]
            # img0 = combineImgs(imgs)
                
            # plot(img0)
                
            # find group name by running OCR on image            
            # get series id from image
            groupkey = get_series_id(img0, reader, True)
            # print(f"\t> Series: {groupkey}")
            
            if groupkey == "unknown":
                groupkey = key


            for file in group:       
                filename =  os.path.basename(file)                
                dest = os.path.join(filtered_dir, filename)
                
                shutil.copy2(file, dest)
                shutil.copystat(file, dest)
                
                
                if filename.endswith(".bmp"):
                    # get img type
                    imgType = get_img_type(filename)
                    imgKey = groupkey
                    
                    filename = f"{imgKey}-{imgType}.bmp"
                    
                    im = cv2.imread(file)
                    _,im = crop_perspective(im, 4000)
                    
                    cropped_img_path = os.path.join(cropped_renamed_dir, filename)
                    cv2.imwrite(cropped_img_path , im)
                    shutil.copystat(file, cropped_img_path)
                    
        except CropException:
            pass
        
        except Exception as e:
            print(f" << Exception in Series {key}: >>")
            print(str(e))
        
        
"""
TODO:
- Eingabe-Bilder immer richtig rotieren (um 90° im Uhrzeigersinn, da im CulletScanner Schrift parallel zur Schubrichtung)
- Einheitliche Beschriftung wählen und auf kommende Scheiben übertragen

"""