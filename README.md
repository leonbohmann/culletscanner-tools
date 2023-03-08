# culletscanner-tools
Tools for CulletScanner Debug output images.

##  PaddleOCR

PaddleOCR gives better results for handwritten series IDs. To use PaddleOCR:

* Switch to `paddle` branch
* Create virtual environment
* `pip install -r paddle_requirements.txt`
* Run `python filter.py -h`


## Basic usage


```bat
usage: filter.py [-h] [-i INPUT_DIR] [-o OUTPUT_DIR]

Use this to filter debug output from CulletScanner.
Debug images often contain nonsense, this helps to automatically analyze debug images.

Inputs:
    -i      Input Directory containing all debug output from CS.
    -o      Output Directory. This is where the results are put.

What it does:
    1.      Sort the input directory for the cullet scanner output ID (first 4 digits in file name)
    2.      Group all files starting with output ID into groups
    3.      Of each group, find first bitmap file and try to extract pane
    4.      Check, if the pane area (in pixels) is approximately 4000x4000
    5.      Save all files in that group in output folder and place cropped bitmaps into output-folder/cropped.

options:
  -h, --help            show this help message and exit
  -i INPUT_DIR, --input-dir INPUT_DIR
                        Input directory with unsorted scan output image files.
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to place the filtered images in.
```

This tool can filter input images (see test_imgs) and check, if a glass pane is detected.

After Cropping the glass pane out of the scan image, the cropped result will be placed in the output directory next to the copied source scan file.

When scanning the output images, the tool uses OCR techniques to try and identify a handwritten Series ID, which must be placed somewhere in the top right corner of the glass. If the recognition fails, the input debug id from cullet scanner file name is used.
