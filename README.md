# culletscanner-tools
Tools for CulletScanner Debug output images.


## Tool: Filter
This tool can filter input images (see test_imgs) and check, if a glass pane is detected.

After Cropping the glass pane out of the scan image, the cropped result will be placed in the output directory next to the copied source scan file.

## Tool: Mark
Does the same as filter, but will rename the cropped and copied files according to a marking on the glass pane.

If the glass pane is marked with a code (i.e. 4.70.1) in the upper right corner, the resulting files will be renamed after that.