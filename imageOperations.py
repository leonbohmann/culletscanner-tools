import cv2
import numpy as np

def crop_rectangle(src, cntr, useMask = False, drawContour = False, drawColor = (0,0,255), drawStroke = 1):
    """Crops the normal rectangle around a contour from an image.

    Args:
        src (image): The source image.
        cntr (contour): The contour to crop out.
        useMask (bool, optional): 
            States, wether the mask should be used as cropped image.
            Helpful, if contour should be isolated (set to True). 
            Defaults to False.
        drawContour (bool, optional):
            States, wether the contour should be drawn into the crop.
        drawColor (tuple(3), optional):
            The drawing color, with which the contour would be drawn.

    Returns:
        image: A rectangle, cropped from either the src or the mask containing the cntr.

    """
    # copy source image
    source = src.copy()
    # create mask with contour as white (background is black)
    mask = np.zeros(source.shape,np.uint8)
    cv2.fillPoly(mask,pts=[cntr],color=(255,255,255))
    # find the bounding 
    x,y,w,h = cv2.boundingRect(cntr)


    # if not using mask, crop from source image
    if not useMask:                
        if drawContour: cv2.drawContours(source, [cntr],0,drawColor,drawStroke)
        cropped = source[y:y+h,x:x+w]
    # otherwise, crop from the mask
    else:
        if drawContour: cv2.drawContours(mask, [cntr],0,drawColor,drawStroke)
        cropped = mask[y:y+h,x:x+w]
    return cropped

def crop_rectangle_rot(src, contour):
    """Crops the rotated rectangle around a contour from an image.

    Args:
        src (image): The source image.
        contour (contour): The contour to crop out.

    Returns:
        image: A rotated rectangle containing the cntr.

    """
    # get minimum area rectangle from contour
    rect = cv2.minAreaRect(contour)
    # Get center, size, and angle from rect
    center, size, theta = rect    
    size = tuple([z * 0.997 for z in size])
    # Convert to int 
    center, size = tuple(map(int, center)), tuple(map(int, size))
    # Get rotation matrix for rectangle
    M = cv2.getRotationMatrix2D( center, theta, 1)
    # Perform rotation on src image
    dst = cv2.warpAffine(src, M, (src.shape[1] ,src.shape[0]))
    out = cv2.getRectSubPix(dst, size, center)
    
    return cv2.flip(out, 0)

def crop(im):
    """Crops an image to its largest containing bounds.

    Args:
        im (image): The image to be cropped.

    Returns:
        image: Cropped image which has no outer white bounds.
    """

    # convert image to grayscale
    orig = im.copy()
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    ret,im = cv2.threshold(im,127,255,0)

    # fetch contour information
    contour_info = []
    contours, _ = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # loop through contours and find their properties
    for c in contours:
        contour_info.append((
            c,
            cv2.isContourConvex(c),
            cv2.contourArea(c)
        ))
    # sort contours after their area
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    # take the second largest contour (this has to be the outer bounds of pane)
    max_contour = contour_info[1]
    # then, crop a rotated rectangle from the original image using that contour
    return crop_rectangle_rot(orig, max_contour[0])














def fourCornersSort(pts):
    """ Sort corners: top-left, bot-left, bot-right, top-right """
    # Difference and sum of x and y value
    # Inspired by http://www.pyimagesearch.com
    diff = np.diff(pts, axis=1)
    summ = pts.sum(axis=1)
    
    # Top-left point has smallest sum...
    # np.argmin() returns INDEX of min
    return np.array([pts[np.argmin(summ)],
                     pts[np.argmax(diff)],
                     pts[np.argmax(summ)],
                     pts[np.argmin(diff)]])


def contourOffset(cnt, offset):
    """ Offset contour, by 5px border """
    # Matrix addition
    cnt += offset
    
    # if value < 0 => replace it by 0
    cnt[cnt < 0] = 0
    return cnt

def rotate_img(img, rotCode):
    return cv2.rotate(img, rotCode)

def prepare_ocr(img0):
    #norm_img = np.zeros((img0.shape[0], img0.shape[1]))
    #img = cv2.normalize(img0, norm_img, 0, 255, cv2.NORM_MINMAX)
    img = cv2.resize(img0, (img0.shape[1] * 2, img0.shape[0] * 2))
    img = cv2.GaussianBlur(img, (5, 5), 3)
    img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)[1]
    return img

def crop_perspective(img):
    """Crops a given image to its containing pane bounds. Finds smallest pane countour with 4 corner points
    and aligns, rotates and scales the pane to fit a resulting image.

    Args:
        img (Image): Input image with a clearly visible glass pane.

    Raises:
        Exception: If the found countour does not have exactly 4 corner points it is not considered a glass pane.

    Returns:
        img: A cropped image which only contains the glass pane.
    """
    # convert to gray
    im = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # apply gaussian blur to image to we can get rid of some noise
    im = cv2.GaussianBlur(im, (5,5), 5)
    # restore original image by thresholding
    _,im = cv2.threshold(im,127,255,0)


    # fetch contour information
    contour_info = []
    contours, _ = cv2.findContours(im, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # loop through contours and find their properties
    for cnt in contours:
        contour_info.append((
            cnt,
            cv2.isContourConvex(cnt),
            cv2.contourArea(cnt)
        ))

    # sort contours after their area
    contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
    # take the second largest contour (this has to be the outer bounds of pane)
    max_contour = contour_info[1][0]
    
    # stencil will have white for every splinter
    stencil = np.zeros(img.shape).astype(img.dtype)    
    # dont fill the pane and scanner boundaries (:-2)
    cv2.fillPoly(stencil, contours[:-2], [255, 255, 255])
    # take original image, where stencil is white
    img = cv2.bitwise_and(img,stencil)

    # Simplify contour
    perimeter = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.03 * perimeter, True)

    # Page has 4 corners and it is convex
    # Page area must be bigger than maxAreaFound 
    if (len(approx) == 4 and
            cv2.isContourConvex(approx)):

        maxAreaFound = cv2.contourArea(approx)
        pageContour = approx
    else:
        raise Exception("Pane boundary could not be found.")

    # Sort and offset corners
    pageContour = fourCornersSort(pageContour[:, 0])
    #pageContour = contourOffset(pageContour, (-3,-3))

    # Create target points
    width=height=1000
    tPoints = np.array([[0, 0],
                    [0, height],
                    [width, height],
                    [width, 0]], np.float32)
    # source points are contour corners
    sPoints = pageContour

    # getPerspectiveTransform() needs float32
    if sPoints.dtype != np.float32:
        sPoints = sPoints.astype(np.float32)
    # Warping perspective
    M = cv2.getPerspectiveTransform(sPoints, tPoints)     

    # then, warp original image to (w*h)-image
    persp = cv2.warpPerspective(img, M, (int(width), int(height)))
    # restore original image by thresholding
    # _,persp = cv2.threshold(persp,127,255,0)
    # and return the transformed image
    return (maxAreaFound, persp)