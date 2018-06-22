import json
import re

import cv2
import pyocr
# import numpy as np

from abyss_deep_learning.utils import cv2_to_Pil

class OcrRegions(object):
    def __init__(self, lang='eng', builder='text', preprocess=None, text_filter=None, **kwargs):
        """Create an OcrRegions object which provides methods to extract text with
        preprocessing and text filtering.
        
        Parameters
        ----------
        lang : str, optional
            Language, default 'eng'.
        builder : str, optional
            Allowable characters in the recognition algorithm.
            Either 'digit' or 'text'
        preprocess : function, optional
            Apply a preprocessing function on the image before OCR.
            Can use default function by passing preprocess=True, or give a custom 
            function of type: f(np.ndarray) -> np.ndarray.
        text_filter : str, optional
            Regexp string. Returns None if no match.
            Useful for filtering out false positives.
        """
        self.rois = []
        self.builder = pyocr.builders.DigitBuilder() if builder == 'digit' else pyocr.builders.WordBoxBuilder()
        self.lang = lang
        self.text_filter = text_filter
        self.preprocess = lambda x: x
        if preprocess:
            if preprocess is True:
                self.preprocess = OcrRegions.__preprocess
            else:
                self.preprocess = preprocess

    @staticmethod
    def __preprocess(image):
        image2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        cv2.medianBlur(image2, 3, image2)
        cv2.threshold(image2, 0, 255, cv2.THRESH_OTSU, dst=image2)
        cv2.dilate(image2, (1, 1), image2)
        cv2.erode(image2, (1, 1), image2)
        return image

    def read_image(self, image):
        """Extract text from an image.

        Parameters
        ----------
        image : numpy.ndarray
            An RGB or single channel image containing legible text.

        Returns
        -------
        text: str
            The text found in the image.
        """
        if self.preprocess is not None:
            image = self.preprocess(image)
        text = pyocr.tesseract.image_to_string(
                cv2_to_Pil(image), lang=self.lang, builder=self.builder)
        if self.text_filter and not re.match(self.text_filter, text):
            return None
        return text

    # def read_rois(self, image, rois=None):
    #     """Extract text from an image at given regions of interest.

    #     Parameters
    #     ----------
    #     image : numpy.ndarray
    #         An RGB or single channel image containing legible text.
    #     rois : numpy.ndarray or None
    #         A numpy array of shape [N, 4] or [4] describing the ROIs to
    #         search for text. The format of the ROIs is [x1, y1, x2, y2].
    #         If rois is None then OcrRegions uses self.rois.

    #     Returns
    #     -------
    #     List
    #         A list of text found in the ROI with corresponding index to rois.
    #     """
    #     roi_text = []
    #     if rois is None:
    #         rois = self.rois
    #     for roi in rois:
    #         roi_img = image[roi[0]:roi[1], roi[2]:roi[3], :]
    #         if self.preprocess is not None:
    #             roi_img = self.preprocess(roi_img)
    #         text = pyocr.tesseract.image_to_string(
    #             cv2_to_Pil(roi_img), lang=self.lang, builder=self.builder)
    #         roi_text.append(self.text_filter(text))
    #     return roi_text
