import json

import cv2
import numpy as np

class OcrRegions(object):
    def __init__(self, data_path=None, image_shape=None, language=None, whitelist=None):
        self.rois = []
        self.tesseract = None
        if data_path:
            datapath = data_path
            self.image_shape = image_shape or (350, 70)
            language = language or 'eng'
            whitelist = whitelist or None
            psmode, oem = 7, 3
            self.tesseract = cv2.text.OCRTesseract_create(
                datapath, language, whitelist, oem, psmode)

    def load_config(self, config_file):
        with open(config_file, "r") as file:
            config = json.load(file)
        self.image_shape = config["image_shape"]
        self.tesseract = cv2.text.OCRTesseract_create(
            config["datapath"], config["language"],
            config["whitelist"], config["oem"], config["psmode"])
        self.rois = config['rois']
        return self

    def _raw_tesseract(self, roi_img, tesseract, closing_iters=None, thresh=None, min_confidence=60):
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        roi_img = cv2.resize(roi_img, self.image_shape)
        if thresh:
            _, roi_img = cv2.threshold(roi_img, 0, 255, cv2.THRESH_OTSU)
        if closing_iters:
            roi_img = cv2.dilate(roi_img, kernel=np.ones((5, 5), np.uint8), iterations=closing_iters)
            roi_img = cv2.erode(roi_img, kernel=np.ones((5, 5), np.uint8), iterations=closing_iters)
        return tesseract.run(roi_img, min_confidence)

    def read(self, image, rois=None):
        """Extract text from an image at given regions of interest.

        Parameters
        ----------
        image : numpy.ndarray
            An RGB or single channel image containing legible text.
        rois : numpy.ndarray or None
            A numpy array of shape [N, 4] or [4] describing the ROIs to
            search for text. The format of the ROIs is [x1, y1, x2, y2].
            If rois is None then OcrRegions uses self.rois.

        Returns
        -------
        List
            A list of text found in the ROI with corresponding index to rois.
        """
        if not self.tesseract:
            raise ValueError("OcrRegions not initialised.")
        text = []
        if rois is None:
            rois = self.rois
        for roi in rois:
            roi_img = image[roi[0]:roi[1], roi[2]:roi[3], :]
            raw_text = self._raw_tesseract(roi_img, self.tesseract)
            if raw_text and len(raw_text) > 3 and raw_text[-3] == "."\
                    and "." not in raw_text[:-3] and "." not in raw_text[-2:]:
                raw_chainage = raw_text
            else:
                raw_chainage = ''
            text.append(raw_chainage)
        return text
