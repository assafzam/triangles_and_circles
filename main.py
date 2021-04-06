import json
import logging
import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from gbox.gbox import PredGBox, GBox
from gbox.gboxes_on_image import PredGBoxesOnImage, GBoxesOnImage
from gbox.types import NumpyImg

IMG_DIR = 'img'

GROUND_TRUTH_DIR = 'ground_truth'
PREDICTION_DIR = 'prediction'

PINK = np.array((255,192,203))
BLUE = np.array((0,0,128))

logger = logging.getLogger()

JPG_EXT = '.jpg'
JSON_ECT = '.json'

class Loader:

    def load_image(self, path: str, show=False) -> NumpyImg:
        if os.path.isfile(path):
            image = np.asarray(Image.open(path))
            if show:
                plt.imshow(image)
                plt.show()
            return image
        else:
            logger.warning(f"No Image was found in {path}")


class GImage:
    def __init__(self,
                 data_dir: str,
                 name: str,
                 loader: Loader = None,
                 img_ext: str = JPG_EXT,
                 gt_ext: str = JSON_ECT,
                 pred_ext: str = JSON_ECT,
                 gt_color: str = BLUE,
                 pred_color: str = PINK,
                 ):

        if loader is not None:
            self.loader = loader
        else:
            self.loader = Loader()

        self.data_dir   : str = data_dir
        self.name       : str = name
        self.image_name : str = self.name + img_ext
        self.pred_name   : str = self.name + pred_ext
        self.gt_name    : str = self.name + gt_ext

        self.raw_image  : NumpyImg = None
        self.raw_gt     : dict      = None
        self.raw_pred   : dict      = None

        self.gt_color = gt_color
        self.pred_color = pred_color

    def load_raw_img(self, show = False) -> NumpyImg:
        path = os.path.join(self.data_dir, IMG_DIR, self.image_name)
        self.raw_image = self.loader.load_image(path=path, show=show)
        return self.raw_image

    def load_raw_pred(self) -> dict:
        path = os.path.join(self.data_dir, PREDICTION_DIR, self.pred_name)
        data = self._load_json(path)
        self.raw_pred = data
        return data

    def load_raw_gt(self) -> dict:
        path = os.path.join(self.data_dir, GROUND_TRUTH_DIR, self.gt_name)
        data = self._load_json(path)
        self.raw_gt = data
        return data

    def draw_pred_on_image(self, image = None, show = False) -> NumpyImg:
        if self.raw_image is None:
            self.load_raw_img()
        if self.raw_pred is None:
            self.load_raw_pred()
        image_to_draw_on = image if image is not None else self.raw_image

        circles = self.raw_pred.get('circle')
        circles_gboxes = [PredGBox(x1=c[0][0], y1=c[0][1], x2=c[1][0], y2=c[1][1], class_name='circle') for c in circles]

        triangles = self.raw_pred.get('triangle')
        triangles_gboxes = [PredGBox(x1=c[0][0], y1=c[0][1], x2=c[1][0], y2=c[1][1], class_name='triangle') for c in triangles]

        boxes = circles_gboxes + triangles_gboxes
        pred_boxes = PredGBoxesOnImage(gboxes=boxes)
        pred_boxes.img_shape = image_to_draw_on.shape
        draw_image = pred_boxes.draw_on_image(image_to_draw_on, single_color=self.pred_color, thickness=1, show=show)
        return draw_image

    def draw_gt_on_image(self, image = None, show = False) -> NumpyImg:
        if self.raw_image is None:
            self.load_raw_img()
        if self.raw_gt is None:
            self.load_raw_gt()
        image_to_draw_on = image if image is not None else self.raw_image
        circles = self.raw_gt.get('circle')
        circles_gboxes = [GBox(x1=c[0][0], y1=c[0][1], x2=c[1][0], y2=c[1][1], class_name='circle') for c in circles]

        triangles = self.raw_gt.get('triangle')
        triangles_gboxes = [GBox(x1=c[0][0], y1=c[0][1], x2=c[1][0], y2=c[1][1], class_name='triangle') for c in triangles]

        boxes = circles_gboxes + triangles_gboxes
        gboxes = GBoxesOnImage(gboxes=boxes)
        gboxes.img_shape = image_to_draw_on.shape
        draw_image = gboxes.draw_on_image(image_to_draw_on, single_color=self.gt_color,thickness=3, show=show)
        return draw_image






    def _load_json(self, path) -> dict:
        if os.path.isfile(path):
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                return data
            except Exception as e:
                logger.warning(f"bad json file in {path} !")
                return {}
        else:
            logger.warning(f"json file does not exist in {path}")
            return {}


if __name__ == '__main__':
    data_dir = 'data'
    file_name = '0a1c697f-817c-46b7-8a0e-ed34e05b85ea'
    image = GImage(data_dir=data_dir, name=file_name)
    draw = image.draw_gt_on_image(show=False)
    image.draw_pred_on_image(image=draw, show=True)

    print('hi')
