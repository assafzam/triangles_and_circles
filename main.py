import json
import logging
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from Evalutor import Evaluator
from gbox.gbox import PredGBox, GBox
from gbox.gboxes_on_image import PredGBoxesOnImage, GBoxesOnImage
from gbox.types import NumpyImg

IMG_DIR = 'img'

GROUND_TRUTH_DIR = 'ground_truth'
PREDICTION_DIR = 'prediction'

PINK = np.array((255, 192, 203))
BLUE = np.array((0, 0, 128))

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

        self.data_dir: str = data_dir
        self.name: str = name
        self.image_name: str = self.name + img_ext
        self.pred_name: str = self.name + pred_ext
        self.gt_name: str = self.name + gt_ext

        self.raw_image: NumpyImg = None
        self.raw_gt: dict = None
        self.raw_pred: dict = None

        self.gt_color = gt_color
        self.pred_color = pred_color

    def load_raw_img(self, show=False) -> NumpyImg:
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

    def draw_pred_on_image(self, image=None, show=False) -> NumpyImg:
        if self.raw_image is None:
            self.load_raw_img()
        image_to_draw_on = image if image is not None else self.raw_image

        pred_boxes = self.get_pred_boxes()
        pred_boxes.img_shape = image_to_draw_on.shape
        plt.title(self.name)
        draw_image = pred_boxes.draw_on_image(image_to_draw_on, single_color=self.pred_color, thickness=1, show=show)
        return draw_image

    def get_pred_boxes(self) -> GBoxesOnImage:
        if self.raw_pred is None:
            self.load_raw_pred()
        circles = self.raw_pred.get('circle')
        circles_gboxes = [PredGBox(x1=c[0][0], y1=c[0][1], x2=c[1][0], y2=c[1][1], probs=1, class_name='circle') for c in
                          circles]

        triangles = self.raw_pred.get('triangle')
        triangles_gboxes = [PredGBox(x1=c[0][0], y1=c[0][1], x2=c[1][0], y2=c[1][1], probs=1, class_name='triangle') for c in
                            triangles]

        boxes = circles_gboxes + triangles_gboxes
        pred_boxes = PredGBoxesOnImage(gboxes=boxes)
        return pred_boxes

    def draw_gt_on_image(self, image=None, show=False) -> NumpyImg:
        if self.raw_image is None:
            self.load_raw_img()

        image_to_draw_on = image if image is not None else self.raw_image

        gboxes = self.get_gt_boxes()
        gboxes.img_shape = image_to_draw_on.shape
        draw_image = gboxes.draw_on_image(image_to_draw_on, single_color=self.gt_color, thickness=3, show=show)
        return draw_image

    def get_gt_boxes(self) -> GBoxesOnImage:
        if self.raw_gt is None:
            self.load_raw_gt()
        circles = self.raw_gt.get('circle')
        circles_gboxes = [GBox(x1=c[0][0], y1=c[0][1], x2=c[1][0], y2=c[1][1], class_name='circle') for c in circles]

        triangles = self.raw_gt.get('triangle')
        triangles_gboxes = [GBox(x1=c[0][0], y1=c[0][1], x2=c[1][0], y2=c[1][1], class_name='triangle') for c in
                            triangles]

        boxes = circles_gboxes + triangles_gboxes
        gt_boxes = GBoxesOnImage(gboxes=boxes)
        return gt_boxes

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

    def get_FP_TP_by_class(self, class_name = 'circle', iou_threshold = 0.8):

        pred_boxes = [p for p in  self.get_pred_boxes() if p.class_name==class_name]
        gt_boxes = [p for p in self.get_gt_boxes() if p.class_name==class_name]

        TP = np.zeros(len(pred_boxes))
        FP = np.zeros(len(pred_boxes))

        already_seen = np.zeros(len(gt_boxes))

        for pred_idx, pred_box in enumerate(pred_boxes):
            iouMax = sys.float_info.min
            gt_max_idx = None
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = pred_box.iou(gt_box)
                if iou > iouMax:
                    iouMax = iou
                    gt_max_idx= gt_idx
            if iouMax >= iou_threshold:
                if already_seen[gt_max_idx] == 0: # not seen
                    TP[pred_idx] = 1 # count as true positive
                    already_seen[gt_max_idx] = 1  # flag as already 'seen'
                else:
                    FP[pred_idx] = 1  # count as false positive
            else:
                FP[pred_idx] = 1  # count as false positive

        return FP, TP

    def get_FP_TP(self, iou_threshold = 0.8 ):
        pred_boxes = self.get_pred_boxes()
        gt_boxes = self.get_gt_boxes()

        TP = np.zeros(len(pred_boxes))
        FP = np.zeros(len(pred_boxes))
        total_true_boxes = len(gt_boxes)

        already_seen = np.zeros(len(gt_boxes))
            # class_pred_boxes = [p for p in pred_boxes if p.class_name==c]
            # class_gt_boxes = [p for p in gt_boxes if p.class_name==c]
        for pred_idx, pred_box in enumerate(pred_boxes):
            iouMax = sys.float_info.min
            gt_max_idx = None
            for gt_idx, gt_box in enumerate(gt_boxes):
                if gt_box.class_name != pred_box.class_name:
                    continue
                iou = pred_box.iou(gt_box)
                if iou > iouMax:
                    iouMax = iou
                    gt_max_idx= gt_idx
            if iouMax >= iou_threshold:
                if already_seen[gt_max_idx] == 0: # not seen
                    TP[pred_idx] = 1 # count as true positive
                    already_seen[gt_max_idx] = 1  # flag as already 'seen'
                else:
                    FP[pred_idx] = 1  # count as false positive
            else:
                FP[pred_idx] = 1  # count as false positive

        return FP, TP

class AlLData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.FP = defaultdict(lambda : np.array([]))
        self.TP = defaultdict(lambda : np.array([]))

        self.acc_FP = {}
        self.acc_TP = {}
        self.acc_recall = {}
        self.recall_num = {}
        self.acc_precision = {}
        self.precision_num = {}
        self.ap = {}

        self.FP_by_class = {}
        self.TP_by_class = {}

        self.num_of_gt, self.num_of_predictions = self.calc_amounts()
        self.iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.all_classes = ['circle', 'triangle']

    def calc_amounts(self):
        img_dir = os.path.join(data_dir, IMG_DIR)
        num_of_gt = 0
        num_of_predictions = 0
        for img_name in os.listdir(img_dir):
            file_name = os.path.splitext(img_name)[0]
            image = GImage(data_dir=data_dir, name=file_name)
            num_of_gt += len(image.get_gt_boxes())
            num_of_predictions += len(image.get_pred_boxes())
        return num_of_gt, num_of_predictions

    def run(self):
        img_dir = os.path.join(data_dir, IMG_DIR)
        fig = plt.figure()
        for iou_threshold in self.iou_thresholds:

            for img_name in os.listdir(img_dir):
                file_name = os.path.splitext(img_name)[0]
                image = GImage(data_dir=data_dir, name=file_name)

                img_FP, img_TP = image.get_FP_TP(iou_threshold)
                self.FP[iou_threshold] = np.concatenate((self.FP[iou_threshold], img_FP))
                self.TP[iou_threshold] = np.concatenate((self.TP[iou_threshold], img_TP))

            self.acc_FP[iou_threshold] = np.cumsum(self.FP[iou_threshold])
            self.acc_TP[iou_threshold] = np.cumsum(self.TP[iou_threshold])
            self.acc_recall[iou_threshold] = self.acc_TP[iou_threshold] / self.num_of_gt
            self.recall_num[iou_threshold] = np.sum(self.TP[iou_threshold]) / self.num_of_gt

            self.acc_precision[iou_threshold] = np.divide(self.acc_TP[iou_threshold], (self.acc_FP[iou_threshold] + self.acc_TP[iou_threshold]))
            self.precision_num[iou_threshold] = np.sum(self.TP[iou_threshold]) / (np.sum(self.FP[iou_threshold]) + np.sum(self.TP[iou_threshold]))

            self.ap[iou_threshold], _, _, _ = Evaluator.CalculateAveragePrecision(self.acc_recall,  self.acc_precision)

            plt.plot(self.acc_recall[iou_threshold], self.acc_precision[iou_threshold], label=iou_threshold)
        plt.legend()
        plt.title('recall/precision')
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.show()




if __name__ == '__main__':
    data_dir = 'data'
    data = AlLData(data_dir=data_dir)
    data.run()
    print('h')


    img_dir = os.path.join(data_dir, IMG_DIR)
    all_fp = np.array([])
    all_tp = np.array([])
    all_gt_boxes = 0
    for img_name in os.listdir(img_dir):
        file_name = os.path.splitext(img_name)[0]

        # file_name = '5e806d84-13ea-4c7e-8321-ce3a33a631e8'
        image = GImage(data_dir=data_dir, name=file_name)
        # image.draw_pred_on_image(image=image.draw_gt_on_image(show=False), show=True)
        all_gt_boxes += len(image.get_gt_boxes())

        FP, TP = image.get_FP_TP(class_name='triangle')
        all_fp = np.concatenate((all_fp,FP))
        all_tp = np.concatenate((all_tp,TP))

    acc_FP = np.cumsum(all_fp)
    acc_TP = np.cumsum(all_tp)
    recall = acc_TP / all_gt_boxes
    recall_num = np.sum(all_tp) / all_gt_boxes
    precision = np.divide(acc_TP, (acc_FP + acc_TP))
    precision_num = np.sum(all_tp)/ (np.sum(all_fp) + np.sum(all_tp))


    [ap, mpre, mrec, ii] = Evaluator.CalculateAveragePrecision(recall, precision)

    plt.plot(recall, precision, label='Precision')
    plt.show()

    print('hi')
