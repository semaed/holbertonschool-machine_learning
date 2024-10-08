#!/usr/bin/env python3
"""Script to initialize YOLOv3"""

import tensorflow.keras as K
import numpy as np


class Yolo():
    """
    Class YOLOv3
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Method init for Yolov3
        Args:
            model_path: path to where a Darknet nperas model is stored
            classes_path: path to where the list of class names used for
                          the Darknet model, listed in order of index,
                          can be found
            class_t: the box score threshold for the initial filtering step
            nms_t: the IOU threshold for non-max suppression
            anchors: the anchor boxes
        """
        # Load model
        self.model = K.models.load_model(model_path)
        # Load classes
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """ sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        Method containing the predictions from the darknet_model
        Args:
            outputs: list of numpy.ndarrays containing the predictions from
                     the Darknet model for a single image:
                     (grid_height, grid_width, anchor_boxes, 4 + 1 + classes)
            image_size: numpy.ndarray containing the image’s original size
                        [image_height, image_width]
        Returns: (boxes, box_confidences, box_class_probs)
        """
        boxes = [pred[:, :, :, 0:4] for pred in outputs]
        for ipred, pred in enumerate(boxes):
            for grid_h in range(pred.shape[0]):
                for grid_w in range(pred.shape[1]):
                    bx = ((self.sigmoid(pred[grid_h,
                                        grid_w, :,
                                        0]) + grid_w) / pred.shape[1])
                    by = ((self.sigmoid(pred[grid_h,
                                        grid_w, :,
                                        1]) + grid_h) / pred.shape[0])
                    anchor_tensor = self.anchors[ipred].astype(float)
                    anchor_tensor[:, 0] *= \
                        np.exp(pred[grid_h, grid_w, :,
                               2]) / self.model.input.shape[1].value  # bw
                    anchor_tensor[:, 1] *= \
                        np.exp(pred[grid_h, grid_w, :,
                               3]) / self.model.input.shape[2].value  # bh

                    pred[grid_h, grid_w, :, 0] = \
                        (bx - (anchor_tensor[:, 0] / 2)) * \
                        image_size[1]  # x1
                    pred[grid_h, grid_w, :, 1] = \
                        (by - (anchor_tensor[:, 1] / 2)) * \
                        image_size[0]  # y1
                    pred[grid_h, grid_w, :, 2] = \
                        (bx + (anchor_tensor[:, 0] / 2)) * \
                        image_size[1]  # x2
                    pred[grid_h, grid_w, :, 3] = \
                        (by + (anchor_tensor[:, 1] / 2)) * \
                        image_size[0]  # y2
        # box confidence

        box_confidences = [self.sigmoid(pred[:, :, :,
                                        4:5]) for pred in outputs]

        # box class probs
        box_class_probs = [self.sigmoid(pred[:, :, :,
                                        5:]) for pred in outputs]
        return boxes, box_confidences, box_class_probs
