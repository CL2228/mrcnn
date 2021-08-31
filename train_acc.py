"""
Author: Zhibo Fan
"""
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
sess_config = tf.ConfigProto()

import sys
import os

COCO_DATA = '/mnt/Disk4/zbfan/coco/'

MASK_RCNN_MODEL_PATH = 'lib/Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)

from samples.coco import coco
# from mrcnn import utils
# from mrcnn import model as modellib
from lib.mrcnn import utils
from lib.mrcnn import model as modellib
from lib.mrcnn import visualize

from lib import utils as m_utils
from lib import model as m_model
from lib import config as m_config

import time
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from optparse import OptionParser

ROOT_DIR = os.getcwd()

usage = 'Test Accuracy during train inference.'

class EvalConfig(m_config.Config):
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = 'coco'
    EXPERIMENT = 'random_repmet_lr=e-3'
    CHECKPOINT_DIR = 'checkpoints/'

config = EvalConfig()

train_classes, test_classes = [], []
for i in range(1, 81):
    if i % 5 == 0:
        test_classes.append(i)
    else:
        train_classes.append(i)
train_classes, test_classes = np.array(train_classes), np.array(test_classes)


class AccEvaluate():
    def __init__(self, model, dataset, config, debug=False, iters=5000):
        self.model = model
        self.dataset = dataset
        self.config = config
        self.iters = iters
        self.nb = {
            "small": 0,
            "medium": 0,
            "large": 0,
            "overall": 0,
            "target": 0,
            "non-target": 0
        }
        self.acc = {
            "small": 0.0,
            "medium": 0.0,
            "large": 0.0,
            "overall": 0.0,
            "target": 0.0,
            "non-target": 0.0
        }
        self.others = {
            "fg_confusion": 0.0,
            "bg_misassertion": 0.0,
        }
        self.debug = debug

    def assert_shape(self, bbox, image_meta):
        w, h, _ = modellib.parse_image_meta(image_meta)['original_image_shape'][0]
        bw, bh = bbox[3] - bbox[1], bbox[2] - bbox[0]
        area = bw * w * bh * h
        if area < 32 * 32:
            return "small"
        elif area <= 96 * 96:
            return "medium"
        else:
            return "large"

    def evaluate_acc(self, verbose=1):
        self.time = time.time()
        gen = m_utils.validation_generator(self.dataset, self.config, random_rois=random_rois)
        for i in range(self.iters):
            if verbose > 0 and i % 100 == 0:
                print('evaluating %dth image out of %d images' % (i, self.iters))
            random_rois = self.config.POST_NMS_ROIS_TRAINING if not self.config.USE_RPN_ROIS else 0
            inputs, _, _ = next(validation_generator(self.dataset, self.config,
                                                        random_rois=random_rois))
            image_meta = inputs[1]
            outputs = model.keras_model.predict(inputs, verbose=0)
            output_detections, target_class_ids = outputs[-6], outputs[-7]
            output_detections, target_class_ids = output_detections[0], target_class_ids[0]
            if self.debug:
                print(output_detections)
                print(target_class_ids)
            for j in range(output_detections.shape[0]):
                size = self.assert_shape(output_detections[j, :4], image_meta)
                self.nb[size] += 1
                self.nb["overall"] += 1
                if int(target_class_ids[j]) == 0:
                    self.nb["non-target"] += 1
                else:
                    self.nb["target"] += 1
                if int(np.asscalar(output_detections[j, 4])) == int(target_class_ids[j]):
                    self.acc[size] += 1
                    self.acc["overall"] += 1
                    if int(target_class_ids[j]) == 0:
                        self.acc["non-target"] += 1
                    else:
                        self.acc["target"] += 1
                elif int(target_class_ids[j]) != 0:
                    pred = int(np.asscalar(output_detections[j, 4]))
                    if pred == 0:
                        self.others["bg_misassertion"] += 1
                    else:
                        self.others["fg_confusion"] += 1
        self.time = time.time() - self.time

    def display(self):
        print('========================= SUMMARY =========================')
        if len(self.dataset.ACTIVE_CLASSES) == 16:
            print("Evaluating on COCO Dataset 2017 Val with 16 categories.")
        else:
            print("Evaluating on COCO Dataset 2017 Train with 64 categories.")
        line = '{}: Average Accuracy: {:0.4f} | Number of Instances: {:d}'
        for key in ['small', 'medium', 'large', 'overall', "target", "non-target"]:
            if self.nb[key] == 0:
                print('{}: Instances of this category not found'.format(key.upper()))
                continue
            avg_acc = self.acc[key] / self.nb[key]
            print(line.format(key.upper(), avg_acc, self.nb[key]))
        false_pred = self.nb['target'] - self.acc['target']
        print('During those false positive predictions:')
        print('{:0.4f} are foreground confusions, {:0.4f} are background misassertion.'.format(
                                                                              self.others['fg_confusion'] / false_pred, 
                                                                              self.others['bg_misassertion'] / false_pred))
        print('Total time cost is {:4.2f}'.format(self.time))
        print('\n===================== END OF SUMMARY ======================')

if __name__ == '__main__':
    parser = OptionParser(usage)
    parser.add_option('--model', dest='model', help='The directory of weight file, default is frcnn_fc_0300')
    parser.add_option('--dataset', dest='dataset', help='which mode to test, default is val 2017')
    parser.add_option('--limit', dest='limit', help='how much S-Q sets to explore, default is 5000')
    parser.add_option('--ways', dest='ways', help='how many ways to support during detection')
    parser.add_option('--shots', dest='shots', help='how many shots to support during detection')
    parser.add_option('-v', '--verbose', dest='verbose', help='print current evaluation process')
    parser.add_option('-c', '--classes', dest='classes', help='which set of classes to evaluate on')
    parser.add_option('--debug', dest='debug', help='debug mode for printing out ground truth')
    options, args = parser.parse_args()

    MODEL_DIR = os.path.join(ROOT_DIR, "logs_frcnn")
    if options.ways: config.WAYS = int(options.ways)
    if options.shots: config.SHOTS = int(options.shots)
    config.SUPPORT_NUMBER = config.WAYS * config.SHOTS
    debug = False if not options.debug else True

    checkpoint = 'logs_frcnn/modify/new_modify/matching_mrcnn_fc_0300.h5' if not options.model else options.model
    # If not passing train_schedule, the model will explode!
    train_schedule = OrderedDict()
    train_schedule[300] = {"learning_rate": config.LEARNING_RATE, "stage1": "n/a", "stage2": "all"}

    model = m_model.MatchingMaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
    model.load_checkpoint(checkpoint, training_schedule=train_schedule)
    subset = 'val' #if not options.dataset else options.dataset
    classes = 'val' if not options.classes else options.classes
    # classes = 'val'
    assert subset in ['val', 'train', 'all'], 'Only val / train / all are supported so far.'
    assert classes in ['val', 'train', 'all'], 'Only val / train / all are supported so far.'
    if classes == 'val':
        classes = [test_classes]
    elif classes == 'train':
        classes = [train_classes]
    else:
        classes = [train_classes, test_classes]
    dataset = []
    if subset == 'train' or subset == 'all':
        coco_train = m_utils.IndexedCocoDataset()
        coco_train.load_coco(COCO_DATA, "train", year="2017")
        coco_train.prepare()
        coco_train.build_indices()
        dataset.append(coco_train)
    if subset == 'val' or subset == 'all':
        coco_val = m_utils.IndexedCocoDataset()
        coco_val.load_coco(COCO_DATA, "val", year="2017")
        coco_val.prepare()
        coco_val.build_indices()
        dataset.append(coco_val)

    iters = 5000 if not options.limit else int(options.limit)
    verbose = 0 if not options.verbose else int(options.verbose)
    for d in dataset:
        for c in classes:
            d.ACTIVE_CLASSES = c
            acc_val = AccEvaluate(model, d, config, debug, iters)
            acc_val.evaluate_acc(verbose)
            acc_val.display()
    print('Weight: {}'.format(checkpoint))

