import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
sess_config = tf.ConfigProto()

import sys
import os

COCO_DATA = '/mnt/Disk4/zbfan/coco/'
MASK_RCNN_MODEL_PATH = 'lib/Mask_RCNN/'
ROOT_DIR = os.getcwd()

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)

from samples.coco import coco
# from mrcnn import utils
# from mrcnn import model as modellib
# from mrcnn import visualize
from lib.mrcnn import utils
from lib.mrcnn import model as modellib
from lib.mrcnn import visualize


from lib import utils as m_utils
from lib import model as m_model
from lib import config as m_config

import time
import datetime
import random
import numpy as np
import skimage.io
import imgaug
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
import keras as K

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from optparse import OptionParser

class TrainConfig(m_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = 'coco'
    EXPERIMENT = 'matching_relation_lr=e-3'
    CHECKPOINT_DIR = 'checkpoints/'
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0,
                    'rpn_bbox_loss': 0.1,
                    'mrcnn_class_loss': 2.0,
                    'mrcnn_embedding_loss': 1.0,
                    'mrcnn_bbox_loss': 0.5,
                    'mrcnn_mask_loss': 1.0}
    DETECTION_MIN_CONFIDENCE = 0.
    DETECTION_NMS_THRESHOLD = 0.5


config = TrainConfig()

train_schedule = OrderedDict()
train_schedule[40] = {"learning_rate": config.LEARNING_RATE, "stage1": "n/a", "stage2": "cls"}
train_schedule[100] = {"learning_rate": config.LEARNING_RATE, "stage1": "n/a", "stage2": "cls"}

parser = OptionParser('')
parser.add_option('--model', dest='model', help='path to weight of the model')
parser.add_option('--dataset', dest='dataset', help='val / train / all is supported')
parser.add_option('--classes', dest='classes', help='which classes to evaluate')
parser.add_option('--ways', dest='ways', help='number of ways to enable')
parser.add_option('--shots', dest='shots', help='number of shots to offer')
parser.add_option('--limit', dest='limit', help='number of images to evaluate in the dataset')
parser.add_option('--verbose', dest='verbose')
parser.add_option('--rois', dest='rois', help='number of random rois to use for stage1')
parser.add_option('--randomroi', dest='randomroi', help='whether or not to use random roi mechanism')
parser.add_option('--nms', dest='nms', help='nms threshold')
parser.add_option('--confidence', dest='confidence', help='minimum detection confidence')
options, args = parser.parse_args()

verbose = 1 if options.verbose is None else int(options.verbose)
limit = 5000 if options.limit is None else int(options.limit)
shots = 1 if options.shots is None else int(options.shots)
ways = 3 if options.ways is None else int(options.ways)
classes_type = options.classes or 'val'
dataset_type = options.classes or 'val'
path = options.model or 'convert_weight/matching_repmet_rpn_lr=e-3_0200.h5'
rois = 1000 if options.rois is None else int(options.rois)
randomroi = True if options.randomroi is None else options.randomroi is True

assert dataset_type in ['train', 'val', 'all']
assert classes_type in ['train', 'val', 'all']

config.USE_RPN_ROIS_INFERENCE = not randomroi
config.POST_NMS_ROIS_INFERENCE = rois
config.WAYS = ways
config.SHOTS = shots
config.SUPPORT_NUMBER = config.WAYS * config.SHOTS
if options.confidence is not None:
    config.DETECTION_MIN_CONFIDENCE = float(options.confidence)
if options.nms is not None:
    config.DETECTION_NMS_THRESHOLD = float(options.nms)

baseline = ['imagenet', 'rpn', 'frcnn'][1]
MODEL_DIR = os.path.join(ROOT_DIR, "logs_" + baseline)
model = m_model.MatchingMaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

model.load_checkpoint(path, training_schedule=train_schedule)

train_classes = []
eval_classes = []
for i in range(1, 81):
    if i % 5 != 0:
        train_classes.append(i)
    else:
        eval_classes.append(i)
train_classes = np.array(train_classes)  #list of class ids
eval_classes = np.array(eval_classes)

datasets = []
if dataset_type == 'train' or dataset_type == 'all':
    coco_train = m_utils.IndexedCocoDataset()
    coco_train_obj = coco_train.load_coco(COCO_DATA, "train", year="2017", return_coco=True)
    coco_train.prepare()
    coco_train.build_indices()
    datasets.append([coco_train, coco_train_obj])
if dataset_type == 'val' or dataset_type == 'all':
    coco_eval = m_utils.IndexedCocoDataset()
    coco_eval_obj = coco_eval.load_coco(COCO_DATA, "val", year="2017", return_coco=True)
    coco_eval.prepare()
    coco_eval.build_indices()  
    datasets.append([coco_eval, coco_eval_obj])

classes = []
if classes_type == 'train' or classes_type == 'all':
    classes.append(train_classes)
if classes_type == 'val' or classes_type == 'all':
    classes.append(eval_classes)

random_rois = config.POST_NMS_ROIS_INFERENCE if not config.USE_RPN_ROIS_INFERENCE else 0
for d, d_obj in datasets:
    for c in classes:
        d.ACTIVE_CLASSES = c
        m_utils.evaluate_dataset(model, d, d_obj, limit=limit, random_rois=random_rois, verbose=verbose)
