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
from debugger import weight_check

class TrainConfig(m_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-4
    VALIDATION_STEPS = 50
    GPU_COUNT = 1
    IMAGES_PER_GPU = 6
    WAYS = 1
    SHOTS = 1
    NAME = 'coco'
    EXPERIMENT = 'spatial_lr=e-3'
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0, 
                    'rpn_bbox_loss': 1.0}
    DATASET_TYPE = 'coco'
    ATTENTION = 'spatial' # else is 'channel'
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    
config = TrainConfig()

train_schedule = OrderedDict()
train_schedule[40] = {"learning_rate": config.LEARNING_RATE, "layers": "4+"}

MODEL_DIR = os.path.join(ROOT_DIR, "logs_stage1")
model = m_model.Stage1Network(mode="training", model_dir=MODEL_DIR, config=config)
try: 
    model.load_latest_checkpoint(training_schedule=train_schedule)
except:
    model.load_imagenet_weights(pretraining='imagenet-1k')
    
train_classes = []
eval_classes = []
for i in range(1, 81):
    if i % 5 != 0: train_classes.append(i)
    else: eval_classes.append(i)
train_classes = np.array(train_classes)
eval_classes = np.array(eval_classes)

coco_train = m_utils.IndexedCocoDataset()
coco_train.load_coco(COCO_DATA, "train", year="2017")
coco_train.prepare()
coco_train.build_indices()
coco_train.ACTIVE_CLASSES = train_classes

coco_eval = m_utils.IndexedCocoDataset()
coco_eval.load_coco(COCO_DATA, "val", year="2017")
coco_eval.prepare()
coco_eval.build_indices()
coco_eval.ACTIVE_CLASSES = eval_classes # This should be modified if not training rpn 
    
for epochs, params in train_schedule.items():
    print("")
    print("training layers {} until epoch {} with learning_rate {}".format(params["layers"], 
                                                                          epochs, 
                                                                          params["learning_rate"]))
    model.train(coco_train, coco_eval, 
                learning_rate=params["learning_rate"], 
                epochs=epochs, 
                regex=params["layers"])


