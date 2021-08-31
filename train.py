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
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-6
    VALIDATION_STEPS = 50
    GPU_COUNT = 2
    IMAGES_PER_GPU = 4
    NAME = 'coco'
    EXPERIMENT = 'spatial+relation_lr=e-4'
    CHECKPOINT_DIR = 'checkpoints/'
    MASK_LATE_FUSION = True
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64
    ATTENTION = 'spatial'
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0, 
                    'rpn_bbox_loss': 1.0, 
                    'mrcnn_class_loss': 2.0,
                    'mrcnn_bbox_loss': 1.0, 
                    'mrcnn_mask_loss': 1.0}
    DATASET_TYPE = 'coco'
    
config = TrainConfig()

parser = OptionParser('')
parser.add_option('-p', '--pretrain', dest='pretrain', help='load which pretrain model')
options, args = parser.parse_args()

train_schedule = OrderedDict()
train_schedule[40] = {"learning_rate": config.LEARNING_RATE, "stage1": "rpn", "stage2": "all"}
train_schedule[80] = {"learning_rate": config.LEARNING_RATE, "stage1": "4+", "stage2": "all"}
train_schedule[100] = {"learning_rate": config.LEARNING_RATE / 10, "stage1": "4+", "stage2": "all"}
train_schedule[120] = {"learning_rate": config.LEARNING_RATE / 10, "stage1": "all", "stage2": "all"}

baseline = ['imagenet', 'rpn', 'frcnn'][0]
if options.pretrain:
    assert options.pretrain in ['imagenet', 'rpn', 'frcnn'], 'only imagenet / rpn / frcnn is supported'
    baseline = options.pretrain
MODEL_DIR = os.path.join(ROOT_DIR, "logs_" + baseline)
print('pretraining model from ' + baseline)

model = m_model.MatchingMaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)

try: 
    model.load_latest_checkpoint(training_schedule=train_schedule)
except:
    if baseline == 'imagenet':
        model.load_imagenet_weights(pretraining='imagenet-1k')
    elif baseline == 'frcnn':
        model.load_frcnn_weights()
    elif baseline == 'rpn':
        model.load_rpn_weights()
    #weight_check(model, baseline, no_head=True)
    
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
    print("training layers {} until epoch {} with learning_rate {}".format(params["stage1"] + " and " + params["stage2"], 
                                                                          epochs, 
                                                                          params["learning_rate"]))
    model.train(coco_train, coco_eval, 
                learning_rate=params["learning_rate"], 
                epochs=epochs, 
                stage1=params["stage1"],
                stage2=params["stage2"])


