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

### Eval Function ###
def visualize_dataset(model, dataset, exp, limit=100, verbose=1, save=True, 
                      save_dir='../image/att-stage1/', positive_only=False):
    assert model.config.WAYS == 1
    if limit < 0:
        limit = len(dataset.image_ids)
    limit = max(limit, len(dataset.image_ids))
    generator = m_utils.stage1_validation_generator(dataset, model.config, shuffle=False)
    save_path = save_dir + exp
    results = []
    for i in range(limit):
        if i % 100 == 0 and verbose > 1:
            print("Processing image {}/{} ...".format(i, len(image_ids)))
        model_inp, _, target_class_ids = next(generator)
        target_class_id = target_class_ids[0]
        target_class = model.config.CLASS_ID[target_class_id]
        meta = model_inp[1]
        m = modellib.parse_image_meta(meta)
        image_id = m['image_id'][0]
        r = model.pseudo_detect(model_inp)
        rois = r['train_rois']
        rois_id = r['train_rois_id']
        N = rois.shape[0]
        colors = visualize.random_colors(N)
        fig = plt.figure()
        ax = fig.add_subplot(121)
        ax.imshow((model_inp[2][0, ..., 0] + model.config.MEAN_PIXEL).astype(np.int32))
        ax.set_title('class_id: ' + target_class)
        ax = fig.add_subplot(122)
        ax.imshow(dataset.load_image(image_id))
        for j in range(N):
            color = colors[j]
            if not np.any(boxes[i]):
                continue
            if positive_only and rois_id[j] == 0:
                continue
            y1, x1, y2, x2 = boxes[j]
            y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
            p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)
        if save:
            fig.savefig(save_dir + '/' + str(image_id) + '.jpg')
        else:
            plt.show()            


### Main ###
from optparse import OptionParser

class EvalConfig(m_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = 'coco'
    EXPERIMENT = 'spatial_lr=e-3'
    CHECKPOINT_DIR = 'checkpoints/'
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0,
                    'rpn_bbox_loss': 1.0}
    DETECTION_NMS_THRESHOLD = 0.5
    POST_NMS_ROIS_INFERENCE = 500
    RPN_NMS_THRESHOLD = 0.7
    ATTENTION = 'spatial'

config = EvalConfig()

train_schedule = OrderedDict()
train_schedule[200] = {"learning_rate": config.LEARNING_RATE, "layers": "4+"}

parser = OptionParser('')
parser.add_option('--model', dest='model', help='path to weight of the model')
parser.add_option('--dataset', dest='dataset', help='val / train / all is supported')
parser.add_option('--classes', dest='classes', help='which classes to evaluate')
parser.add_option('--ways', dest='ways', help='number of ways to enable')
parser.add_option('--shots', dest='shots', help='number of shots to offer')
parser.add_option('--limit', dest='limit', help='number of images to evaluate in the dataset')
parser.add_option('--verbose', dest='verbose')
parser.add_option('--nms', dest='nms', help='nms threshold')
parser.add_option('--rois', dest='rois', help='number of proposals generated by rpn')
parser.add_option('-e', '--exp', dest='exp')
parser.add_option('--save', dest='save', help='boolean, save if True or show')
parser.add_option('--path', dest='path', help='path to save dir')
parser.add_option('-p', '--positive_only', help='show positive train rois only')
options, args = parser.parse_args()

verbose = 1 if options.verbose is None else int(options.verbose)
limit = 100 if options.limit is None else int(options.limit)
shots = 1 if options.shots is None else int(options.shots)
ways = 1 if options.ways is None else int(options.ways)
classes_type = options.classes or 'val'
dataset_type = options.classes or 'val'
if options.exp:
    config.EXPERIMENT = options.exp
save_dir = options.path or '../image/att-stage1/'
save = True if options.save is None else options.save.lower() == 'true'
positive_only = False if options.pos is None else options.pos.lower() == 'false'

# Manually set is supported.
path = options.model or 'convert_weight/matching_repmet_rpn_lr=e-3_0200.h5' 

assert dataset_type in ['train', 'val', 'all']
assert classes_type in ['train', 'val', 'all']

config.USE_RPN_ROIS_INFERENCE = not randomroi
config.POST_NMS_ROIS_INFERENCE = float(options.rois)
config.WAYS = ways
config.SHOTS = shots
config.SUPPORT_NUMBER = config.WAYS * config.SHOTS
if options.nms is not None:
    config.RPN_NMS_THRESHOLD = float(options.nms)

MODEL_DIR = os.path.join(ROOT_DIR, "logs_stage1")
model = m_model.Stage1Network(mode="inference", model_dir=MODEL_DIR, config=config)
model.load_checkpoint(path, training_schedule=train_schedule)

train_classes = []
eval_classes = []
for i in range(1, 81):
    if i % 5 != 0:
        train_classes.append(i)
    else:
        eval_classes.append(i)
train_classes = np.array(train_classes)
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
        evaluate_dataset(model, d, config.EXPERIMENT, limit=limit, verbose=verbose, save=save,
                         save_dir=save_dir, positive_only=positive_only)