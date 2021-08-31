import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)
sess_config = tf.ConfigProto()

import sys
import os

COCO_DATA = '/mnt/Disk4/zbfan/coco/'
MASK_RCNN_MODEL_PATH = 'lib/Mask_RCNN/'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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

# Root directory of the project
ROOT_DIR = os.getcwd()

# train_classes = coco_nopascal_classes


class EvalConfig(m_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    LEARNING_RATE = 0.0001
    WEIGHT_DECAY = 0.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1
    NAME = 'coco'
    EXPERIMENT = 'matching_repmet_lr=1e-3'
    CHECKPOINT_DIR = 'checkpoints/'
    IMAGE_SHAPE = np.array([512, 512, 3])
    
config = EvalConfig()

train_schedule = OrderedDict()
train_schedule[300] = {"learning_rate": config.LEARNING_RATE, "stage1": "n/a", "stage2": "all"}

# Load and evaluate model
# Create model object in inference mode.
MODEL_DIR = os.path.join(ROOT_DIR, "logs_frcnn")
model = m_model.MatchingMaskRCNN(mode="debug", model_dir=MODEL_DIR, config=config)

#save_path = "../image/inference_image"

checkpoint = '/mnt/Disk1/zbfan/code/mm-rcnn/logs_frcnn/matching_mrcnn_coco_matching_repmet_lr=1e-3/matching_mrcnn_0140.h5'
model.load_checkpoint(checkpoint, training_schedule=train_schedule)
#model.load_frcnn_weights()

# Directory to save logs and trained model
train_classes, test_classes = [], []
for i in range(1, 81):
    if i % 5 == 0: test_classes.append(i)
    else: train_classes.append(i)
train_classes, test_classes = np.array(train_classes), np.array(test_classes)

# Load COCO/val dataset
coco_val = m_utils.IndexedCocoDataset()
# coco_val.set_active_classes(train_classes)
coco_object = coco_val.load_coco(COCO_DATA, "val", year="2017", return_coco=True)
#coco_object = coco_val.load_coco(COCO_DATA, "train", year="2017", return_coco=True)
coco_val.prepare()
coco_val.build_indices()
coco_val.ACTIVE_CLASSES = test_classes 
#coco_val.ACTIVE_CLASSES = train_classes 

#np.random.seed(123)
image_ids = np.copy(coco_val.image_ids)
image_count = 0
instance_count = 0
for image_id in image_ids:
    #print('image_id: ', image_id)
    image, image_meta, gt_class_ids, gt_bboxes, gt_masks = \
            modellib.load_image_gt(coco_val, config, image_id, 
                                    use_mini_mask=config.USE_MINI_MASK)
    
    active_gt_class = []

    for gc in gt_class_ids:
        if gc in coco_val.ACTIVE_CLASSES:
            active_gt_class.append(gc)

    if len(active_gt_class) == 0: continue
    else: 
        image_count += 1
        instance_count += len(active_gt_class)
    print('active_gt_class:', active_gt_class)

    # Select category
    category = np.random.choice(active_gt_class)
    categories = [category]

    for i in range(config.WAYS - 1):
        while True:
            res_cat = np.random.choice(coco_val.ACTIVE_CLASSES)
            if res_cat not in categories:
                categories.append(res_cat)
                break

    print('categories: ', categories)

    # Load target
    
    targets = []
    input_targets = []
    tbs = []
    target_ids = []
    for c in categories:
        while True:
            target, tb, target_id = m_utils.get_one_target(c, coco_val, config)[:3]
            if target is not None and tb is not None:
                break

        input_target = m_utils.mold_image(target, config)
        input_targets.append(input_target)
        targets.append(target)
        tbs.append(tb)
        target_ids.append(target_id)

    # Load image
    image = coco_val.load_image(image_id)
    # one more mold
    #input_image = m_utils.mold_image(image, None)
    print("image_id", image_id)

    mask, gt_class_id = coco_val.load_mask(image_id)
    gt_bbox = utils.extract_bboxes(mask)

    gt_class_ids = []
    gt_bboxes = []

    for index, class_id in enumerate(gt_class_id):
        if class_id in coco_val.ACTIVE_CLASSES:
            print(class_id)
            gt_class_ids.append(class_id)
            gt_bboxes.append(gt_bbox[index])

    # Generate random_roi
    active_class_bool = np.zeros_like(gt_class_ids, dtype=np.bool)
    active_class_index = []
    for indexs, x in enumerate(gt_class_ids):
        if x in categories:
            active_class_index.append(indexs)

    for x in categories:
        active_class_bool += gt_class_ids == x
    #print('active_class_bool: ', type(active_class_bool))
    #rpn_gt_bboxes = gt_bboxes[active_class_bool, :]
    rpn_gt_bboxes = np.array([gt_bboxes[i] for i in active_class_index])
    random_rois = 1000
    rpn_rois = m_utils.generate_random_rois(
        image.shape, random_rois, rpn_gt_bboxes, ratio=0.5)
    rpn_rois = np.expand_dims(rpn_rois, axis=0)

    # Run detection
    results = model.detect([input_targets], [tbs], [image], verbose=1, random_rois=rpn_rois)
    """
    print('mold_inpus: ', model.mold_inputs([image], branch='cond').shape)
    molded_images, image_metas, windows = model.mold_inputs([image], branch='cond')

    # molded_targets, target_metas, target_windows = self.mold_inputs(targets)
    molded_targets = [model.mold_inputs(target, branch='cond') for target in [input_targets]]
    molded_targets = np.stack(molded_targets, axis=0)
    targets_bbox = [np.transpose(np.array(x), [1, 0]) for x in [tbs]]
    targets_bbox = np.stack(targets_bbox, axis=0)
    image_shape = molded_images[0].shape
    anchors = model.get_anchors(image_shape)

    detections, _, _, mrcnn_mask, _, _, _, negative_detections, mrcnn_class_logits = \
        model.keras_model.predict([molded_images, image_metas, molded_targets, targets_bbox, anchors, rpn_rois], verbose=0)
    zeros_mask = np.zeros_like(mrcnn_mask)
    """

    # Display results
    #save_path = "train_pretrain_rpn_image"
    #print("targets.shape: ", targets.shape)

    # save detections
    r = results[0]
    # Display results
    rois = r['rois']
    class_ids = r['class_ids']
    scores = r['scores']
    masks = r['masks']
    all_rois = r['all_rois']
    logits = r['logits']
    print('rois: ', len(rois))
    print('logits: ', len(logits))
    print('all_rois: ', len(all_rois))


    path = '../image/matching_repmet_image/all_result.json'
    m_utils.save_inference_with_logits(target_ids, tbs, categories, image_id, r, gt_class_ids, gt_bboxes, path)
    
print('image_count: ', image_count)
print('instance_count', instance_count)


