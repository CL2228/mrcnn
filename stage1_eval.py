"""
Author: Zhibo Fan
This file loads the whole model and evaluate
stage1 metrics and visualize stage1 results.
"""
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
sess_config = tf.ConfigProto()

import sys
import os
from argparse import ArgumentParser

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
from lib import backbone as backbone

import random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import keras as K
import keras.layers as KL
import keras.models as KM
import keras.engine as KE

class Config(m_config.Config):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = 'coco'
    EXPERIMENT = 'relation_lr=e-3'
    CHECKPOINT_DIR = 'checkpoints/'
    MASK_LATE_FUSION = True
    # Adapt loss weights
    DATASET_TYPE = 'coco'
    POST_NMS_INFERENCE = 500
    RPN_NMS_THRESHOLD = 0.7
    TRAIN_ROIS_PER_IMAGE = 30
    ROI_POSITIVE_RATIO = 0.33
    
class PositiveFilter(KE.Layer):
    """ This layer filters ROIs to at most roi_count ROIs that
    have higher than 0.5 IOU with ground truth boxes.
    """
    def __init__(self, config, roi_count, **kwargs):
        super(PositiveFilter, self).__init__(**kwargs)
        self.config = config
        self.roi_count = roi_count
            
    def call(self, inputs):
        proposals, gt_boxes = inputs
        output = utils.batch_slice([proposals, gt_boxes], 
                                   lambda x, y: self._batch_func(x, y), self.config.IMAGES_PER_GPU)
        return outputs
            
    def _batch_func(self, prop, gt_boxes):
        prop, _ = modellib.trim_zeros_graph(prop[:, :4])
        gt_boxes, _ = backbone.trim_zeros_graph(gt_boxes)
        overlaps = modellib.overlaps_graph(prop, gt_boxes)
        roi_iou_max = tf.reduce_max(overlaps, axis=1)
        positive_roi_bool = (roi_iou_max >= 0.5)
        positive_indices = tf.where(positive_roi_bool)[:, 0]
        positive_indices = tf.random_shuffle(positive_indices)[:self.roi_count]
        positive_count = tf.shape(positive_indices)[0]
        positive_rois = tf.gather(proposals, positive_indices)
        indicators = tf.reduce_max(tf.ones_like(positive_rois, dtype=tf.float32), axis=-1, keepdims=True)
        rois = tf.concat([positive_rois, indicators], axis=-1)
        P = tf.maximum(self.roi_count - tf.shape(positive_rois)[0], 0) 
        rois = tf.pad(rois, [(0, P), (0, 0)])
        return rois
           
    def compute_output_shape(self, input_shape):
        return (None, self.roi_count, 5)

class ROIEval:
    def __init__(self, model, dataset, kwargs):
        assert model.mode == 'training' and model.config.MODEL == 'mrcnn'
        self.model = model
        self.dataset = dataset
        self.config = model.config
        self.roi_type = kwargs.get('roi_type', 'train_roi')
        self.usage = kwargs.get('usage', 'visualize')
        self.positive_only = kwargs.get('pos_only', False) and self.usage == 'vis'
        self.positive_number = kwargs.get('pos_num', self.config.TRAIN_ROIS_PER_IMAGE)
        if self.usage == 'metric':
            self.dataset_obj = kwargs.get('dataset_obj')
            assert self.dataset_obj is not None, "dataset obj must be fed if evaluate metrics."
        self.refiner = self.build_refiner()
        self.gen = m_utils.validation_generator(dataset, self.config, shuffle=False)
        
    def build_refiner(self):
        input_rois = KL.Input([None, 4], name='input_rois')
        input_meta = KL.Input([13,], name='input_meta')
        clipper_mode = 'detection_target' if self.roi_type == 'train_roi' else\
                       'proposal'
        clipper = backbone.ROIDetectionLayer(clipper_mode, self.config, name='clipper')
        rois = clipper([input_rois, input_meta])
        inputs = [input_rois, input_meta]
        if self.positive_only:
            input_gt_boxes = KL.Input([None, 4], name='input_gt_boxes')
            gt_boxes = KL.Lambda(lambda x: modellib.norm_boxes_graph(
                x, modellib.parse_image_meta(meta)['image_shape'][0, :2]))(input_gt_boxes)
            filter = PositiveFilter(self.config, self.roi_count)
            rois = filter([rois, gt_boxes])
            inputs.append(input_gt_boxes)
        return KM.Model(inputs, [rois], name='refiner')
    
    def generate_detection_results(self, rois, gt_class_ids, gt_boxes, image_meta):
        """ Returns unnormalized roi coordinates of original image and assign rois with
        class ids according to their best match.
        """
        # Trim zeros.
        zero_ix = np.where(rois[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else rois.shape[0]
        rois = rois[:N, :4]
        non_zeros = np.sum(np.abs(gt_boxes), axis=1).astype(np.bool)
        gt_boxes = gt_boxes[non_zeros]
        gt_class_ids = gt_class_ids[non_zeros]
        # Assign class ids for each roi.
        overlaps = utils.compute_overlaps(rois, gt_boxes)
        roi_iou_max = np.max(overlaps, axis=1)
        pos_roi_bool = roi_iou_max >= 0.5
        rois = rois[pos_roi_bool]
        overlaps = overlaps[pos_roi_bool]
        roi_gt_box_assignment = np.argmax(overlaps, axis=1)
        roi_gt_classes = gt_class_ids[roi_gt_box_assignment]
        # Unnormalize and formalize outputs.
        m = modellib.parse_image_meta(image_meta)
        original_image_shape = m['original_image_shape'][0, :]
        image_shape = m['image_shape'][0, :]
        window = m['window'][0, :]
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        boxes = np.divide(rois - shift, scale)
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])
        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            roi_gt_classes = np.delete(roi_gt_classes, exclude_ix, axis=0)
            N = boxes.shape[0]
        masks = np.zeros(original_image_shape[:2] + (N,))
        scores = np.ones(N).astype(np.float32)
        return boxes, roi_gt_classes, scores, masks
    
    def visualize(self, limit=100, save=True, save_path='images/debug_stage1'): 
        for i in range(limit):
            inputs, _, target_class_ids = next(self.gen)
            image = inputs[0][0] + self.config.MEAN_PIXEL
            meta = inputs[1]
            image_id = modellib.parse_image_meta(meta)['image_id'][0]
            targets_tensor = inputs[2][0]
            targets = []
            for j in range(targets_tensor.shape[-1]):
                targets.append(targets_tensor[..., j] + self.config.MEAN_PIXEL)
            gt_boxes = inputs[-2]
            index = 7 if self.roi_type != 'train_roi' else 8
            rois = self.model.keras_model.predict(inputs)[index]
            print(rois.shape)
            refiner_inputs = [rois, meta]
            if self.positive_only:
                refiner_inputs.append(gt_boxes)
            rois = self.refiner.predict(refiner_inputs)[0]
            rois = rois[rois[:, 4].astype(np.int32) == 1]
            rois = rois[:, :4]
            fig = plt.figure()
            for j, target in enumerate(targets):
                ax = fig.add_subplot(len(targets), 2, j*2+1)
                ax.imshow(target.astype(np.uint8))
                ax.set_title(self.config.CLASS_ID[target_class_ids[j]])
            ax = fig.add_subplot(122)
            ax.imshow(image.astype(np.uint8))
            N = rois.shape[0]
            colors = visualize.random_colors(N)
            for j in range(N):
                color = colors[j]
                if not np.any(rois[i]):
                    continue
                y1, x1, y2, x2 = rois[j]
                y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
                p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
                ax.add_patch(p)
            if save:
                fig.savefig(save_path + '/' + str(image_id) + '.jpg')
            else:
                plt.show()
                
    def evaluate(self, limit=-1, verbose=1):
        assert self.roi_type != 'train_roi'
        if limit < 0:
            limit = len(self.dataset.image_ids)
        image_ids = self.dataset.image_ids[:limit]
        dataset_image_ids = [self.dataset.image_info[id]["id"] for id in image_ids]
        results = []
        real_image_ids = []
        for i in range(limit):
            if i % 100 == 0 and verbose > 1:
                print("Processing image {}/{} ...".format(i, limit))
            inputs, _, target_class_ids = next(self.gen)
            image = inputs[0][0]
            meta = inputs[1]
            image_id = modellib.parse_image_meta(meta)['image_id'][0]
            
            gt_class_ids, gt_boxes = inputs[-3], inputs[-2]
            rois = self.model.keras_model.predict(inputs)[8]
            refiner_inputs = [rois, meta]
            if self.positive_only:
                refiner_inputs.append(gt_boxes)
            rois = self.refiner.predict(refiner_inputs)[0]
            rois = self.refiner.predict(refiner_inputs)[0]
            gt_class_ids, gt_boxes = gt_class_ids[0], gt_boxes[0]
            final_rois, class_ids, scores, masks = self.generate_detection_results(rois, gt_class_ids, gt_boxes, meta)
            class_ids = np.array([target_class_ids[np.asscalar(j-1)] for j in class_ids])
            image_results = coco.build_coco_results(self.dataset, [self.dataset.image_info[image_id]["id"]],
                                                        final_rois, class_ids,
                                                        scores,
                                                        masks.astype(np.uint8))
            real_image_ids.append(image_id)
            results.extend(image_results)
        dataset_results = self.dataset_obj.loadRes(results)
        cocoEval = m_utils.customCOCOeval(self.dataset_object, dataset_results, "bbox")
        cocoEval.params.imgIds = dataset_image_ids[real_image_ids]
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize(verbose=verbose)
        
config = Config()
train_schedule = OrderedDict()
train_schedule[40] = {"learning_rate": config.LEARNING_RATE, "stage1": "n/a", "stage2": "cls"}
train_schedule[100] = {"learning_rate": config.LEARNING_RATE, "stage1": "n/a", "stage2": "mask"}
train_schedule[200] = {"learning_rate": config.LEARNING_RATE, "stage1": "n/a", "stage2": "all"}
MODEL_DIR = os.path.join(ROOT_DIR, "logs_debug")
train_classes, test_classes = [], []
for i in range(1, 81):
    if i % 5 == 0:
        test_classes.append(i)
    else:
        train_classes.append(i)
train_classes, test_classes = np.array(train_classes), np.array(test_classes)


parser = ArgumentParser(description='Debugger for stage1 rois.')
parser.add_argument("command", metavar="<command>",
                    help="metric or visualize")
parser.add_argument('--model', metavar='path/to/weight', required=False,
                    default='logs_imagenet/matching_mrcnn_coco_relation_lr=e-3/matching_mrcnn_0006.h5')
parser.add_argument('--limit', required=False, default=config.TRAIN_ROIS_PER_IMAGE,
                    help='number of images to be evaluated')
parser.add_argument('--classes', required=False, default='train',
                    help='which class split to use, train / val / all')
parser.add_argument('--dataset', required=False, default='val',
                    help='which dataset to use, train / val / all')
parser.add_argument('--verbose', default=1)
parser.add_argument('--roi_type', required=True, metavar='<train_roi|rpn_roi>',
                    help='which roi to be evaluated')
parser.add_argument('--pos_only', required=False, metavar='<True|False>',
                    default=False, help='whether remain roi with >0.5 iou only')
parser.add_argument('--pos_num', required=False, default=config.TRAIN_ROIS_PER_IMAGE,
                    help='number of boxes to remain after positive-only filter')
parser.add_argument('--save', required=False, default=True, metavar='<True|False>',
                    help='whether or not save visializations or not')
parser.add_argument('--ways', required=False, default='3')
parser.add_argument('--shots', required=False, default='1')
parser.add_argument('--attention', required=False, default='spatial')
args = parser.parse_args()

config.WAYS = int(args.ways)
config.SHOTS = int(args.shots)
config.SUPPORT_NUMBERS = config.WAYS * config.SHOTS
config.ATTENTION = args.attention
model = m_model.MatchingMaskRCNN(mode="training", model_dir=MODEL_DIR, config=config)
model.load_checkpoint(args.model, train_schedule)

assert args.classes in ['val', 'train', 'all'], 'Only val / train / all are supported so far.'
assert args.dataset in ['val', 'train', 'all'], 'Only val / train / all are supported so far.'
classes, datasets = [], []
if args.classes == 'val':
    classes = [test_classes]
elif args.classes == 'train':
    classes = [train_classes]
else:
    classes = [train_classes, test_classes]
dataset, data_obj = [], []
return_obj = args.command == 'metric'
if args.dataset == 'train' or args.dataset == 'all':
    coco_train = m_utils.IndexedCocoDataset()
    obj = coco_train.load_coco(COCO_DATA, "train", year="2017", return_coco=return_obj)
    coco_train.prepare()
    coco_train.build_indices()
    dataset.append(coco_train)
    if obj: data_obj.append(obj)
if args.dataset == 'val' or args.dataset == 'all':
    coco_val = m_utils.IndexedCocoDataset()
    obj = coco_val.load_coco(COCO_DATA, "val", year="2017", return_coco=return_obj)
    coco_val.prepare()
    coco_val.build_indices()    
    dataset.append(coco_val)
    if obj: data_obj.append(obj)
    
pos_num = int(args.pos_num)
pos_only = args.pos_only != 'True'
save = args.save != 'False'
verbose = int(args.verbose)
limit = int(args.limit)
assert args.roi_type in ['train_roi', 'rpn_roi'] and args.command in ['visualize', 'metric']
arg_dict = {'roi_type': args.roi_type, 'usage': args.command, 'pos_num': pos_num, 'pos_only': pos_only}
for i, d in enumerate(dataset):
    for c in classes:
        d.ACTIVE_CLASSES = c
        if len(data_obj) > 0: arg_dict['dataset_obj'] = data_obj[i]
        evaluater = ROIEval(model, d, arg_dict)
        if args.command == 'metric':
            evaluater.evaluate(limit=limit, verbose=verbose)
        else:
            evaluater.visualize(limit=limit, save=save)
            




 
                        
        