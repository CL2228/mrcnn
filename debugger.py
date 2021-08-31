"""
Author: Zhibo Fan
Version 2.0
"""
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


class TrainConfig(m_config.Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 0.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = 'coco'
    EXPERIMENT = 'random_repmet=e-3_buggy'
    CHECKPOINT_DIR = 'checkpoints/'
    # Adapt loss weights
    LOSS_WEIGHTS = {'rpn_class_loss': 2.0,
                    'rpn_bbox_loss': 0.1,
                    'mrcnn_class_loss': 2.0,
                    'mrcnn_embedding_loss': 1.0,
                    'mrcnn_bbox_loss': 0.5,
                    'mrcnn_mask_loss': 1.0}


config = TrainConfig()

train_schedule = OrderedDict()
train_schedule[40] = {"learning_rate": config.LEARNING_RATE, "stage1": "n/a", "stage2": "cls"}
train_schedule[100] = {"learning_rate": config.LEARNING_RATE, "stage1": "n/a", "stage2": "mask"}
train_schedule[200] = {"learning_rate": config.LEARNING_RATE, "stage1": "n/a", "stage2": "all"}

baseline = ['imagenet', 'rpn', 'frcnn'][2]
MODEL_DIR = os.path.join(ROOT_DIR, "logs_" + baseline)

def weight_check(model, baseline, no_head=False):
    import h5py
    print('checking weight consistency')
    real_model = model.keras_model
    if hasattr(real_model, 'inner_model'):
        real_model = real_model.inner_model
        
    def assertion(a1, a2, name, judge=True):
        if judge is True: func = lambda x, y: np.all(np.equal(x, y))
        else: func = lambda x, y: np.any(np.not_equal(x, y))
        if not func(a1, a2):
            print('first array: ', a1, '\n')
            print('second array: ', a2, '\n')
            print(name + ' check error')
            exit(-1)
            
    if baseline != 'imagenet':
        f = h5py.File('checkpoints/pretrained_frcnn.h5')
        layer_names = [str(n)[2 : -1] for n in f.attrs['layer_names']]
        # ResNet conv layers
        res_conv_names = [n for n in layer_names if n[:3] == 'res']
        res_conv_names.append('conv1')
        resnet = real_model.get_layer('resnet_model')
        for n in res_conv_names:
            if f[n].attrs['weight_names'].size == 0: continue
            file_kernel = f[n][n]['kernel:0'][:]
            file_bias = f[n][n]['bias:0'][:]
            model_kernel, model_bias = resnet.get_layer(n).get_weights()
            assertion(model_kernel, file_kernel, n + '_kernel')
            assertion(model_bias, file_bias, n + '_bias')
        # ResNet bn layers
        res_bn_names = [n for n in layer_names if n[:2] == 'bn']
        for n in res_bn_names:
            if f[n].attrs['weight_names'].size == 0: continue
            file_gamma = f[n][n]['gamma:0'][:]
            file_beta = f[n][n]['beta:0'][:]
            file_mov_mean = f[n][n]['moving_mean:0'][:]
            file_mov_var = f[n][n]['moving_variance:0'][:]
            model_gamma, model_beta, model_mov_mean, model_mov_var = resnet.get_layer(n).get_weights()
            assertion(model_gamma, file_gamma, n + '_gamma')
            assertion(model_beta, file_beta, n + '_beta')
            assertion(model_mov_mean, file_mov_mean, n + '_mov_mean')
            assertion(model_mov_var, file_mov_var, n + '_mov_var')
        # FPN layers
        fpn_names = [n for n in layer_names if n[:3] == 'fpn']
        fpn = real_model.get_layer('fpn_model')
        for n in fpn_names:
            if f[n].attrs['weight_names'].size == 0: continue
            file_kernel = f[n][n]['kernel:0'][:]
            file_bias = f[n][n]['bias:0'][:]
            model_kernel, model_bias = fpn.get_layer(n).get_weights()
            assertion(model_kernel, file_kernel, n + '_kernel')
            assertion(model_bias, file_bias, n + '_bias')
        # RPN layers
        rpn_names = ['rpn_bbox_pred', 'rpn_class_raw', 'rpn_conv_shared']
        rpn = real_model.get_layer('rpn_model')
        for n in rpn_names:
            file_kernel = f['rpn_model'][n]['kernel:0'][:]
            file_bias = f['rpn_model'][n]['bias:0'][:]
            model_kernel, model_bias = rpn.get_layer(n).get_weights()
            assertion(model_kernel, file_kernel, n + '_kernel')
            assertion(model_bias, file_bias, n + '_bias')
        
        # Heads
        if no_head: return
        judge = baseline == 'frcnn'
        heads_conv_names = ['mrcnn_class_conv' + str(i) for i in range(1, 3)]
        heads_bn_names = ['mrcnn_class_bn' + str(i) for i in range(1, 3)]
        for n in heads_conv_names:
            file_kernel = f[n][n]['kernel:0'][:]
            file_bias = f[n][n]['bias:0'][:]
            model_kernel, model_bias = real_model.get_layer(n).get_weights()
            assertion(model_kernel, file_kernel, n + '_kernel', judge=judge)
            assertion(model_bias, file_bias, n + '_bias', judge=judge)
        for n in heads_bn_names:
            file_gamma = f[n][n]['gamma:0'][:]
            file_beta = f[n][n]['beta:0'][:]
            file_mov_mean = f[n][n]['moving_mean:0'][:]
            file_mov_var = f[n][n]['moving_variance:0'][:]
            model_gamma, model_beta, model_mov_mean, model_mov_var = real_model.get_layer(n).get_weights()
            assertion(model_gamma, file_gamma, n + '_gamma', judge=judge)
            assertion(model_beta, file_beta, n + '_beta', judge=judge)
            # assertion(model_mov_mean, file_mov_mean, n + '_mov_mean', judge=judge)
            # assertion(model_mov_var, file_mov_var, n + '_mov_var', judge=judge)
    else:
        print('imagenet weights checking is not implemented')
        #f = h5py.File('checkpoints/imagenet_resnet_1k.h5')
    try: f.close()
    except: print('No file found: ' + baseline)
    print('checking done')
            

def display_random_rois(config, dataset,
                     limit=0, image_ids=None, random_rois=0):
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids
    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]
    # Get corresponding COCO image IDs.

    for i, image_id in enumerate(image_ids):
        # Load GT data
        resized_image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
            modellib.load_image_gt(dataset, config,
                                   image_id, augmentation=False, use_mini_mask=config.USE_MINI_MASK)
        
        # BOILERPLATE: Code duplicated in siamese_data_loader

        # Skip images that have no instances. This can happen in cases
        # where we train on a subset of classes and the image doesn't
        # have any of the classes we care about.
        if not np.any(gt_class_ids > 0):
            continue

        # Use only positive class_ids
        categories = np.unique(gt_class_ids)
        _idx = categories > 0
        categories = categories[_idx]
        # Use only active classes
        active_categories = []
        for c in categories:
            if any(c == dataset.ACTIVE_CLASSES):
                active_categories.append(c)

        # Skiop image if it contains no instance of any active class
        if not np.any(np.array(active_categories) > 0):
            continue

        # Filter out crowd class
        active_class_bool = np.zeros_like(gt_class_ids, dtype=np.bool)
        for x in active_categories:
            active_class_bool += gt_class_ids == x
        # Generate random rois instead of rpn
        rpn_gt_class_ids = gt_class_ids[active_class_bool]
        rpn_gt_boxes = gt_boxes[active_class_bool, :]
        rpn_rois = m_utils.generate_random_rois(
                resized_image.shape, random_rois, rpn_gt_boxes, ratio=0.96)
        category = np.random.choice(np.array(active_categories))
        # Draw random target
        targets = []
        tbs = []
        all_cats = [category]
        for j in range(config.WAYS - 1):
            while True:
                rest_cats = np.random.choice(dataset.ACTIVE_CLASSES)
                if rest_cats not in active_categories:
                    break
            all_cats.append(rest_cats)
        for c in all_cats:
            while True:
                target, tb = m_utils.get_one_target(c, dataset, config)[:2]
                if target is not None and tb is not None:
                    break
            targets.append(target)
            tbs.append(tb)
        # Resize generated targets
        targets = np.stack(targets, axis=0)
        tbs = np.stack(tbs, axis=0)
        
        # Prepare parameters for display function
        if np.shape(dataset.ACTIVE_CLASSES)[0] == 16:
            classes = 'val'
        else:
            classes = 'train'
        try:
            m_utils.display_results(targets, tbs, all_cats, resized_image,
                                    rpn_rois, None, np.ones(random_rois), gt_class_ids, gt_boxes,
                                    ways=3, path='images/debug_random_rois_{}'.format(classes), config=config)
        except:
            os.system('mkdir images/debug_random_rois_' + classes)
            m_utils.display_results(targets, tbs, all_cats, resized_image,
                                    rpn_rois, None, np.ones(random_rois), rpn_gt_class_ids, rpn_gt_boxes,
                                    ways=3, path='images/debug_random_rois_{}'.format(classes), config=config)  

class RandomROIMetric:
    def __init__(self, dataset, limit=0, image_ids=None, random_rois=1000, ways=3, 
                 few_shot_sets=10, random_roi_ratio=0.5):
        self.image_ids = image_ids or dataset.image_ids
        self.image_ids = self.image_ids[:limit]
        self.dataset = dataset
        
        self.random_rois = random_rois
        self.tps = [{
            "maxdet1": 0.,
            "maxdet10": 0.,
            "maxdet100": 0.,
            "small": 0.,
            "medium": 0.,
            "large": 0.,
        } for i in range(10)]
        self.fns = [{
            "maxdet1": 0.,
            "maxdet10": 0.,
            "maxdet100": 0.,
            "small": 0.,
            "medium": 0.,
            "large": 0.,
        } for i in range(10)]
        self.detections = {
            "overall": 0,
            "small": 0,
            "medium": 0,
            "large": 0
        }
        self.true_det = [{
            "overall": 0,
            "small": 0,
            "medium": 0,
            "large": 0
        } for i in range(10)]
        self.few_shot_tps = [{
            "maxdet1": 0.,
            "maxdet10": 0.,
            "maxdet100": 0.,
            "small": 0.,
            "medium": 0.,
            "large": 0.,
        } for i in range(10)]
        self.few_shot_fns = [{
            "maxdet1": 0.,
            "maxdet10": 0.,
            "maxdet100": 0.,
            "small": 0.,
            "medium": 0.,
            "large": 0.,
        } for i in range(10)]
        self.few_shot_detections = {
            "overall": 0,
            "small": 0,
            "medium": 0,
            "large": 0
        }
        self.few_shot_true_det = [{
            "overall": 0,
            "small": 0,
            "medium": 0,
            "large": 0
        } for i in range(10)]
        self.dets = {
            "small": 0,
            "medium": 0,
            "large": 0
        }
        self.ways = ways
        self.ratio = random_roi_ratio

    def _compute_overlaps(self, pred, gt):
        """
        Returns a iou matrix where prediction are on the y axis.
        :return: [num_pred, num_gt]
        """
        b1 = np.reshape(np.tile(np.expand_dims(pred, 1),
                                [1, 1, np.shape(gt)[0]]), [-1, 4])
        b2 = np.tile(gt, [np.shape(pred)[0], 1])
        b1_y1, b1_x1, b1_y2, b1_x2 = np.split(b1, 4, axis=1)
        b2_y1, b2_x1, b2_y2, b2_x2 = np.split(b2, 4, axis=1)
        y1 = np.maximum(b1_y1, b2_y1)
        x1 = np.maximum(b1_x1, b2_x1)
        y2 = np.minimum(b1_y2, b2_y2)
        x2 = np.minimum(b1_x2, b2_x2)
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
        b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
        union = b1_area + b2_area - intersection
        iou = intersection / union
        overlaps = np.reshape(iou, [np.shape(pred)[0], np.shape(gt)[0]])
        return overlaps

    def _parse_shape(self, boxes, image_meta):
        """
        This function parses box size to small, medium and large and represent with
        0, 1, 2 responsively.
        :param boxes: [N, 4] in normalized coordination
        :param image_meta: meta contains shape information of origin image
        :return: vector [N] with 0, 1, 2 corresponding to small, medium, large
        """
        bw, bh = boxes[:, 3] - boxes[:, 1], boxes[:, 2] - boxes[:, 0]
        area = bw * bh
        levels = np.ones_like(area)
        levels[area < 32 * 32] = 0
        levels[area > 96 * 96] = 2
        return levels

    def _compute_recalls(self, overlaps, levels, iou, return_result=False):
        """Returns true positives and false negatives at current iou threshold.
            :return: [tp@maxdet=1, tp@maxdet=10, tp@maxdet=100, fn@overall
                      tp@small, fn@small, tp@medium, fn@medium, tp@large, fn@large],
                      each data includes number of true positives and false negatives.
        """
        ix = round((iou - 0.5) * 20)
        true_pos = np.zeros_like(overlaps)
        true_pos[overlaps >= iou] = 1
        origin_true_pos = np.sum(true_pos, axis=0)
        origin_true_pos_maxdet1 = np.minimum(origin_true_pos, 1)
        origin_true_pos_maxdet10 = np.minimum(origin_true_pos, 10)
        origin_true_pos_maxdet100 = np.minimum(origin_true_pos, 100)
        false_neg = np.shape(origin_true_pos == 0)[0]
        if not return_result:
            self.tps[ix]["maxdet1"] += np.asscalar(np.sum(origin_true_pos_maxdet1))
            self.tps[ix]["maxdet10"] += np.asscalar(np.sum(origin_true_pos_maxdet10))
            self.tps[ix]["maxdet100"] += np.asscalar(np.sum(origin_true_pos_maxdet100))
            self.fns[ix]["maxdet1"] += false_neg
            self.fns[ix]["maxdet10"] += false_neg
            self.fns[ix]["maxdet100"] += false_neg
        else:
            res = [np.asscalar(np.sum(origin_true_pos_maxdet1)), np.asscalar(np.sum(origin_true_pos_maxdet10)),
                   np.asscalar(np.sum(origin_true_pos_maxdet100)), false_neg]
        for i, key in zip(range(3), ["small", "medium", "large"]):
            tp_vec = origin_true_pos_maxdet100[levels == i]
            tp = np.sum(tp_vec)
            fn = np.shape(tp_vec == 0)[0]
            if not return_result:
                self.tps[ix][key] += np.asscalar(tp)
                self.fns[ix][key] += fn
            else:
                res.append(np.asscalar(tp))
                res.append(fn)
        if return_result:
            return res

    def _compute_precisions(self, overlaps, levels, iou, return_result=False):
        """Returns true positives and total detections at current iou threshold.
           :return: [tp@overall, det@overall, tp@small, det@small,
                     tp@medium, det@medium, tp@large, det@large],
                     each data includes number of true positives and total detections.
        """
        ix = round((iou - 0.5) * 20)
        true_pos = np.zeros_like(overlaps)
        true_pos[overlaps >= iou] = 1
        true_pos = np.minimum(np.sum(true_pos, axis=-1), 100)
        if not return_result:
            tp = np.asscalar(np.sum(true_pos))
            self.true_det[ix]["overall"] += tp
            self.detections["overall"] += (np.shape(true_pos == 0)[0] + tp) / 10
        else:
            res = [np.asscalar(np.sum(true_pos)), np.shape(true_pos)[0]]
        for i, key in zip(range(3), ["small", "medium", "large"]):
            tp_vec = true_pos[levels == i]
            if not return_result:
                tp = np.asscalar(np.sum(tp_vec))
                self.true_det[ix][key] += tp
                self.detections[key] += (np.shape(tp_vec == 0)[0] + tp) / 10
            else:
                res.append(np.asscalar(np.sum(tp_vec)))
                res.append(np.shape(tp_vec)[0])
        if return_result:
            return res

    def _gather_results(self, rois, gt, image_meta):
        # Compute non few-shot metrics
        overlaps = self._compute_overlaps(rois, gt)
        ious = np.arange(0.5, 1.0, 0.05).tolist()
        roi_levels = self._parse_shape(rois, image_meta)
        gt_levels = self._parse_shape(gt, image_meta)
        for iou in ious:
            self._compute_precisions(overlaps, roi_levels, iou)
            self._compute_recalls(overlaps, gt_levels, iou)

    def _gather_few_shot_results(self, rois, target_gt, image_meta):
        overlaps = self._compute_overlaps(rois, target_gt)
        ious = np.arange(0.5, 1.0, 0.05).tolist()
        roi_levels = self._parse_shape(rois, image_meta)
        gt_levels = self._parse_shape(target_gt, image_meta)
        for iou in ious:
            ix = round((iou - 0.5) * 20)
            prec_res = self._compute_precisions(overlaps, roi_levels, iou, return_result=True)
            rec_res = self._compute_recalls(overlaps, gt_levels, iou, return_result=True)
            self.few_shot_tps[ix]["maxdet1"] += rec_res[0]
            self.few_shot_tps[ix]["maxdet10"] += rec_res[1]
            self.few_shot_tps[ix]["maxdet100"] += rec_res[2]
            self.few_shot_fns[ix]["maxdet1"] += rec_res[3]
            self.few_shot_fns[ix]["maxdet10"] += rec_res[3]
            self.few_shot_fns[ix]["maxdet100"] += rec_res[3]
            self.few_shot_tps[ix]["small"] += rec_res[4]
            self.few_shot_fns[ix]["small"] += rec_res[5]
            self.few_shot_tps[ix]["medium"] += rec_res[6]
            self.few_shot_fns[ix]["medium"] += rec_res[7]
            self.few_shot_tps[ix]["large"] += rec_res[8]
            self.few_shot_fns[ix]["large"] += rec_res[9]
            self.few_shot_true_det[ix]["overall"] += prec_res[0]
            self.few_shot_detections["overall"] += prec_res[1] / 10 # divide by 10 to avoid duplication
            self.few_shot_true_det[ix]["small"] += prec_res[2]
            self.few_shot_detections["small"] += prec_res[3] / 10
            self.few_shot_true_det[ix]["medium"] += prec_res[4]
            self.few_shot_detections["medium"] += prec_res[5] / 10
            self.few_shot_true_det[ix]["large"] += prec_res[6]
            self.few_shot_detections["large"] += prec_res[7] / 10

    def _random_categories(self, unique_classes, few_shot_sets=10):
        support_sets = []
        for i in range(few_shot_sets):
            support_set = []
            for j in range(self.ways):
                while True:
                    cls = np.random.choice(unique_classes)
                    if cls not in support_set:
                        support_set.append(np.asscalar(cls))
                        break
                support_set.append(cls)
            support_sets.append(np.array(support_set))
        return support_sets

    def evaluate(self, verbose=1):
        import time
        self.time = time.time()
        for i, id in enumerate(self.image_ids):
            if verbose > 0 and i % 100 == 0:
                print('evaluating {}th image out of {} images'.format(i, len(self.image_ids)))
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                modellib.load_image_gt(self.dataset, config, id, augmentation=None,
                                       use_mini_mask=config.USE_MINI_MASK)
            image_meta = np.expand_dims(image_meta, axis=0)
            t = time.time()
            if not np.any(gt_class_ids > 0):
                continue
            # Filter crowd classes
            categories = np.unique(gt_class_ids)
            _idx = categories > 0
            categories = categories[_idx]
            class_bool = np.zeros_like(gt_class_ids, dtype=np.bool)
            for x in categories:
                class_bool += gt_class_ids == x
            rpn_gt_class = gt_class_ids[class_bool]
            rpn_gt_boxes = gt_boxes[class_bool, :]
            # Generate support sets and rois
            support_sets = self._random_categories(categories)

            rpn_rois = m_utils.generate_random_rois(
                image.shape, self.random_rois, rpn_gt_boxes, ratio=self.ratio)
            image_meta = np.expand_dims(image_meta, axis=0)
            self._gather_results(rpn_rois, rpn_gt_boxes, image_meta)
            for s in support_sets:
                gt_boxes = []
                for cls in s:
                    idx = rpn_gt_class == cls
                    gt_box = rpn_gt_boxes[idx, :]
                    gt_boxes.append(gt_box)
                gt_boxes = np.concatenate(gt_boxes, axis=0)
                self._gather_few_shot_results(rpn_rois, gt_boxes, image_meta)
            self.time += time.time() - t
        print(self.tps)
        print('\n', self.fns)
        self.precision, self.few_shot_precision, self.recall, self.few_shot_recall = [dict() for i in range(4)]
        def _summary_prec(obj, gt_set='all'):
            prec = obj.precision if gt_set == 'all' else obj.few_shot_precision
            true_det = obj.true_det if gt_set == 'all' else obj.few_shot_true_det
            det = obj.detections if gt_set == 'all' else obj.few_shot_detections
            prec["mAP@50"] = true_det[0]["overall"] / det["overall"]
            prec["mAP@75"] = true_det[5]["overall"] / det["overall"]
            prec["mAP"] = sum([true_det[ix]["overall"] / det["overall"] \
                                         for ix in range(10)]) / 10
            prec["small"] = sum([true_det[ix]["small"] / det["small"] \
                                         for ix in range(10)]) / 10
            prec["medium"] = sum([true_det[ix]["medium"] / det["medium"] \
                                           for ix in range(10)]) / 10
            prec["large"] = sum([true_det[ix]["large"] / det["large"] \
                                            for ix in range(10)]) / 10
        _summary_prec(self)
        _summary_prec(self, gt_set='few-shot')
        def _summary_recall(obj, gt_set='all'):
            recall = obj.recall if gt_set == 'all' else obj.few_shot_recall
            tps = obj.tps if gt_set == 'all' else obj.few_shot_tps
            fns = obj.fns if gt_set == 'all' else obj.few_shot_fns
            recall["maxdet1"] = sum([tps[ix]["maxdet1"] / (tps[ix]["maxdet1"] + fns[ix]["maxdet1"]) \
                                         for ix in range(10)]) / 10
            recall["maxdet10"] = sum([tps[ix]["maxdet10"] / (tps[ix]["maxdet10"] + fns[ix]["maxdet10"]) \
                                     for ix in range(10)]) / 10
            recall["maxdet100"] = sum([tps[ix]["maxdet100"] / (tps[ix]["maxdet100"] + fns[ix]["maxdet100"]) \
                                      for ix in range(10)]) / 10
            recall["small"] = sum([tps[ix]["small"] / (tps[ix]["small"] + fns[ix]["small"]) \
                                      for ix in range(10)]) / 10
            recall["medium"] = sum([tps[ix]["medium"] / (tps[ix]["medium"] + fns[ix]["medium"]) \
                                   for ix in range(10)]) / 10
            recall["large"] = sum([tps[ix]["large"] / (tps[ix]["large"] + fns[ix]["large"]) \
                                   for ix in range(10)]) / 10
        _summary_recall(self)
        _summary_recall(self, gt_set='few-shot')
        self.time = time.time() - self.time
    
    def display(self):
        print('========================= SUMMARY =========================')
        line = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} | {:<6}] = {:0.3f}'
        def _display(obj, gt_set='all'):
            prec = obj.precision if gt_set == 'all' else obj.few_shot_precision
            recall = obj.recall if gt_set == 'all' else obj.few_shot_recall
            print(line.format('Average Precision', '(AP)', '0.50:0.95', 'all', 100, gt_set, prec["mAP"]))
            print(line.format('Average Precision', '(AP)', 0.5, 'all', 100, gt_set, prec["mAP@50"]))
            print(line.format('Average Precision', '(AP)', 0.75, 'all', 100, gt_set, prec["mAP@75"]))
            print(line.format('Average Precision', '(AP)', '0.50:0.95', 'small', 100, gt_set, prec["small"]))
            print(line.format('Average Precision', '(AP)', '0.50:0.95', 'medium', 100, gt_set, prec["medium"]))
            print(line.format('Average Precision', '(AP)', '0.50:0.95', 'large', 100, gt_set, prec["large"]))
            print(line.format('Average Recall', '(AR)', '0.50:0.95', 'all', 1, gt_set, recall["maxdet1"]))
            print(line.format('Average Recall', '(AR)', '0.50:0.95', 'all', 10, gt_set, recall["maxdet10"]))
            print(line.format('Average Recall', '(AR)', '0.50:0.95', 'all', 100, gt_set, recall["maxdet100"]))
            print(line.format('Average Recall', '(AR)', '0.50:0.95', 'small', 100, gt_set, recall["small"]))
            print(line.format('Average Recall', '(AR)', '0.50:0.95', 'medium', 100, gt_set, recall["medium"]))
            print(line.format('Average Recall', '(AR)', '0.50:0.95', 'large', 100, gt_set, recall["large"]))
        _display(self)
        print('========================= FEWSHOT =========================')
        _display(self, '{}-way'.format(self.ways))
        print('Total time cost is {:4.2f}'.format(self.time))
        print('===================== END OF SUMMARY ======================')
        
if __name__ == '__main__':
    # display_random_rois(model, coco_eval, limit=10, random_rois=30)
    import argparse

    parser = argparse.ArgumentParser(
        description='Debugger for random rois.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'display' or 'metric' or 'target'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/coco/",
                        help='Directory of the MS-COCO dataset')
    parser.add_argument('--rois', required=False,
                        help='number of random rois to generate')
    parser.add_argument('--limit', required=False,
                        help='number of images to be evaluated')
    parser.add_argument('--nb_set', required=False,
                        help='number of few shot groups to be evaluated')
    parser.add_argument('--ratio', required=False,
                        help='ratio of positive rois to total random rois')
    parser.add_argument('--verbose', required=False,
                        help="'0' to close logs and '1' to open up logs, default is 1")
    parser.add_argument('--ways', required=False,
                        help='number of ways')
    parser.add_argument('--classes', required=False, help='train / val')
    args = parser.parse_args()

    if args.dataset is not None: COCO_DATA = args.dataset
    train_classes = []
    eval_classes = []
    for i in range(1, 81):
        if i % 5 != 0:
            train_classes.append(i)
        else:
            eval_classes.append(i)
    train_classes = np.array(train_classes)
    eval_classes = np.array(eval_classes)
    coco_eval = m_utils.IndexedCocoDataset()
    coco_eval.load_coco(COCO_DATA, "val", year="2017")
    coco_eval.prepare()
    coco_eval.build_indices()
    coco_eval.ACTIVE_CLASSES = eval_classes
    if args.classes is not None and args.classes == 'train':
        coco_eval.ACTIVE_CLASSES = train_classes 

    limit = 5000 if args.limit is None else int(args.limit)
    nb_set = 10 if args.nb_set is None else int(args.nb_set)
    ways = 1
    rois = config.POST_NMS_ROIS_INFERENCE if args.rois is None else int(args.rois)
    ratio = 0.5 if args.ratio is None else float(args.ratio)
    verbose = 1 if args.verbose is None else int(args.verbose)
    if args.command == "display":
        display_random_rois(config, coco_eval, limit=limit, random_rois=rois)
    else:
        metrics = RandomROIMetric(coco_eval, limit=limit, random_rois=rois,
                                  ways=ways, few_shot_sets=nb_set, random_roi_ratio=ratio)
        metrics.evaluate(verbose=verbose)
        metrics.display()