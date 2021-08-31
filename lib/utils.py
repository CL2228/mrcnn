import tensorflow as tf
import sys
import os
import shutil
import time
import random
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import skimage.io
import skimage.transform as skt
import imgaug
import json
from PIL import Image
plt.rcParams['figure.figsize'] = (12.0, 6.0)

MASK_RCNN_MODEL_PATH = 'Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)
    
from samples.coco import coco
from .mrcnn import utils
from .mrcnn import model as modellib
from .mrcnn import visualize

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from keras.applications.vgg16 import preprocess_input
import warnings
warnings.filterwarnings("ignore")

def mold_image(images, config):
    return images.astype(np.float32) - config.MEAN_PIXEL

def unmold_image(normalized_images, config):
    """Takes a image normalized with mold() and returns the original."""
    return (normalized_images + config.MEAN_PIXEL).astype(np.uint8)

### Data Generator ###
def load_support_gt(dataset, config, image_id, category, 
                    target_size_limit=0.001, augmentation=None, use_mini_mask=True):
    """
    Returns image, image_meta, bbox, mask, same as load_image_gt but selected with given category
    and cropped with corresponding contexts as well.
    Returns None if the target is smaller than expected, i.e., target-image ratio less than TARGET_SIZE_LIMIT.
    """
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)
    
    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Get 1 random mask of the category.
    ind = np.random.choice(np.where(class_ids == category)[0])
    mask = mask[:, :, ind : ind+1] # Trick to remain the first dimension to facilitate utils.
    class_ids = class_ids[ind : ind+1] # [1,]
    # Active classes (Not used)
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1
    
    # Get target window with 16px contexts.
    target_window = utils.extract_bboxes(mask)[0]
    target_size = np.sum(mask[..., 0])
    target_ratio = target_size / (image.shape[0] * image.shape[1])
    if target_ratio <= target_size_limit:
        return None
    target_window += np.array([-16, -16, 16, 16])
    target_window = np.maximum(0, target_window).astype(np.int32)
    # Crop target with its mask
    image = image[target_window[0]:target_window[2], target_window[1]:target_window[3], :]
    mask = mask[target_window[0]:target_window[2], target_window[1]:target_window[3], :]
    original_image_shape = image.shape
    resized_image, window, scale, padding, crop = utils.resize_image(
        image, min_dim=config.TARGET_MIN_DIM, max_dim=config.TARGET_MAX_DIM,
        min_scale=config.IMAGE_MIN_SCALE, mode=config.IMAGE_RESIZE_MODE)
    resized_mask = utils.resize_mask(mask, scale, padding, crop)
    bbox = utils.extract_bboxes(resized_mask)
    if use_mini_mask:
        mask = utils.minimize_mask(bbox, resized_mask, config.MINI_MASK_SHAPE)
        
    #rect = patches.Rectangle(bbox[0,:2][::-1], bbox[0,3]-bbox[0,1], bbox[0,2]-bbox[0,0], linewidth=2, edgecolor='r', facecolor='none')
    #fig = plt.figure()
    #ax = fig.add_subplot(121)
    #ax.imshow(resized_image)
    #ax.add_patch(rect)
    #ax = fig.add_subplot(122)
    #ax.imshow(mask[..., 0])
    #plt.show()
    image_meta = modellib.compose_image_meta(image_id, original_image_shape, resized_image.shape,
                                             window, scale, active_class_ids)
    return resized_image, image_meta, ind, bbox[0,:], mask[...,0]

def get_one_target_with_ids(category, dataset, config, augmentation=None,
                            target_size_limit=0, max_attempts=10, get_mask=False):
    """
    adapted from siamese mask rcnn lib utils.py
    generate target image by masking and resize to 224*224 as an ordinary input size of vgg
        param: mode: mask - mask the instance from the original image. Causing most places black.
                    crop - crop the instance from the original image. Same as Siamese mrcnn.
                    mask&crop - mask then crop. Default implementation.
    """
    #assert mode in ['mask', 'crop', 'mask&crop', 'crop&mask'] # the latter two options are identical
    n_attempts = 0
    target_mask = None 
    while True:
        n_attempts = n_attempts + 1
        # Get index with corresponding images for each category
        category_image_index = dataset.category_image_index
        # Draw a random image
        random_image_id = np.random.choice(category_image_index[category])
        rtns = load_support_gt(dataset, config, random_image_id, category, 
                               augmentation=augmentation, use_mini_mask=config.USE_MINI_MASK)
        if rtns is None: 
            continue
        target_image, target_meta, ind, target_boxes, target_masks = rtns
        
        if n_attempts >= max_attempts:
            break
        if get_mask:
            target_mask = target_masks
    return (target_image, target_boxes, target_meta[:13], target_mask,  
           np.asscalar(random_image_id), np.asscalar(ind)) # scalar id instead of ndarray

def get_one_target(category, dataset, config, augmentation=None, target_size_limit=0, max_attempts=10, get_mask=False):
    rtns = list(get_one_target_with_ids(category, dataset, config, augmentation,
                                    target_size_limit, max_attempts, get_mask))
    if get_mask: return rtns[:5]
    else:
        res = rtns[:3]
        res.append(rtns[4])
        return res

def get_k_targets(category, dataset, config, shots, augmentation=None, target_size_limit=0, max_attempts=10, get_mask=False):
    targets = []
    tbs = [] # tbs is short for a list including target boxes
    if get_mask: target_masks = []
    targets_ids = []
    instance_ids = []
    k = 0
    while True:
        target, tb, meta, mask, target_id, inst_id = get_one_target_with_ids(category, dataset, config, augmentation,
                                                                         target_size_limit, max_attempts, get_mask)
        if target_id in targets_ids and inst_id in instance_ids:
            print('duplicated sample, throw it')
            continue
        targets.append(target)
        tbs.append(tb)
        if get_mask: target_masks.append(mask)
        targets_ids.append(targets_ids)
        instance_ids.append(inst_id)
        k += 1
        if k >= shots:
            rtn = [targets, tbs, meta]
            if get_mask: rtn.append(target_masks)
            return rtn

def matching_data_generator(dataset, config, shuffle=True, augmentation=imgaug.augmenters.Fliplr(0.5), random_rois=0,
                            batch_size=1, detection_targets=False, get_mask=False, mode='normal'):
    # NOTICE: This is very IMPORTANT, read first before you do ANY modification.
    # This function is adapted from siamese mrcnn. If any code is problematic and you want to
    # modify any, please refer to their implementation and make sure you fully understand that.
    # Some of the codes are tough to comment on, sorry.

    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.
    NOTE: The images are processed by rescaling, centering as pretrained backbones.
    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: If True, applies image augmentation to images (currently only
            horizontal flips are supported)
    random_rois: If > 0 then generate proposals to be used to train the
                network classifier and mask heads. Useful if training
                the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    diverse: Float in [0,1] indicatiing probability to draw a target
        from any random class instead of one from the image classes
    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The containtes
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - targets: [batch, H, W, C, ways]
    - image_meta: [batch, size of image meta]
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.
    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = modellib.compute_backbone_shapes(config, config.IMAGE_SHAPE)
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinately.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Load image
            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                modellib.load_image_gt(dataset, config, image_id, augmentation=augmentation,
                                        use_mini_mask=config.USE_MINI_MASK)
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
            
            # Skip image if it contains no instance of any active class
            if not np.any(np.array(active_categories) > 0):
                continue
            
            # Randomly select a category as first target class
            category = np.random.choice(active_categories)
            # Dictionary maps target id with target image
            target_dict = dict()
            tbs_dict = dict()
            target_meta_dict = dict()
            if get_mask: tms_dict = dict()
            # Sample all target classes
            restart = 0    # restart count
            all_cats = [category]
            for i in range(config.WAYS - 1): # Sample other target classes.
                while True:
                    cat = np.random.choice(dataset.ACTIVE_CLASSES)
                    if cat not in all_cats:
                        all_cats.append(cat)
                        break
            # Generate RPN ground truths
            active_class_bool = np.zeros_like(gt_class_ids, dtype=np.bool)
            for x in all_cats:
                active_class_bool += gt_class_ids == x
            rpn_gt_class = gt_class_ids[active_class_bool]
            rpn_gt_boxes = gt_boxes[active_class_bool, :]
            # TODO: visualize rpn ground truths with faster rcnn for comparision
            # RPN ground truths
            # Change: different from siamese mrcnn, we are real fg/bg detector at rpn
            rpn_match, rpn_bbox = modellib.build_rpn_targets(image.shape, anchors,
                                                             rpn_gt_class, rpn_gt_boxes, config)
            # Pick support images
            for i in range(config.WAYS):
                cat = all_cats[i]
                if config.SHOTS == 1:
                    if not get_mask:
                        target, tb, target_meta, _ = get_one_target(cat, dataset, config,
                                            augmentation=augmentation, get_mask=get_mask)
                    else:
                        target, tb, target_meta, mask, _ = get_one_target(cat, dataset, config,
                                                          augmentation=augmentation, get_mask=get_mask)
                    if target is None:
                        print('skip rest target')
                        restart = 1
                else:
                    if not get_mask:
                        target, tb, target_meta = get_k_targets(cat, dataset, config, config.SHOTS,
                                            augmentation=augmentation, get_mask=get_mask)
                    else:
                        target, tb, target_meta, mask = get_k_targets(cat, dataset, config, config.SHOTS,
                                                         augmentation=augmentation, get_mask=get_mask)
                    if (target is None or len(target) != config.SHOTS):
                        print('skip rest target')
                        restart = 1
                target_dict[cat] = target
                tbs_dict[cat] = tb
                target_meta_dict[cat] = target_meta
                if get_mask : tms_dict[cat] = mask
            if restart: continue

            # Shuffle targets to avoid same order
            target_class_ids = list(target_dict.keys())
            random.shuffle(target_class_ids)
            # NOTE: Variable TARGETS is a list of lists if config.shots > 1
            targets = [target_dict[idx] for idx in target_class_ids]
            tbs = [tbs_dict[idx] for idx in target_class_ids]
            targets_meta = [target_meta_dict[idx] for idx in target_class_ids]
            if get_mask: tms = [tms_dict[idx] for idx in target_class_ids]

            # Initialize a list to contain gt of instances of different class
            match_class_ids_list = []
            gt_class_ids_list = []
            gt_boxes_list = []
            gt_masks_list = []
            for i in range(len(target_class_ids)):
                # idx is a bool list to choose match class' instance
                idx = gt_class_ids == target_class_ids[i]
                bool_class_ids = idx.astype('int8')[idx]
                num_instances = len(bool_class_ids)

                if num_instances == 0: continue # No instance shares the same class as the ith shot
                match_class_id = i + 1 # ground truth for stage2 classifier is "the current order + 1" of the target!
                # choose the matching instances' gt
                gt_class_id = gt_class_ids[idx] # The real gt class id is used for generating gt for rpn and roi
                gt_box = gt_boxes[idx, :]
                gt_mask = gt_masks[:, :, idx]

                for j in range(num_instances):
                    match_class_ids_list.append(match_class_id)
                gt_class_ids_list.append(gt_class_id)
                gt_boxes_list.append(gt_box)
                gt_masks_list.append(gt_mask)

            match_class_ids = np.array(match_class_ids_list)
            gt_class_ids = np.concatenate(gt_class_ids_list, axis=0)
            gt_boxes = np.concatenate(gt_boxes_list, axis=0)
            gt_masks = np.concatenate(gt_masks_list, axis=-1)
            image_meta = image_meta[:13] # only the first 12 data are needed, the 13th element is included to support original mrcnn functions

            # Mask R-CNN Targets
            if random_rois:
                rpn_rois = generate_random_rois(
                    image.shape, random_rois, rpn_gt_boxes, ratio=0.5)
                if detection_targets:
                    rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
                        modellib.build_detection_targets(
                            rpn_rois, match_class_ids, gt_boxes, gt_masks, config)

            # Init batch arrays
            if b == 0:
                batch_targets_meta = np.zeros(
                    (batch_size,) + image_meta.shape + (config.SUPPORT_NUMBER,), dtype=image_meta.dtype)
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                if config.SHOTS == 1:
                    batch_targets = np.zeros(
                        (batch_size,) + target.shape + (config.WAYS,), dtype=np.float32)
                    batch_tbs = np.zeros(
                        (batch_size, 4, config.WAYS), dtype=np.float32)
                    if get_mask:
                        batch_masks = np.zeros(
                            (batch_size,) + mask.shape + (config.WAYS,), dtype=np.float32)
                else:
                    batch_targets = np.zeros(
                        (batch_size,) + target[0].shape + (config.SUPPORT_NUMBER,), dtype=np.float32)
                    batch_tbs = np.zeros(
                        (batch_size,) + (4, config.SUPPORT_NUMBER,), dtype=np.float32)
                    if get_mask:
                        batch_masks = np.zeros(
                            (batch_size,) + mask[0].shape + (config.SUPPORT_NUMBER,), dtype=np.float32)
                if config.USE_MINI_MASK:
                    batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],
                                                config.MAX_GT_INSTANCES))
                else:
                    batch_gt_masks = np.zeros(
                        (batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))
                if random_rois: # DEPRECATED!
                    batch_rpn_rois = np.zeros(
                        (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                    if detection_targets:
                        batch_rois = np.zeros(
                            (batch_size,) + rois.shape, dtype=rois.dtype)
                        batch_mrcnn_class_ids = np.zeros(
                            (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                        batch_mrcnn_bbox = np.zeros(
                            (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                        batch_mrcnn_mask = np.zeros(
                            (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                match_class_ids = match_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]
            
            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32), config)
            for i in range(config.WAYS):
                if config.SHOTS == 1:
                    batch_targets[b, :, :, :, i] = mold_image(targets[i].astype(np.float32), config)
                    batch_tbs[b, :, i] = tbs[i]
                    batch_targets_meta[b, :, i] = targets_meta[i]
                    if get_mask:
                        batch_masks[b, :, :, i] = tms[i]
                else:
                    for j, inst in enumerate(targets[i]):
                        batch_targets[b, :, :, :, i * config.SHOTS + j] = mold_image(
                            inst.astype(np.float32), config)
                    for j, inst in enumerate(tbs[i]):
                        batch_tbs[b, :, i * config.SHOTS + j] = inst
                    for j, inst in enumerate(targets_meta[i]):
                        batch_targets_meta[b, :, i * config.SHOTS + j] = inst
                    if get_mask:
                        for j, inst in enumerate(tms[i]):
                            batch_masks[b, :, :, i * config.SHOTS + j] = inst
            batch_gt_class_ids[b, :match_class_ids.shape[0]] = match_class_ids
            # batch_target_class_ids[b, :target_class_ids.shape[0]] = target_class_ids # for debug
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            if random_rois:
                batch_rpn_rois[b] = rpn_rois
                if detection_targets:
                    batch_rois[b] = rois
                    batch_mrcnn_class_ids[b] = mrcnn_class_ids
                    batch_mrcnn_bbox[b] = mrcnn_bbox
                    batch_mrcnn_mask[b] = mrcnn_mask
            
            
            if mode == 'save':
                targets = np.transpose(batch_targets[b], [3, 0, 1, 2])
                tbs = np.transpose(batch_tbs[b], [1, 0])
                print('tbs.shape: ', tbs.shape)
                print('all_cats.shape: ', batch_gt_class_ids[b])
                print('image.shape: ', batch_images[b].shape)
                print('gt_box: ', batch_gt_boxes[b])
                print('gt_class_ids.shape: ', batch_gt_class_ids[b].shape)
                display_results(unmold_image(targets, config), tbs, target_class_ids, unmold_image(batch_images[b], config), 
                                batch_gt_boxes[b], None, batch_gt_class_ids[b], 3, 'gen_image', config)

            b += 1

            # Batch full?
            if b >= batch_size:
                if not get_mask:
                    inputs = [batch_images, batch_image_meta, batch_targets, batch_targets_meta, 
                            batch_tbs, batch_rpn_match, batch_rpn_bbox,
                            batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                else:
                    inputs = [batch_images, batch_image_meta, batch_targets, batch_targets_meta,
                              batch_tbs, batch_masks, batch_rpn_match, batch_rpn_bbox,
                              batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []                        

                if random_rois:
                    inputs.extend([batch_rpn_rois])
                    if detection_targets:
                        inputs.extend([batch_rois])
                        # Keras requires that output and targets have the same number of dimensions
                        batch_mrcnn_class_ids = np.expand_dims(
                            batch_mrcnn_class_ids, -1)
                        outputs.extend(
                            [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])
                
                yield inputs, outputs
                #Sreturn inputs, outputs
                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            modellib.logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise

def stage1_data_generator(*args, **kwargs):
    wrapped_gen = matching_data_generator(*args, **kwargs)
    while True:
        inputs, outputs = next(wrapped_gen)
        yield inputs[:7], outputs
         
def validation_generator(dataset, config, shuffle=False, augmentation=None, random_rois=0,
                         batch_size=1, detection_targets=False, image_id=None, get_mask=True):
    # NOTICE: This is very IMPORTANT, read first before you do ANY modification.
    # This function is adapted from siamese mrcnn. If any code is problematic and you want to
    # modify any, please refer to their implementation and make sure you fully understand that.
    # Some of the codes are tough to comment on, sorry.
    b = 0  # batch item index
    image_index = -1
    if image_id is None:
        image_ids = np.copy(dataset.image_ids)
    else:
        image_ids = np.array([image_id])
    error_count = 0

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    backbone_shapes = modellib.compute_backbone_shapes(config, np.array([512, 512, 3]))
    anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
                                             config.RPN_ANCHOR_RATIOS,
                                             backbone_shapes,
                                             config.BACKBONE_STRIDES,
                                             config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinately.
    while True:
        # Increment index to pick next image. Shuffle if at the start of an epoch.
        image_index = (image_index + 1) % len(image_ids)
        if shuffle and image_index == 0:
            np.random.shuffle(image_ids)
        # Load image
        # Get GT bounding boxes and masks for image.
        image_id = image_ids[image_index]
        image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
            modellib.load_image_gt(dataset, config, image_id, augmentation=augmentation,
                                   use_mini_mask=config.USE_MINI_MASK)
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

        # Skip image if it contains no instance of any active class
        if not np.any(np.array(active_categories) > 0):
            continue
        # Randomly select a category as first target class
        category = np.random.choice(active_categories)
        # Dictionary maps target id with target image
        target_dict = dict()
        tbs_dict = dict()
        target_meta_dict = dict()
        if get_mask: tms_dict = dict()
        # Sample all target classes
        restart = 0  # restart count
        all_cats = [category]
        for i in range(config.WAYS - 1):  # Sample other target classes.
            while True:
                cat = np.random.choice(dataset.ACTIVE_CLASSES)
                if cat not in all_cats:
                    all_cats.append(cat)
                    break
        # Get RPN ground truths.
        active_class_bool = np.zeros_like(gt_class_ids, dtype=np.bool)
        for x in active_categories:
            active_class_bool += gt_class_ids == x

        rpn_gt_class = gt_class_ids[active_class_bool]
        rpn_gt_boxes = gt_boxes[active_class_bool, :]
        # RPN ground truths
        # Change: different from siamese mrcnn, we are real fg/bg detector at rpn
        rpn_match, rpn_bbox = modellib.build_rpn_targets(image.shape, anchors,
                                                         rpn_gt_class, rpn_gt_boxes, config)
        # Pick support images
        for i in range(config.WAYS):
            cat = all_cats[i]
            if config.SHOTS == 1:
                if not get_mask:
                    target, tb, target_meta, _ = get_one_target(cat, dataset, config,
                                            augmentation=augmentation, get_mask=get_mask)
                else:
                    target, tb, target_meta, mask, _ = get_one_target(cat, dataset, config,
                                                          augmentation=augmentation, get_mask=get_mask)
                if target is None:
                    print('skip rest target')
                    restart = 1
            else:
                if not get_mask:
                    target, tb, target_meta = get_k_targets(cat, dataset, config, config.SHOTS,
                                            augmentation=augmentation, get_mask=get_mask)
                else:
                    target, tb, target_meta, mask = get_k_targets(cat, dataset, config, config.SHOTS,
                                                         augmentation=augmentation, get_mask=get_mask)
                if (target is None or len(target) != config.SHOTS):
                    print('skip rest target')
                    restart = 1
            target_dict[cat] = target
            tbs_dict[cat] = tb
            target_meta_dict[cat] = target_meta
            if get_mask: tms_dict[cat] = mask
        if restart: 
            continue
        # Shuffle targets to avoid same order
        target_class_ids = list(target_dict.keys())
        random.shuffle(target_class_ids)
        # NOTE: Variable TARGETS is a list of lists if config.shots > 1
        targets = [target_dict[idx] for idx in target_class_ids]
        tbs = [tbs_dict[idx] for idx in target_class_ids]
        targets_meta = [target_meta_dict[idx] for idx in target_class_ids]
        if get_mask: tms = [tms_dict[idx] for idx in target_class_ids]
        # Initialize a list to contain gt of instances of different class
        match_class_ids_list = []
        gt_class_ids_list = []
        gt_boxes_list = []
        gt_masks_list = []
        for i in range(len(target_class_ids)):
            # idx is a bool list to choose match class' instance
            idx = gt_class_ids == target_class_ids[i]
            bool_class_ids = idx.astype('int8')[idx]
            num_instances = len(bool_class_ids)
            if num_instances == 0: continue  # No instance shares the same class as the ith shot
            match_class_id = i + 1  # ground truth for stage2 classifier is "the current order + 1" of the target!
            # choose the matching instances' gt
            gt_class_id = gt_class_ids[idx]  # The real gt class id is used for generating gt for rpn and roi
            gt_box = gt_boxes[idx, :]
            gt_mask = gt_masks[:, :, idx]
            for j in range(num_instances):
                match_class_ids_list.append(match_class_id)
            gt_class_ids_list.append(gt_class_id)
            gt_boxes_list.append(gt_box)
            gt_masks_list.append(gt_mask)
        match_class_ids = np.array(match_class_ids_list)
        if len(match_class_ids.shape) == 0:  # if there is only one gt instance, then there will be no shape, fix it
            match_class_ids = np.reshape(match_class_ids, (1,))
        gt_class_ids = np.concatenate(gt_class_ids_list, axis=0)
        gt_boxes = np.concatenate(gt_boxes_list, axis=0)
        gt_masks = np.concatenate(gt_masks_list, axis=-1)
        image_meta = image_meta[
                     :13]  # only the first 12 data are needed, the 13th element is included to support original mrcnn functions
        # Mask R-CNN Targets
        if random_rois:
            rpn_rois = generate_random_rois(
                image.shape, random_rois, rpn_gt_boxes, ratio=0.5)
            if detection_targets:
                rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask = \
                    modellib.build_detection_targets(
                        rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

        # Init batch arrays
        if b == 0:
            batch_targets_meta = np.zeros(
                (batch_size,) + image_meta.shape + (config.SUPPORT_NUMBER,), dtype=image_meta.dtype)
            batch_image_meta = np.zeros(
                (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
            batch_rpn_match = np.zeros(
                [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
            batch_rpn_bbox = np.zeros(
                [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
            batch_images = np.zeros(
                (batch_size,) + image.shape, dtype=np.float32)
            batch_gt_class_ids = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
            batch_gt_boxes = np.zeros(
                (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
            if config.SHOTS == 1:
                batch_targets = np.zeros(
                    (batch_size,) + target.shape + (config.WAYS,), dtype=np.float32)
                batch_tbs = np.zeros(
                    (batch_size, 4, config.WAYS), dtype=np.float32)
                if get_mask:
                    batch_masks = np.zeros(
                        (batch_size,) + mask.shape + (config.WAYS,), dtype=np.float32)
            else:
                batch_targets = np.zeros(
                    (batch_size,) + target[0].shape + (config.SUPPORT_NUMBER,), dtype=np.float32)
                batch_tbs = np.zeros(
                    (batch_size,) + (4, config.SUPPORT_NUMBER,), dtype=np.float32)
                if get_mask:
                    batch_masks = np.zeros(
                        (batch_size,) + mask[0].shape + (config.SUPPORT_NUMBER,), dtype=np.float32)
            if config.USE_MINI_MASK:
                batch_gt_masks = np.zeros((batch_size, config.MINI_MASK_SHAPE[0], config.MINI_MASK_SHAPE[1],
                                           config.MAX_GT_INSTANCES))
            else:
                batch_gt_masks = np.zeros(
                    (batch_size, image.shape[0], image.shape[1], config.MAX_GT_INSTANCES))
            if random_rois:  # DEPRECATED!
                batch_rpn_rois = np.zeros(
                    (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                if detection_targets:
                    batch_rois = np.zeros(
                        (batch_size,) + rois.shape, dtype=rois.dtype)
                    batch_mrcnn_class_ids = np.zeros(
                        (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                    batch_mrcnn_bbox = np.zeros(
                        (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                    batch_mrcnn_mask = np.zeros(
                        (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)
        # If more instances than fits in the array, sub-sample from them.
        if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
            gt_class_ids = gt_class_ids[ids]
            match_class_ids = match_class_ids[ids]
            gt_boxes = gt_boxes[ids]
            gt_masks = gt_masks[:, :, ids]

        # Add to batch
        batch_image_meta[b] = image_meta
        batch_rpn_match[b] = rpn_match[:, np.newaxis]
        batch_rpn_bbox[b] = rpn_bbox
        batch_images[b] = mold_image(image.astype(np.float32), config)

        for i in range(config.WAYS):
            if config.SHOTS == 1:
                batch_targets[b, :, :, :, i] = mold_image(targets[i].astype(np.float32), config)
                batch_tbs[b, :, i] = tbs[i]
                batch_targets_meta[b, :, i] = targets_meta[i]
                if get_mask:
                    batch_masks[b, :, :, i] = tms[i]
            else:
                for j, inst in enumerate(targets[i]):
                    batch_targets[b, :, :, :, i * config.SHOTS + j] = mold_image(
                        inst.astype(np.float32), config)
                for j, inst in enumerate(tbs[i]):
                    batch_tbs[b, :, i * config.SHOTS + j] = inst
                for j, inst in enumerate(targets_meta[i]):
                    batch_targets_meta[b, :, i * config.SHOTS + j] = inst
                if get_mask:
                    for j, inst in enumerate(tms[i]):
                        batch_masks[b, :, :, i * config.SHOTS + j] = inst
        batch_gt_class_ids[b, :match_class_ids.shape[0]] = match_class_ids
        # batch_target_class_ids[b, :target_class_ids.shape[0]] = target_class_ids # for debug
        batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
        batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

        if random_rois:
            batch_rpn_rois[b] = rpn_rois
            if detection_targets:
                batch_rois[b] = rois
                batch_mrcnn_class_ids[b] = mrcnn_class_ids
                batch_mrcnn_bbox[b] = mrcnn_bbox
                batch_mrcnn_mask[b] = mrcnn_mask

        b += 1
        # Batch full?
        if b >= batch_size:
            if not get_mask:
                inputs = [batch_images, batch_image_meta, batch_targets, batch_targets_meta, 
                      batch_tbs, batch_rpn_match, batch_rpn_bbox,
                      batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
            else:
                inputs = [batch_images, batch_image_meta, batch_targets, batch_targets_meta, 
                          batch_tbs, batch_masks, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]

            outputs = []
            if random_rois:
                inputs.extend([batch_rpn_rois])
                if detection_targets:
                    inputs.extend([batch_rois])
                    # Keras requires that output and targets have the same number of dimensions
                    batch_mrcnn_class_ids = np.expand_dims(
                        batch_mrcnn_class_ids, -1)
                    outputs.extend(
                        [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])
            yield inputs, outputs, target_class_ids
            b = 0
            
def stage1_validation_genenrator(dataset, config, shuffle=False, augmentation=None, random_rois=0, 
                         batch_size=1, detection_targets=False, image_id=None, get_mask=False, mode='inference'):
    assert mode in ['training', 'inference'], 'MODE should only be training / inference'
    wrapped_gen = validation_generator(dataset, config, shuffle=shuffle, augmentation=augmentation, random_rois=random_rois,
                         batch_size=batch_size, detection_targets=detection_targets, image_id=image_id, get_mask=get_mask)
    while True:
        inputs, outputs, target_class_ids = next(wrapped_gen)
        if mode == 'training': yield inputs[:7], outputs, target_class_ids
        else:
            model_inp = inputs[:5] + inputs[-3:]
            yield model_inp, outputs, target_class_ids

def generate_random_rois(image_shape, count, gt_boxes, ratio=0.5):
    """Generates ROI proposals similar to what a region proposal network
    would generate.
    image_shape: [Height, Width, Depth]
    count: Number of ROIs to generate
    gt_class_ids: [N] Integer ground truth class IDs
    gt_boxes: [N, (y1, x1, y2, x2)] Ground truth boxes in pixels.
    Returns: [count, (y1, x1, y2, x2)] ROI boxes in pixels.
    """
    # placeholder
    rois = np.zeros((count, 4), dtype=np.int32)

    # Generate random ROIs around GT boxes (90% of count)
    rois_per_box = int(ratio * count / gt_boxes.shape[0])
    for i in range(gt_boxes.shape[0]):
        gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes[i]
        h = gt_y2 - gt_y1
        w = gt_x2 - gt_x1
        # random boundaries
        r_y1 = max(gt_y1 - h, 0)
        r_y2 = min(gt_y2 + h, image_shape[0])
        r_x1 = max(gt_x1 - w, 0)
        r_x2 = min(gt_x2 + w, image_shape[1])

        # To avoid generating boxes with zero area, we generate double what
        # we need and filter out the extra. If we get fewer valid boxes
        # than we need, we loop and try again.
        while True:
            y1y2 = np.random.randint(r_y1, r_y2, (rois_per_box * 2, 2))
            x1x2 = np.random.randint(r_x1, r_x2, (rois_per_box * 2, 2))
            # Filter out zero area boxes
            threshold = 1
            y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                        threshold][:rois_per_box]
            x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                        threshold][:rois_per_box]
            if y1y2.shape[0] == rois_per_box and x1x2.shape[0] == rois_per_box:
                break

        # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
        # into x1, y1, x2, y2 order
        x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
        y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
        box_rois = np.hstack([y1, x1, y2, x2])
        rois[rois_per_box * i:rois_per_box * (i + 1)] = box_rois

    # Generate random ROIs anywhere in the image (10% of count)
    remaining_count = count - (rois_per_box * gt_boxes.shape[0])
    # To avoid generating boxes with zero area, we generate double what
    # we need and filter out the extra. If we get fewer valid boxes
    # than we need, we loop and try again.
    while True:
        y1y2 = np.random.randint(0, image_shape[0], (remaining_count * 2, 2))
        x1x2 = np.random.randint(0, image_shape[1], (remaining_count * 2, 2))
        # Filter out zero area boxes
        threshold = 1
        y1y2 = y1y2[np.abs(y1y2[:, 0] - y1y2[:, 1]) >=
                    threshold][:remaining_count]
        x1x2 = x1x2[np.abs(x1x2[:, 0] - x1x2[:, 1]) >=
                    threshold][:remaining_count]
        if y1y2.shape[0] == remaining_count and x1x2.shape[0] == remaining_count:
            break

    # Sort on axis 1 to ensure x1 <= x2 and y1 <= y2 and then reshape
    # into x1, y1, x2, y2 order
    x1, x2 = np.split(np.sort(x1x2, axis=1), 2, axis=1)
    y1, y2 = np.split(np.sort(y1y2, axis=1), 2, axis=1)
    global_rois = np.hstack([y1, x1, y2, x2])
    rois[-remaining_count:] = global_rois
    return rois

### Dataset Utils ###

class IndexedCocoDataset(coco.CocoDataset):
    def __init__(self):
        super(IndexedCocoDataset, self).__init__()
        self.active_classes = []

    def set_active_classes(self, active_classes):
        """active_classes could be an array of integers (class ids), or
           a filename (string) containing these class ids (one number per line)"""
        if type(active_classes) == str:
            with open(active_classes, 'r') as f:
                content = f.readlines()
            active_classes = [int(x.strip()) for x in content]
        self.active_classes = list(active_classes)

    def get_class_ids(self, active_classes, dataset_dir, subset, year):
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        class_ids = sorted(list(filter(lambda c: c in coco.getCatIds(), self.active_classes)))
        return class_ids

        self.class_ids_with_holes = class_ids

    def build_indices(self):

        self.image_category_index = IndexedCocoDataset._build_image_category_index(self)
        self.category_image_index = IndexedCocoDataset._build_category_image_index(self.image_category_index)

    def _build_image_category_index(dataset):

        image_category_index = []
        for im in range(len(dataset.image_info)):
            # List all classes in an image
            coco_class_ids = list(\
                                  np.unique(\
                                            [dataset.image_info[im]['annotations'][i]['category_id']\
                                             for i in range(len(dataset.image_info[im]['annotations']))]\
                                           )\
                                 )
            # Map 91 class IDs 81 to Mask-RCNN model type IDs
            class_ids = [dataset.map_source_class_id("coco.{}".format(coco_class_ids[k]))\
                         for k in range(len(coco_class_ids))]
            # Put list together
            image_category_index.append(class_ids)

        return image_category_index

    def _build_category_image_index(image_category_index):

        category_image_index = []
        # Loop through all 81 Mask-RCNN classes/categories
        for category in range(max(image_category_index)[0]+1):
            # Find all images corresponding to the selected class/category 
            images_per_category = np.where(\
                [any(image_category_index[i][j] == category\
                 for j in range(len(image_category_index[i])))\
                 for i in range(len(image_category_index))])[0]
            # Put list together
            category_image_index.append(images_per_category)

        return category_image_index

    
### Evaluation ###



class customCOCOeval(COCOeval):
    
    def summarize(self, class_index=None, verbose=1):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''
        def _summarize( ap=1, iouThr=None, areaRng='all', maxDets=100 ):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap==1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if not class_index is None:
                    s = s[:,:,class_index,aind,mind]
                else:
                    s = s[:,:,:,aind,mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                if not class_index is None:
                    s = s[:,class_index,aind,mind]
                else:
                    s = s[:,:,aind,mind]
            if len(s[s>-1])==0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s>-1])
            if verbose > 0:
                print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s
        def _summarizeDets():
            stats = np.zeros((12,))
            stats[0] = _summarize(1)
            stats[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[8] = _summarize(0, maxDets=self.params.maxDets[2])
            stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats
        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats
        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()

    def __str__(self, class_index=None):
        self.summarize(class_index)

def evaluate_coco(model, dataset, coco_object, eval_type="bbox", 
                  limit=0, image_ids=None, class_index=None, verbose=1, return_results=False):
    """Wrapper to keep original function name usable"""
        
    results = evaluate_dataset(model, dataset, coco_object, eval_type=eval_type, dataset_type='coco',
                     limit=limit, image_ids=image_ids, class_index=class_index, verbose=verbose, return_results=return_results)
    
    if return_results:
        return results


def evaluate_dataset(model, dataset, dataset_object, eval_type="bbox", dataset_type='coco',
                     limit=0, image_ids=None, class_index=None, verbose=1, random_detections=False,
                     return_results=False, random_rois=0):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    assert dataset_type in ['coco']
    # Pick COCO images from the dataset
    image_ids = image_ids or dataset.image_ids
    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]
    # Get corresponding COCO image IDs.
    dataset_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()

    results = []
    get_mask = model.config.MASK_LATE_FUSION
    for i, image_id in enumerate(image_ids):
        if i % 100 == 0 and verbose > 1:
            print("Processing image {}/{} ...".format(i, len(image_ids)))

        # Load GT data
        resized_image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
            modellib.load_image_gt(dataset, model.config,
                                   image_id, augmentation=False, use_mini_mask=model.config.USE_MINI_MASK)

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
        #print('categories: ', categories)
        #print('active_categories: ', active_categories)

        # Skiop image if it contains no instance of any active class
        if not np.any(np.array(active_categories) > 0):
            continue

        # Filter out crowd class
        active_class_bool = np.zeros_like(gt_class_ids, dtype=np.bool)
        for x in active_categories:
            active_class_bool += gt_class_ids == x
        # Generate random rois instead of rpn
        if random_rois:
            rpn_gt_boxes = gt_boxes[active_class_bool, :]
            rpn_rois = generate_random_rois(
                resized_image.shape, random_rois, rpn_gt_boxes, ratio=0.5)
            rpn_rois = np.expand_dims(rpn_rois, axis=0) # FIX: add batch dim for evaluation

        # END BOILERPLATE

        # Evaluate for every category individually
        for category in active_categories:
            # Load image
            image = dataset.load_image(image_id)
            # Draw random target
            try:
                targets = []
                tbs = []
                targets_meta = []
                if get_mask: tms = []
                all_cats = [category]
                for j in range(model.config.WAYS - 1):
                    while True:
                        rest_cats = np.random.choice(dataset.ACTIVE_CLASSES)
                        if rest_cats not in active_categories:
                            break
                    all_cats.append(rest_cats)
                # Shuffle ground truth labels
                random.shuffle(all_cats)
                for c in all_cats:
                    while True:
                        if not get_mask:
                            target, tb, meta, _ = get_one_target(c, dataset, model.config, get_mask=get_mask)
                        else:
                            target, tb, meta, mask, _ = get_one_target(c, dataset, model.config, get_mask=get_mask)
                        if target is not None and tb is not None:
                            break
                    targets.append(mold_image(target, model.config))
                    tbs.append(tb)
                    targets_meta.append(meta)
                    if get_mask: tms.append(mask)
            except ValueError:
                print('error fetching target of category', category)
                continue
            # Run detection
            t = time.time()
            model_inputs = [[targets], [tbs], [targets_meta], [image]]
            if get_mask: model_inputs.insert(3, [tms])
            if random_rois: # TODO: version error
                r = model.detect(*model_inputs, verbose=0, random_rois=rpn_rois)[0]
            else:
                r = model.detect(*model_inputs, verbose=0)[0]
            # Format detections
            r["class_ids"] = np.array([all_cats[np.asscalar(j-1)] for j in r["class_ids"]])
            t_prediction += (time.time() - t)

            # Convert results to COCO format
            # Cast masks to uint8 because COCO tools errors out on bool
            if dataset_type == 'coco':
                image_results = coco.build_coco_results(dataset, dataset_image_ids[i:i + 1],
                                                        r["rois"], r["class_ids"],
                                                        r["scores"],
                                                        r["masks"].astype(np.uint8))
            results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    dataset_results = dataset_object.loadRes(results)

    # allow evaluating bbox & segm:
    if not isinstance(eval_type, (list,)):
        eval_type = [eval_type]

    for current_eval_type in eval_type:
        # Evaluate
        cocoEval = customCOCOeval(dataset_object, dataset_results, current_eval_type)
        cocoEval.params.imgIds = dataset_image_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize(class_index=class_index, verbose=verbose)
        if verbose > 0:
            print("Prediction time: {}. Average {}/image".format(
                t_prediction, t_prediction / len(image_ids)))
            print("Total time: ", time.time() - t_start)

    if return_results:
        return cocoEval
    
    
### Visualization ###
def display_results(target, tb, category, image, boxes, masks, class_ids, 
                    gt_class_ids, gt_boxes,
                    ways, path, config,
                    scores=None, title="",  
                    figsize=(16, 16), ax=None,
                    show_mask=True, show_bbox=True,
                    colors=None, captions=None):
    """
    target: [support_number, (height, width, 3)]
    tb: [support_number, (y1, x1, y2, x2)]
    image: [(height, width, 3)]
    boxes: [num_instances, (y1, x1, y2, x2)]
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    gt_class_ids: [num_gt]
    gt_boxes: [num_gt, (y1, x1, y2, x2)]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        #assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
        assert boxes.shape[0] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    
    if not ax:
        from matplotlib.gridspec import GridSpec
        # Use GridSpec to show target smaller than image
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(ways, 3)
        ax = plt.subplot(gs[:, 1:])
        #target_ax = plt.subplot(gs[1, 0])
        target_height, target_width = 512, 512
        auto_show = True
        # show targets
        for i in range(ways):
            target_colors = visualize.random_colors(ways)
            color = target_colors[i]
            target_ax = plt.subplot(gs[i, 0])
            target_ax.set_ylim(target_height + 10, -10)
            target_ax.set_xlim(-10, target_width + 10)
            target_ax.axis('off')
            #target_image = target[:, :, :, i*config.SHOTS]
            target_image = target[i]
            #print(target.shape)
            #target = np.squeeze(target, axis=-1)
            # target_box
            y1, x1, y2, x2 = tb[i]
            p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                    alpha=0.7, linestyle="dashed",
                    edgecolor=color, facecolor='none')
            target_ax.add_patch(p)
            # target_class

            target_ax.imshow(target_image.astype(np.uint8))
            #show_title = 
            target_ax.set_title('class: ' + config.CLASS_ID[category[i]] + ' size: ' + str(y2 - y1) + ' * ' + str(x2 - x1))
            #target_ax.set_title('targets')
        

    # Generate random colors
    query_colors = colors or visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = query_colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        if show_bbox:
            p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1,
                                alpha=0.7, linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            #c = int(class_ids[i-1])
            #class_name = config.CLASS_ID[category[c]]
            score = scores[i] if scores is not None else None
            x = random.randint(x1, (x1 + x2) // 2)
            #caption = "{:.3f}".format(score) if score else 'no score'
            #caption = "{}: {:.3f}".format(class_name, score) if score and class_name else 'nothing'
            caption = "{}".format(str(class_ids[i])) 
            #caption = "{}".format(class_name) if class_name else 'nothing'
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")


    # show ground truth
    query_colors = colors or visualize.random_colors(len(gt_class_ids))
    for i in range(len(gt_class_ids)):
        color = query_colors[i]

        # Bounding box
        if not np.any(gt_boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = gt_boxes[i]
        y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
        if show_bbox:
            p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=4,
                                alpha=0.7, #linestyle="dashed",
                                edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            #c = int(class_ids[i-1])
            #class_name = config.CLASS_ID[category[c]]
            score = scores[i] if scores is not None else None
            x = random.randint(x1, (x1 + x2) // 2)
            #caption = "{:.3f}".format(score) if score else 'no score'
            #caption = "{}: {:.3f}".format(class_name, score) if score and class_name else 'nothing'
            caption = "{} : {}".format(config.CLASS_ID[gt_class_ids[i]], str(y2-y1)+' * '+str(x2-x1)) 
            #caption = "{}".format(class_name) if class_name else 'nothing'
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        """
        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = visualize.apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = visualize.find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = visualize.Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
            """
    ax.imshow(masked_image.astype(np.uint8))
    #target_ax.imshow(target.astype(np.uint8))
    count = 0
    name = str(count) + ".jpg"
    if auto_show:
        while os.path.exists(os.path.join(path, name)):
            count += 1
            name = str(count) + ".jpg"
        plt.savefig(os.path.join(path, name))
        print(os.path.join(path, name) + "has saved")
    plt.close()
    return


def display_grid(target_list, image_list, boxes_list, masks_list, class_ids_list,
                      scores_list=None, category_names_list=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None,
                      target_shift=10, fontsize=14):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """

    if type(target_list) == list:
        M = int(np.sqrt(len(target_list)))
        if len(target_list) - M**2 > 1e-3:
            M = M + 1
    else:
        M = 1
        target_list = [target_list]
        image_list = [image_list]
        boxes_list = [boxes_list]
        masks_list = [masks_list]
        class_ids_list = [class_ids_list]
        if scores_list is not None:
            scores_list = [scores_list]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        from matplotlib.gridspec import GridSpec
        # Use GridSpec to show target smaller than image
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(M, M, hspace=0.1, wspace=0.02, left=0, right=1, bottom=0, top=1)
        # auto_show = True REMOVE

    index = 0
    for m1 in range(M):
        for m2 in range(M):
            ax = plt.subplot(gs[m1, m2])

            if index >= len(target_list):
                continue

            target = target_list[index]
            image = image_list[index]
            boxes = boxes_list[index]
            masks = masks_list[index]
            class_ids = class_ids_list[index]
            scores = scores_list[index]

            # Number of instances
            N = boxes.shape[0]
            if not N:
                print("\n*** No instances to display *** \n")
            else:
                assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

            # Generate random colors
            colors = visualize.random_colors(N)

            # Show area outside image boundaries.
            height, width = image.shape[:2]
            ax.set_ylim(height, 0)
            ax.set_xlim(0, width)
            ax.axis('off')
            ax.set_title(title)
            
            masked_image = image.astype(np.uint32).copy()
            for i in range(N):
                color = colors[i]

                # Bounding box
                if not np.any(boxes[i]):
                    # Skip this instance. Has no bbox. Likely lost in image cropping.
                    continue
                y1, x1, y2, x2 = boxes[i]
                if show_bbox:
                    p = visualize.patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                        alpha=0.7, linestyle="dashed",
                                        edgecolor=color, facecolor='none')
                    ax.add_patch(p)

                # Label
                if not captions:
                    class_id = class_ids[i]
                    score = scores[i] if scores is not None else None
                    x = random.randint(x1, (x1 + x2) // 2)
                    caption = "{:.3f}".format(score) if score else 'no score'
                else:
                    caption = captions[i]
                ax.text(x1, y1 + 8, caption,
                        color='w', size=11, backgroundcolor="none")

                # Mask
                mask = masks[:, :, i]
                if show_mask:
                    masked_image = visualize.apply_mask(masked_image, mask, color)

                # Mask Polygon
                # Pad to ensure proper polygons for masks that touch image edges.
                padded_mask = np.zeros(
                    (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
                padded_mask[1:-1, 1:-1] = mask
                contours = visualize.find_contours(padded_mask, 0.5)
                for verts in contours:
                    # Subtract the padding and flip (y, x) to (x, y)
                    verts = np.fliplr(verts) - 1
                    p = visualize.Polygon(verts, facecolor="none", edgecolor=color)
                    ax.add_patch(p)
            ax.imshow(masked_image.astype(np.uint8))

            target_height, target_width = target.shape[:2]
            target_height = target_height // 2
            target_width = target_width // 2
            ax.imshow(target, extent=[target_shift, target_shift + target_width * 2, height - target_shift, height - target_shift - target_height * 2], zorder=9)
            rect = visualize.patches.Rectangle((target_shift, height - target_shift), target_width * 2, -target_height * 2, linewidth=5, edgecolor='white', facecolor='none', zorder=10)
            ax.add_patch(rect)
            if category_names_list is not None:
                plt.title(category_names_list[index], fontsize=fontsize)
            index = index + 1

    if auto_show:
        plt.show()
        
    return

def save_inference_with_logits(target_ids, tbs, categories, image_id, result,
                               gt_class_ids, gt_boxes, path, tms=None):

    """
    target_ids:    list of target_ids - support_number * [1]
    tbs:           list of tbs        - support_number * [(y1, x1, y2, x2)]
    categorise:    list of category   - support_number * [1]
    image_id:      a int point tiimage_id - [1]
    result: a dict contains {boxes, masks, score, class_ids, logits}
        rois:      num_instances * [(y1, x1, y2, x2)]
        class_ids: num_instances * [1]
        masks:     [height, width, num_instances](not need)
        scores:    num_instances * [1] - the maximum after softmax
        logits:    num_instances * [sup_nb] - before softmax
    
    gt_class_ids:  list of gt_box - num_gt * [1]
    gt_boxes:      list of gt_box - num_gt * [(y1, x1, y2, x2)]
    
    path:          point to json's path
    return: a .json file contain data:
    """
    # init dict
    image_data = {}
    with open(path, 'r') as json_file:
        try:
            image_list = json.load(json_file)
        except:
            image_list = []
    print('image_data: ', image_data)

    # init image_data
    image_data = {}

    # load targets
    image_data['target_ids'] = target_ids
    image_data['target_boxes'] = [i.tolist() for i in tbs]
    if tms:
        image_data['target_masks'] = [i.tolist() for i in tms]
    image_data['target_categories'] = [int(c) for c in categories]

    # load query 
    image_data['image_ids'] = [np.asscalar(image_id)]
    
    # load reuslts                
    image_data['class_ids'] = result['class_ids'].tolist()
    image_data['scores'] = result['scores'].tolist()
    image_data['boxes'] = result['rois'].tolist()
    image_data['logits'] = result['logits'].tolist()#[:len(image_data['scores'])]
    image_data['all_rois'] = result['all_rois'].tolist()
    image_data['full_scores'] = result['all_scores'].tolist()
    image_data['refined_rois'] = result['refined_rois'].tolist()

    # load ground truth bbox and class
    image_data['gt_class_ids'] = [int(gc) for gc in gt_class_ids]
    image_data['gt_boxes'] = [i.tolist() for i in gt_boxes]
    #print('image_data: ', image_data)

    image_list.append(image_data)

    with open(path, 'w', encoding='utf-8') as json_file:
        json_file.write(json.dumps(image_list, json_file, indent=4))
    
    print(path, ' json_file have save!')

