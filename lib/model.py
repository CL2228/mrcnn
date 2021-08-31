
import tensorflow as tf
import glob
import sys
import os
import re
import time
import numpy as np
import math

import keras
import keras.backend as K
import keras.layers as KL
import keras.initializers as KI
import keras.engine as KE
import keras.models as KM
import multiprocessing

from .mrcnn import utils
from .mrcnn import model as modellib
from .mrcnn import visualize
from lib import utils as my_utils
from lib.backbone import *

MASK_RCNN_MODEL_PATH = 'Mask_RCNN/'

if MASK_RCNN_MODEL_PATH not in sys.path:
    sys.path.append(MASK_RCNN_MODEL_PATH)

############################################################
# Two different versions of attention RPN.
############################################################

class RPNAttentionLayer(KE.Layer):
    def __init__(self, config, **kwargs):
        super(RPNAttentionLayer, self).__init__(**kwargs)
        self.config = config

    def call(self, inputs):
        """ Apply attention map on feature_map, attention map is attained by SG-One style.
        :param sup_emb: [1, 1, depth]
        :param feature_map: [h, w, depth]
        :return: attention_map: [h, w, 1]
        """
        sup_emb, feature_map = inputs
        # L2-norm
        sup_emb = tf.nn.l2_normalize(sup_emb, axis=-1)
        feature_map_rep = tf.nn.l2_normalize(feature_map, axis=-1)
        att = tf.reduce_sum(sup_emb * feature_map_rep, axis=-1, keepdims=True)
        return att

    def compute_output_shape(self, input_shape):
        return input_shape[-1][:-1] + (1,)

def build_attention_rpn_model(anchor_stride, anchors_per_location, depth, config):
    """
    :param anchor_stride:
    :param anchors_per_location:
    :param depth:
    :param config:
    :return:  logits, probs: [b,n,2]
                bbox: [b,n,4]
    """
    sup_emb = KL.Input(shape=[None, None, depth], name='arpn_input_support_feature_map') #[b,n,n,d]
    input_feature_map = KL.Input(shape=[None, None, depth], name='arpn_input_query_feature_map')
    sup_emb_gap = KL.Lambda(lambda x: tf.reduce_mean(
        tf.reduce_mean(x, axis=2, keepdims=True), axis=1, keepdims=True), name='arpn_sup_gap')(sup_emb) #[b,1,1,d]

    att_rpn = RPNAttentionLayer(config)
    attention = att_rpn([sup_emb_gap, input_feature_map]) #[b,h,w,1]
    shared = KL.Conv2D(256, (3, 3), padding='same', activation='relu', strides=anchor_stride,
                       name='arpn_conv_shared')(input_feature_map)   #[b,h,w,256]
    shared = KL.Multiply(name='arpn_attention')([shared, attention]) #[b,h,w,256]
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)  #[b,h,w,2*apl]

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)
    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    outputs = [rpn_class_logits, rpn_probs, rpn_bbox]
    return KM.Model([input_feature_map, sup_emb], outputs, name="arpn_model")
    
# Channel level attention is viewed as 1*1 kernel spatial correlation with query feature map.
# This idea is proposed in Few-Shot Object Detection with Attention-RPN and Multi-Relation Detector.
def build_channel_attention_rpn_model(anchor_stride, anchors_per_location, depth, config):
    sup_emb = KL.Input(shape=[None, None, depth], name='arpn_input_support_feature_map')
    input_feature_map = KL.Input(shape=[None, None, depth], name='arpn_input_query_feature_map')
    sup_emb_gap = KL.Lambda(lambda x: tf.reduce_mean(
        tf.reduce_mean(x, axis=2, keepdims=True), axis=1, keepdims=True), name='arpn_sup_gap')(sup_emb)
    shared = KL.Multiply(name='arpn_correlation')([sup_emb_gap, input_feature_map])
    shared = KL.Conv2D(256, (3, 3), padding='same', activation='relu', strides=anchor_stride,
                       name='arpn_conv_shared')(shared)
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    outputs = [rpn_class_logits, rpn_probs, rpn_bbox]
    return KM.Model([input_feature_map, sup_emb], outputs, name="arpn_model")
    
def build_guidance_rpn_model(anchor_stride, anchors_per_location, depth, config):
    sup_emb = KL.Input(shape=[None, None, depth], name='arpn_input_support_feature_map')
    input_feature_map = KL.Input(shape=[None, None, depth], name='arpn_input_query_feature_map')
    sup_emb_gap = KL.GlobalAveragePooling2D(name='arpn_sup_gap')(sup_emb)
    guidance = KL.Lambda(lambda x: tf.abs(x[0] - tf.expand_dims(tf.expand_dims(x[1], axis=1), axis=1)))([input_feature_map, sup_emb_gap])
    shared = KL.Concatenate(axis=-1, name='arpn_guidance')([input_feature_map, guidance])
    shared = KL.Conv2D(3 * depth // 2, (1, 1), activation='relu', name='arpn_conv')(shared)
    shared = KL.Conv2D(256, (3, 3), padding='same', activation='relu', strides=anchor_stride,
                       name='arpn_conv_shared')(shared)
    x = KL.Conv2D(2 * anchors_per_location, (1, 1), padding='valid',
                  activation='linear', name='rpn_class_raw')(shared)

    # Reshape to [batch, anchors, 2]
    rpn_class_logits = KL.Lambda(
        lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 2]))(x)

    # Softmax on last dimension of BG/FG.
    rpn_probs = KL.Activation(
        "softmax", name="rpn_class_xxx")(rpn_class_logits)

    # Bounding box refinement. [batch, H, W, anchors per location * depth]
    # where depth is [x, y, log(w), log(h)]
    x = KL.Conv2D(anchors_per_location * 4, (1, 1), padding="valid",
                  activation='linear', name='rpn_bbox_pred')(shared)

    # Reshape to [batch, anchors, 4]
    rpn_bbox = KL.Lambda(lambda t: tf.reshape(t, [tf.shape(t)[0], -1, 4]))(x)
    outputs = [rpn_class_logits, rpn_probs, rpn_bbox]
    return KM.Model([input_feature_map, sup_emb], outputs, name="arpn_model")


############################################################
# Classifier branch & Bbox detection branch
############################################################
def fpn_classifier_graph(rois, feature_maps, image_meta,
                         pool_size, num_classes, sup_featmaps,
                         config, train_bn=True, fc_layers_size=1024):
    """
    :param rois: [b, rois, 4]
    :param feature_maps: list of FPN outputs
    :param pool_size: 7
    :param num_classes: set to ways+1
    :param sup_featmaps: [b, rois, 7, 7, channels]
    :return:  mrcnn_class_logits, [b, rois, ways + 1], used for loss graphs.
               mrcnn_probs, [b, rois, ways+1], used for aggregating detections.
               mrcnn_bbox, [b, rois, 4, ways+1], box deltas, the last dim is for consistency
                   when using original mrcnn loss graphs
               mrcnn_support_logits, [b, rois, sup_nb], used for mask branch selecting guidance
               scores: [b, rois, sup_nb, 2], used for debug
    """
    # sup_featmaps [batch, sup_nb, 7, 7, FPN_FEATMAPS]
    # ROI Pooling
    shared = modellib.PyramidROIAlign([pool_size, pool_size],  # [7, 7]
                                 name="roi_align_classifier")([rois, image_meta] + feature_maps) #[b,n,7,7,c]
    x = KL.Lambda(lambda x: relation_module(*x), name='relation_concat')([shared, sup_featmaps])
    # x [batch, rois * sup_nb, 7, 7, FPN_FEATMAPS * 2]
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (3, 3), padding='valid', name='mrcnn_relation_conv1'))(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size - 2, pool_size - 2), padding="valid"),
                           name="mrcnn_relation_conv2")(x)  # x - [b, num_rois*sup_nb, 1, 1, FPN_FEATMAPS]
    x = KL.TimeDistributed(modellib.BatchNorm(), name='mrcnn_relation_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(fc_layers_size // 4, (1, 1)),
                           name="mrcnn_relation_fc1")(x)  # x-[b, num_rois*sup_nb, 1, 1, FPN_FEATMAPS // 4]
    x = KL.TimeDistributed(modellib.BatchNorm(), name='mrcnn_relation_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    # Change: detach the support embeddings and roi embeddings
    x = KL.Lambda(lambda x: K.squeeze(K.squeeze(x, 3), 2),
                       name="pool_squeeze")(x)  # [b, roi * sup_nb, ch]
    scores = KL.TimeDistributed(KL.Dense(2, name='mrcnn_relation_fc2', activation='linear'))(x)
    scores = KL.Lambda(lambda x: tf.stack(tf.split(x, config.SUPPORT_NUMBER, axis=1), axis=2))(scores) # [b, num_rois, sup_nb, 2]
    
    mrcnn_support_logits = KL.Lambda(lambda x: tf.squeeze(tf.split(x, 2, axis=-1)[0], axis=-1),
                                     name='mrcnn_relation_support_logits')(scores) # [b, num_rois, sup_nb]
    mrcnn_class_logits = KL.TimeDistributed(ScoreAggregator(config),
                                           name='mrcnn_relation_class_logits')(scores) # [b, num_rois, ways + 1]
    mrcnn_probs = KL.Softmax(axis=-1, name='mrcnn_relation_class_probs')(mrcnn_class_logits) # [b, num_rois, ways + 1]

    # BBox head
    # [batch, boxes, (dy, dx, log(dh), log(dw))]
    y = KL.TimeDistributed(KL.Conv2D(fc_layers_size, (pool_size, pool_size), padding='valid'),
                           name='mrcnn_bbox_fc1')(shared)
    y = KL.TimeDistributed(KL.Dense(4, activation='linear'),  # x - [b, num_rois, 4]
                           name='mrcnn_bbox_fc2')(y)
    # Reshape to [batch, boxes, (dy, dx, log(dh), log(dw))]
    s = K.int_shape(y)  # s=[b, num_rois, 4]
    y = KL.Reshape((s[1], 1, 4), name="mrcnn_bbox")(y)  # x - [b, num_rois, 1, 4]
    # Duplicate output for fg/bg detections
    mrcnn_bbox = Concat(axis=-2)([y for i in range(num_classes)])
    # mrcnn_bbox - [b, num_rois, ways+1, 4]
    return mrcnn_class_logits, mrcnn_probs, mrcnn_bbox, mrcnn_support_logits, scores

def relation_module(roi_feat, sup_feat):
    """ Concatenate rois' feature maps and support boxes' feature maps.
    :param roi_feat: [batch, rois, 7, 7, channels]
    :param sup_feat: [batch, sup_nb, 7, 7, channels]
    :return: [batch, rois * sup_nb, 7, 7, channels*2, sup_nb]
    """
    num_roi = K.int_shape(roi_feat)[1]
    sup_nb = K.int_shape(sup_feat)[1]
    sup_feat = tf.expand_dims(tf.transpose(sup_feat, [0, 2, 3, 4, 1]), axis=1)  # [b, 1, 7, 7, ch, sup_nb]
    sup_feat = tf.tile(sup_feat, [1, num_roi, 1, 1, 1, 1])  # [b, roi, 7, 7, ch, sup_nb]
    roi_feat = tf.tile(tf.expand_dims(roi_feat, axis=-1), [1, 1, 1, 1, 1, sup_nb])  # [b, roi, 7, 7, ch, sup_nb]
    concatted = tf.concat([roi_feat, sup_feat], axis=-2)
    concatted = tf.split(concatted, sup_nb, axis=-1)  # [b, roi, 7, 7, ch*2, 1]
    return tf.squeeze(tf.concat(concatted, axis=1), axis=-1)  # [b, roi * sup, 7, 7, ch*2, 1]

class ScoreAggregator(KE.Layer):
    """ Aggregate predicted scores from support-wise to way-wise.
    INPUT scores: [batch, sup_nb, 2]
    OUTPUT logits: [batch, 1 + ways],
    """
    def __init__(self, config, **kwargs):
        super(ScoreAggregator, self).__init__(**kwargs)
        self.config = config

    def call(self, scores):
        neg_scores, pos_scores = tf.split(scores, 2, axis=-1) # [b, sup_nb, 1]
        score_mask = tf.one_hot(K.argmax(pos_scores, axis=1), depth=self.config.SUPPORT_NUMBER, axis=1) # [b, sup_nb, 1]
        neg_prob = tf.reduce_sum(neg_scores * score_mask, axis=1) # [b, 1]
        scores_per_way = tf.split(pos_scores, self.config.WAYS, axis=1)
        scores_per_way = [tf.reduce_max(score, axis=1) for score in scores_per_way]
        scores_per_way.insert(0, neg_prob)
        scores = tf.concat(scores_per_way, axis=-1)
        return scores

    def compute_output_shape(self, input_shape):
        return (None, self.config.WAYS + 1)

def mrcnn_class_loss_graph(target_class_ids, mrcnn_class_logits):
    target_class_ids = tf.cast(target_class_ids, 'int64')
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=target_class_ids, logits=mrcnn_class_logits)
    loss = K.switch(tf.size(loss) > 0, K.mean(loss), tf.constant(0.0))
    return loss

############################################################
# Many different versions of mask head
############################################################
class FeatmapSelectionLayer(KE.Layer):
    """
    :param sup_featmaps: [batch, sup_nb, 14, 14, featmaps]
    :param sup_logits: [batch, rois, sup_nb]
    :return: sup_featmap: [batch, rois, 14, 14, featmaps]
    """
    def __init__(self, config, mode, **kwargs):
        super(FeatmapSelectionLayer, self).__init__(**kwargs)
        self.config = config
        self.mode = mode

    def call(self, inputs):
        sup_featmaps, sup_logits = inputs
        sup_featmap = utils.batch_slice([sup_featmaps, sup_logits],
                                        lambda x, y: self._batchwise_func(x, y, self.mode),
                                        self.config.IMAGES_PER_GPU)
        return sup_featmap

    def _batchwise_func(self, sup_featmaps, sup_logits, mode):
        """
        :param sup_featmaps: [sup_nb, 7, 7, featmaps]
        :param sup_logits: [rois, sup_nb]
        :return: sup_featmaps: [rois, 7, 7, featmaps]
        """
        sup_nb = tf.shape(sup_featmaps)[0]
        tiling = self.config.TRAIN_ROIS_PER_IMAGE if mode == 'training' else \
                 self.config.DETECTION_MAX_INSTANCES
        sup_featmaps = tf.tile(tf.expand_dims(sup_featmaps, axis=0),
                               [tiling, 1, 1, 1, 1]) # [rois, sup_nb, w, h, c] 
        selections = tf.cast(tf.argmax(sup_logits, axis=-1), tf.int32)
        selections = tf.one_hot(selections, sup_nb)
        sup_featmap = tf.boolean_mask(sup_featmaps, selections)
        return sup_featmap

    def compute_output_shape(self, input_shape):
        return (None, None,
                self.config.MASK_POOL_SIZE, self.config.MASK_POOL_SIZE, self.config.TOP_DOWN_PYRAMID_SIZE)

def coFCN_graph(rois, sup_featmaps):
    """
    :param rois: [batch, num_roi, 14, 14, FPN_FEATMAPS]
    :param sup_featmaps: [batch, num_roi, 14, 14, FPN_FEATMAPS]
    :return: [batch, num_roi*sup_nb, 14, 14, 1]
    """
    pool_size = K.int_shape(sup_featmaps)[2]
    # GAP, [batch, num_roi, 1, 1, FPN_FEATMAPS]
    sup_emb = tf.reduce_mean(tf.reduce_mean(sup_featmaps, axis=2, keepdims=True, name='cofcn_gap1'),
                             axis=3, keepdims=True, name='cofcn_gap2')
    sup_featmaps = tf.tile(sup_emb, [1, 1, pool_size, pool_size, 1])
    return tf.concat([rois, sup_featmaps], axis=-1)

def fpn_mask_graph_coFCN(rois, feature_maps, image_meta,
                         pool_size, num_classes,
                         sup_featmaps, support_logits, config, mode,
                         sup_masks=None, train_bn=True):
    """
    :param rois: [batch, rois, 4]
    :param feature_maps: P2-P6
    :param num_classes: WAYS + 1
    :param sup_featmaps: [batch, sub_nb, 14, 14, featmaps]
    :param support_logits: [batch, rois, sup_nb]
    :param(Optional) sup_masks: [batch, 56, 56, sup_nb]
    :return: single mask tiled WAYS+1 times
    """
    # Prepare supporting feature maps
    if config.MASK_LATE_FUSION:
        assert sup_masks is not None, "Support masks must be offered if enable late fusion strategy"
        sup_masks = KL.Lambda(lambda x: tf.expand_dims(tf.transpose(x, [0, 3, 1, 2]), axis=-1),
                              name='support_mask_reshape')(sup_masks)
        sup_masks = KL.TimeDistributed(KL.AveragePooling2D((4, 4), strides=4),
                                       name='support_mask_downsampling')(sup_masks) # equivalent to bi-interp
        sup_featmaps = KL.Multiply(name='late_fusion')([sup_featmaps, sup_masks])
    sup_featmaps = FeatmapSelectionLayer(config, mode, name='mrcnn_featmap_sel')([sup_featmaps, support_logits])

    x = modellib.PyramidROIAlign([pool_size, pool_size],
                                 name="roi_align_mask")([rois, image_meta] + feature_maps)
    x = KL.Lambda(lambda x: coFCN_graph(*x), name='coFCN_fusion')([x, sup_featmaps])
    x = KL.Lambda(KL.Conv2D(config.FPN_CLASSIF_FC_LAYERS_SIZE, (1, 1)), name='mrcnn_mask_feat_fusion')(x)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(1, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask_support")(x)
    x = Concat(axis=-1)([x for i in range(num_classes)])
    return x

def sgone_cosine_graph(rois, sup_featmaps):
    """
    :param rois: [batch, num_roi, 7, 7, FPN_FEATMAPS]
    :param sup_featmaps: [batch, num_roi, 7, 7, FPN_FEATMAPS]
    :return: [batch, num_roi*sup_nb, 7, 7, 1]
    """
    num_roi = K.int_shape(rois)[1]
    # GAP, [batch, num_roi, 1, 1, FPN_FEATMAPS]
    sup_emb = tf.reduce_mean(tf.reduce_mean(sup_featmaps, axis=2, keepdims=True, name='cofcn_gap1'),
                             axis=3, keepdims=True, name='cofcn_gap2')
    sup_emb = tf.nn.l2_normalize(sup_emb, axis=-1)
    rois = tf.nn.l2_normalize(rois, axis=-1)
    sim_maps = tf.reduce_sum(sup_emb * rois, axis=-1, keepdims=True)  # [batch, num_roi, 7, 7, 1]
    return sim_maps

def fpn_mask_graph_sgone(rois, feature_maps, image_meta,
                         pool_size, num_classes,
                         sup_featmaps, support_logits, config, mode,
                         sup_masks=None, train_bn=True):
    """
    :param rois: [batch, rois, 4]
    :param feature_maps: P2-P6
    :param num_classes: WAYS + 1
    :param sup_featmaps: [batch, sub_nb, 14, 14, featmaps]
    :param support_logits: [batch, rois, sup_nb]
    :param(Optional) sup_masks: [batch, 56, 56, sup_nb]
    :return: single mask tiled WAYS+1 times
    """
    if config.MASK_LATE_FUSION:
        assert sup_masks is not None, "Support masks must be offered if enable late fusion strategy"
        sup_masks = KL.Lambda(lambda x: tf.expand_dims(tf.transpose(x, [0, 3, 1, 2]), axis=-1),
                              name='support_mask_reshape')(sup_masks)
        sup_masks = KL.TimeDistributed(KL.AveragePooling2D((4, 4), strides=4),
                                       name='support_mask_downsampling')(sup_masks) # equivalent to bi-interp
        sup_featmaps = KL.Multiply(name='late_fusion')([sup_featmaps, sup_masks])
    # Select the corresponding feature map for each query
    sup_featmaps = FeatmapSelectionLayer(config, mode, name='mrcnn_featmap_sel')([sup_featmaps, support_logits])
    # sup_featmaps - [batch, rois, 14, 14, 128]
    # NOTE: this mask head is now incumbent
    x = modellib.PyramidROIAlign([pool_size, pool_size],
                                 name="roi_align_mask")([rois, image_meta] + feature_maps)
    # FIXME: change this accordingly
    sim_maps = KL.Lambda(lambda x: sgone_cosine_graph(*x), name='sgone_similarity')([x, sup_featmaps])
    # sim_maps - [batch, num_roi * sup_nb, 7, 7, 1]
    x = KL.Multiply(name='sgone_pixel_mul')([x, sim_maps])
    # x - [batch, num_roi * sup_nb, 7, 7, 128]
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv1")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn1')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv2")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn2')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv3")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn3')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2D(256, (3, 3), padding="same"),
                           name="mrcnn_mask_conv4")(x)
    x = KL.TimeDistributed(modellib.BatchNorm(),
                           name='mrcnn_mask_bn4')(x, training=train_bn)
    x = KL.Activation('relu')(x)
    x = KL.TimeDistributed(KL.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                           name="mrcnn_mask_deconv")(x)
    x = KL.TimeDistributed(KL.Conv2D(1, (1, 1), strides=1, activation="sigmoid"),
                           name="mrcnn_mask_support")(x)
    # FIXME: select mask at the beginning to save calculation
    x = Concat(axis=-1)([x for i in range(num_classes)])
    return x

class SplitLambda:
   def __init__(self, func, **kwargs):
       self.layer = KL.Lambda(func, **kwargs)
       
   def __call__(self, inputs):
       out = self.layer(inputs)
       if isinstance(out, list):
           return out
       else: return [out]
       
class Concat:
   def __init__(self, *args, **kwargs):
       self.layer = KL.Concatenate(*args, **kwargs)
       
   def __call__(self, inputs):
       if len(inputs) == 1: return inputs[0]
       return self.layer(inputs)
            
############################################################
# Matching Mask R-CNN
############################################################
class MatchingMaskRCNN(modellib.MaskRCNN):
    def __init__(self, mode, config, model_dir):
        assert mode in ['training', 'inference', 'debug']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        # self.set_log_dir()
        self.keras_model = self.build(mode=mode, config=config)
        
    def prepare_input(self, mode, config,
                      input_targets_tensor, input_supports_meta, input_supports_bbox):
        # Input targets and other supporting inputs share some specific functional layer
        reshaper = KL.Lambda(lambda x: tf.squeeze(x, axis=-1), name='target_reshaper')
        spliter = SplitLambda(lambda x: tf.split(x, config.SUPPORT_NUMBER, axis=-1),
                            name='targets_split')
        input_targets = spliter(input_targets_tensor)  # targets - s_n * [b, w, h, c, 1]
        input_targets = [reshaper(target) for target in input_targets]  # targets - s_n * [b, w, h, c]
        supports_meta = spliter(input_supports_meta)
        
        supports_bbox = spliter(input_supports_bbox)  # tbs - s_n * [b, 4, 1]
        supports_bbox = [reshaper(x) for x in supports_bbox]  # tbs - s_n * [b, 4]
        norm_bbox = KL.Lambda(lambda x: modellib.norm_boxes_graph(x, K.shape(input_targets_tensor)[1:3]),
                              name='norm_bbox')
        supports_bbox = [norm_bbox(x) for x in supports_bbox]  # tbs - s_n * [b, 4]
        tiler = KL.Lambda(lambda x: tf.expand_dims(x, axis=1), name='tile_bbox')
        supports_bbox = [tiler(x) for x in supports_bbox]  # tbs - s_n * [b, 1, 4]
        return input_targets, supports_meta, supports_bbox
        

    def build(self, mode, config):
        assert mode in ['training', 'inference', 'debug']
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_targets_tensor = KL.Input(
            shape=config.TARGET_SHAPE.tolist() + [config.SUPPORT_NUMBER, ], name='input_targets')  # targets - [b, w, h, c, s_n]
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")  # meta - [b, 13]
        input_supports_meta = KL.Input(shape=[config.IMAGE_META_SIZE, config.SUPPORT_NUMBER], 
                                       name='input_target_meta') # target_meta - [b, 13, sup_nb]
        input_supports_bbox = KL.Input(shape=[4, config.SUPPORT_NUMBER],
                                       name='input_support_bbox')  # tbs - [b, 4, s_n]
        mask_head_kwargs = {'train_bn': config.TRAIN_BN}
        if config.MASK_LATE_FUSION:
            input_supports_mask = KL.Input(shape=config.MINI_MASK_SHAPE + (config.SUPPORT_NUMBER,),
                                           name='input_support_mask')
            mask_head_kwargs['sup_masks'] = input_supports_mask
        input_targets, supports_meta, supports_bbox = self.prepare_input(mode, config, 
                                                         input_targets_tensor, input_supports_meta, input_supports_bbox)
                                                         
        if mode == "training":
            # RPN GT
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)  # rpn_match - [b, i_n, 1]
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)  # rpn_bbox - [b, i_n, 4]
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)  # gt_class - [b, i_n]
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)  # gt_boxes - [b, i_n, 4]
            gt_boxes = KL.Lambda(lambda x: modellib.norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)  # gt_boxes - [b, i_n, 4]
            if config.MODEL == 'mrcnn':
                if config.USE_MINI_MASK:
                    input_gt_masks = KL.Input(
                        shape=[config.MINI_MASK_SHAPE[0],
                               config.MINI_MASK_SHAPE[1], None],
                        name="input_gt_masks", dtype=bool)  # gt_mask - [b, 56, 56, i_n]
                else:
                    input_gt_masks = KL.Input(
                        shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                        name="input_gt_masks", dtype=bool)  # gt_mask - [b, 512, 512, i_n] (not use)
        else:
            input_anchors = KL.Input(shape=[None, 4], name="input_anchors")
            
        resnet = build_resnet_model(self.config)
        fpn = build_fpn_model(self.config.FPN_FEATUREMAPS)
        _, C2, C3, C4, C5 = resnet(input_image)
        P2, P3, P4, P5, P6 = fpn([C2, C3, C4, C5])

        # Prepare lists to receive the featmaps
        # 1. target featmap for cls branch, 7*7
        # 2. target featmap for mask branch, 14*14
        # 3. background featmap for cls branch, 7*7
        targets_cls_featmaps = []
        targets_mask_featmaps = []
        cls_pyramid = modellib.PyramidROIAlign([config.POOL_SIZE, config.POOL_SIZE],
                                               name='cls_pyramid')  # POOL_SIZE = 7

        mask_pyramid = modellib.PyramidROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE],
                                                name='mask_pyramid')  # MASK_POOL_SIZE = 14

        for img, bbox, meta in zip(input_targets, supports_bbox, supports_meta):
            _, TC2, TC3, TC4, TC5 = resnet(img)
            TP2, TP3, TP4, TP5, TP6 = fpn([TC2, TC3, TC4, TC5])
            cls_featmap = cls_pyramid([bbox, meta, TP2, TP3, TP4, TP5])        #[b,1,7,7,c]
            mask_featmap = mask_pyramid([bbox, meta, TP2, TP3, TP4, TP5])      #[b,1,14,14,c]
            #support pooled features
            targets_cls_featmaps.append(cls_featmap)    # list of [b,1,7,7,c]
            targets_mask_featmaps.append(mask_featmap)
        
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        # Anchors
        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)  # anchors - [a_n, 4]
            # Duplicate across the batch dimension because Keras requires it
            anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)  # anchors -[b, a_n, 4]
            # A hack to get around Keras's bad support for constants
            anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        else:
            anchors = input_anchors

        # RPN Models
        if config.ATTENTION == 'spatial':
            rpn = build_attention_rpn_model(config.RPN_ANCHOR_STRIDE,
                                            len(config.RPN_ANCHOR_RATIOS),
                                            self.config.FPN_FEATUREMAPS, config)
        elif config.ATTENTION == 'channel':
            rpn = build_channel_attention_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS),
                                                    self.config.FPN_FEATUREMAPS, config)
        else: 
            rpn = build_guidance_rpn_model(config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS),
                                           self.config.FPN_FEATUREMAPS, config)

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        squeezer = KL.Lambda(lambda x: tf.squeeze(x, axis=1), name='squeezer')
        for p in rpn_feature_maps:
            #p [b,h,w,c]
            level_logits, level_probs, level_bbox = [], [], []
            for sf in targets_cls_featmaps:
                #sf [b,1,7,7,c]
                sf = squeezer(sf)
                #sf [b,7,7,c]
                logits, probs, bbox = rpn([p, sf])
                level_logits.append(logits)    #list of [b,n,2]
                level_probs.append(probs)
                level_bbox.append(bbox)         #list of [b,n,4]
            level_logits = Concat(axis=1)(level_logits)  #[b, s*n, 2]
            level_probs = Concat(axis=1)(level_probs)
            level_bbox = Concat(axis=1)(level_bbox)     #[b, s*n, 4]
            layer_outputs.append([level_logits, level_probs, level_bbox])
            # layer_outputs = [P2~P6's [[rpn_class_logits], [rpn_probs], [rpn_bbox]]]
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [Concat(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]
        rpn_class_logits, rpn_class, rpn_bbox = outputs
        # [b, l*s*n, 2]  [b, l*s*n, 4]

        # Generate proposals
        # Proposals are [batch, N, (y1, x1, y2, x2)] in normalized coordinates
        # and zero padded.
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else config.POST_NMS_ROIS_INFERENCE  
        rpn_rois = modellib.ProposalLayer(
            proposal_count=proposal_count,  
            nms_threshold=config.RPN_NMS_THRESHOLD,  
            name="ROI",
            config=config)([rpn_class, rpn_bbox, anchors])
        # Concatenate target feature maps
        targets_cls_featmaps = Concat(axis=1)(targets_cls_featmaps)  #[b,s,7,7,c]?
        targets_mask_featmaps = Concat(axis=1)(targets_mask_featmaps)
        if mode == "training":
            if not config.USE_RPN_ROIS:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_TRAINING, 4],
                                      name="input_roi", dtype=tf.float32)
                # Normalize coordinates
                target_rois = KL.Lambda(lambda x: modellib.norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            else:
                target_rois = rpn_rois  # target_rois - [b, i_n, 4]

            # Generate detection targets
            # Subsamples proposals and generates target outputs for training
            # Note that proposal class IDs, gt_boxes, and gt_masks are zero
            # padded. Equally, returned rois and targets are zero padded.
            rois, target_class_ids, target_bbox, target_mask, paddings = \
                DetectionTargetLayer(config, name="proposal_targets")([
                    target_rois, input_gt_class_ids, gt_boxes, input_gt_masks])

            # Network Heads
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_support_logits, scores = \
                fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.WAYS + 1, targets_cls_featmaps,
                                     config, train_bn=config.TRAIN_BN, fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
            if config.MODEL == 'mrcnn':
                mrcnn_mask = fpn_mask_graph_sgone(rois, mrcnn_feature_maps, input_image_meta,
                                                  config.MASK_POOL_SIZE, config.WAYS + 1,
                                                  targets_mask_featmaps, mrcnn_support_logits,
                                                  config, mode, **mask_head_kwargs)

            output_rois = KL.Lambda(lambda x: x * 1, name="output_rois")(rois)
            # Losses
            rpn_class_loss = KL.Lambda(lambda x: modellib.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: modellib.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            class_loss = KL.Lambda(lambda x: mrcnn_class_loss_graph(*x), name='mrcnn_class_loss') \
                ([target_class_ids, mrcnn_class_logits])
            bbox_loss = KL.Lambda(lambda x: modellib.mrcnn_bbox_loss_graph(*x), name="mrcnn_bbox_loss")(
                [target_bbox, target_class_ids, mrcnn_bbox])

            output_detections = TrainDetectionLayer(config, name='train_detection')([
                rois, mrcnn_class, mrcnn_bbox, input_image_meta])
            if config.MODEL == 'mrcnn':
                mask_loss = KL.Lambda(lambda x: modellib.mrcnn_mask_loss_graph(*x), name="mrcnn_mask_loss")(
                    [target_mask, target_class_ids, mrcnn_mask])

            inputs = [input_image, input_image_meta, input_targets_tensor, input_supports_meta, input_supports_bbox,
                      input_rpn_match, input_rpn_bbox, input_gt_class_ids, input_gt_boxes, input_gt_masks]
            if config.MASK_LATE_FUSION:
                inputs.insert(5, input_supports_mask)
            if not config.USE_RPN_ROIS:
                inputs.append(input_rois)
            outputs = [rpn_class_logits, rpn_class, rpn_bbox,
                       mrcnn_class_logits, mrcnn_class, mrcnn_bbox, 
                       rpn_rois, output_rois, target_class_ids, output_detections,
                       rpn_class_loss, rpn_bbox_loss, class_loss, bbox_loss]
            if config.MODEL == 'mrcnn':
                outputs.insert(6, mrcnn_mask)
                outputs.append(mask_loss)
            model = KM.Model(inputs, outputs, name='mask_rcnn')

        else:
            if not config.USE_RPN_ROIS_INFERENCE:
                # Ignore predicted ROIs and use ROIs provided as an input.
                input_rois = KL.Input(shape=[config.POST_NMS_ROIS_INFERENCE, 4],
                                      name="input_roi", dtype=tf.float32)
                # Normalize coordinates
                rpn_rois = KL.Lambda(lambda x: modellib.norm_boxes_graph(
                    x, K.shape(input_image)[1:3]))(input_rois)
            elif mode == 'debug':
                clipped_rpn_rois = ROIDetectionLayer('proposal', config, name='rpn_clip')([rpn_rois, input_image_meta])
            mrcnn_class_logits, mrcnn_class, mrcnn_bbox, mrcnn_support_logits, scores = \
                fpn_classifier_graph(rpn_rois, mrcnn_feature_maps, input_image_meta,
                                     config.POOL_SIZE, config.WAYS + 1, targets_cls_featmaps,
                                     config, train_bn=config.TRAIN_BN,
                                     fc_layers_size=config.FPN_CLASSIF_FC_LAYERS_SIZE)
            det_with_sup = CustomDetectionLayer(config)(
                    [rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta, mrcnn_support_logits])
            detections, support_logits = KL.Lambda(lambda x: tf.split(x, [-1, config.SUPPORT_NUMBER], axis=-1),
                                                       name='detection_support_split')(det_with_sup)
            detection_boxes = KL.Lambda(lambda x: x[..., :4])(detections)
            if config.MODEL == 'mrcnn':
                mrcnn_mask = fpn_mask_graph_sgone(detection_boxes, mrcnn_feature_maps, input_image_meta,
                                              config.MASK_POOL_SIZE, config.WAYS + 1,
                                              targets_mask_featmaps, support_logits,
                                              config, mode, **mask_head_kwargs)
            if mode == 'debug':
                # Set config.TRAIN_ROIS_PER_IMAGE as POST_NMS_INFERENCE to reuse the train det layer.
                train_rois = config.TRAIN_ROIS_PER_IMAGE
                config.TRAIN_ROIS_PER_IMAGE = config.POST_NMS_ROIS_INFERENCE
                rois_after_refine = TrainDetectionLayer(config=config, name='inference_detection_layer')([
                                                        rpn_rois, mrcnn_class, mrcnn_bbox, input_image_meta])
            inputs = [input_image, input_image_meta, input_targets_tensor, input_supports_meta,
                      input_supports_bbox, input_anchors]
            if config.MASK_LATE_FUSION:
                inputs.insert(5, input_supports_mask)
            if not config.USE_RPN_ROIS_INFERENCE:
                inputs.append(input_rois)
            outputs = [detections, mrcnn_class_logits, mrcnn_class, mrcnn_bbox,
                       clipped_rpn_rois, rpn_class, rpn_bbox]
            if config.MODEL == 'mrcnn':
                outputs.insert(4, mrcnn_mask)
            if mode == 'debug':
                outputs.append(rois_after_refine)
                outputs.append(scores)
            model = KM.Model(inputs, outputs, name='mask_rcnn')

        if config.GPU_COUNT > 1:
            from .mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)
        return model

    def compile(self, learning_rate, momentum):
        """Gets the model ready for training. Adds losses, regularization, and
        metrics. Then calls the Keras compile() function.
        """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        # Add Losses
        # First, clear previously set losses to avoid duplication
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss", "rpn_bbox_loss",
            "mrcnn_class_loss", "mrcnn_bbox_loss", "mrcnn_mask_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=0):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            modellib.log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                if verbose > 0:
                    print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.
        model_path: If None, or a format different from what this code uses
            then set a new log directory and start epochs from 0. Otherwise,
            extract the log directory and the epoch counter from the file
            name.
        """
        # Set date and epoch counter as if starting a new model
        # load the time like "2019_05_10_18_32_40"
        time_now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.epoch = 0
        self.log_dir = os.path.join(self.model_dir,
                                    "matching_{}_{}_{}".format(self.config.MODEL.lower(),
                                                               self.config.NAME.lower(),
                                                               self.config.EXPERIMENT.lower()))
        # time_now)

        # Create log_dir if not exists
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "matching_mrcnn_*epoch*.h5")
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")

    def train(self, train_dataset, val_dataset, learning_rate, epochs, stage1, stage2, augmentation=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting which layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heaads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train Resnet stage 3 and up
              4+: Train Resnet stage 4 and up
              5+: Train Resnet stage 5 and up
              stage1: Resnet + fpn + rpn
              stage2: cls + mask heads
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gausssian blur with a random sigma in range 0 to 5.
                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        """
        assert self.mode == "training", "Create model in training mode."

        # Pre-defined layer regular expressions
        stage1_regex = {
            "n/a": "",
            # all layers but the backbone
            "rpn": r"(arpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "2+": r"(res2.*)|(bn2.*)|(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            # All layers
            "all": r"(res.*)|(bn.*)|(fpn\_.*)|(arpn\_.*)"
        }
        stage2_regex = {
            "n/a": "",
            "cls": r"(mrcnn\_relation.*)|(mrcnn\_bbox.*)",
            "mask": r"(mrcnn\_mask.*)",
            "all": r"(mrcnn\_.*)"
        }
        layers = ""
        if stage1 in stage1_regex.keys():
            layers += stage1_regex[stage1]
        if stage2 in stage2_regex.keys():
            layers += stage2_regex[stage2]
        rois = 0 if self.config.USE_RPN_ROIS else self.config.POST_NMS_ROIS_TRAINING
        train_generator = my_utils.matching_data_generator(train_dataset, self.config, shuffle=True,
                                                           augmentation=augmentation,
                                                           batch_size=self.config.BATCH_SIZE,
                                                           random_rois=rois,
                                                           detection_targets=False,
                                                           get_mask=self.config.MASK_LATE_FUSION
                                                           )
        val_generator = my_utils.matching_data_generator(val_dataset, self.config, shuffle=True,
                                                         batch_size=self.config.BATCH_SIZE,
                                                         random_rois=rois,
                                                         detection_targets=False,
                                                         get_mask=self.config.MASK_LATE_FUSION
                                                         )

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Train
        modellib.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        modellib.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)

    def mold_inputs(self, images, branch):
        """Takes a list of images and modifies them to the format expected
        as an input to the neural network.
        images: List of image matrices [height,width,depth]. Images can have
            different sizes.
        branch: 'segm' or 'cond', 'cond' branch only returns molded images
        Returns 3 Numpy matrices:
        molded_images: [N, h, w, 3]. Images resized and normalized.
        image_metas: [N, length of meta data]. Details about each image.
        windows: [N, (y1, x1, y2, x2)]. The portion of the image that has the
            original image (padding excluded).
        """
        assert branch in ['cond', 'segm']
        molded_images = []
        image_metas = []
        windows = []
        if branch == 'segm':
            for image in images:
                # Resize image
                molded_image, window, scale, padding, crop = utils.resize_image(
                    image,
                    min_dim=self.config.IMAGE_MIN_DIM,
                    min_scale=self.config.IMAGE_MIN_SCALE,
                    max_dim=self.config.IMAGE_MAX_DIM,
                    mode=self.config.IMAGE_RESIZE_MODE)
                molded_image = my_utils.mold_image(molded_image, self.config)
                # Build image_meta
                image_meta = modellib.compose_image_meta(
                    0, image.shape, molded_image.shape, window, scale,
                    np.zeros([self.config.NUM_CLASSES], dtype=np.int32))
                # Append
                molded_images.append(molded_image)
                windows.append(window)
                image_metas.append(image_meta)
        else:
            for image in images:
                molded_images.append(image)
            molded_images = np.stack(molded_images, axis=-1)
            return molded_images
        # Pack into arrays
        molded_images = np.stack(molded_images)
        image_metas = np.stack(image_metas)
        windows = np.stack(windows)
        return molded_images, image_metas, windows

    def detect(self, targets, targets_meta, targets_bbox, images, verbose=0, random_detections=False,
               random_rois=None, targets_mask=None, eval_mask=True):
        """Runs the detection pipeline.
        images: List of images, potentially of different sizes.
        targets: List of ways of targets [[...(sup_nb)][]...(batch)], each is molded by
            subtracting pixel mean and resize to (512, 512, 3).
        targets_bbox: List of ways of target bbox [[...(sup_nb)][]...(batch)], each is [4].
        targets_mask: List of ways of target mask [[...(sup_nb)][]...(batch)], each is [56, 56] if use mini-mask.
        random_detections: True then returns optimal model outputs expected.
        random_rois: True if use randomly generated rois instead of RPN.
        ========================================================================================================
        Returns a list of dicts, one dict per image. The dict contains:
        rois: [N, (y1, x1, y2, x2)] detection bounding boxes.
        class_ids: [N] int class IDs.
        scores: [N] float probability scores for the class IDs.
        masks: [H, W, N] instance binary masks.
        If the mode is in debug mode, the outputs are consistent with inference_json.py, see that file
        for details.
        """
        assert self.mode in ["inference", "debug"], "Create model in inference or debug mode."
        assert len(images) == self.config.BATCH_SIZE, "len(images) must be equal to BATCH_SIZE"
        assert (targets_mask and self.config.MASK_LATE_FUSION) or \
               (not targets_mask and not self.config.MASK_LATE_FUSION), "Please check if you are using late fusion"
        if not self.config.USE_RPN_ROIS_INFERENCE:
            assert random_rois is not None, "You have to create random rois as inputs to the network."
        if verbose:
            modellib.log("Processing {} support-query sets".format(len(images)))
            for image in images:
                modellib.log("image", image)

        # Mold inputs to format expected by the neural network
        # CHANGE: Removed molding of target -> detect expects molded target
        molded_images, image_metas, windows = self.mold_inputs(images, 'segm')
        # molded_targets, target_metas, target_windows = self.mold_inputs(targets)
        molded_targets = [self.mold_inputs(target, 'cond') for target in targets]
        molded_targets = np.stack(molded_targets, axis=0)
        targets_meta = [np.transpose(np.array(x), [1, 0]) for x in targets_meta]
        targets_meta = np.stack(targets_meta, axis=0)
        targets_bbox = [np.transpose(np.array(x), [1, 0]) for x in targets_bbox]
        targets_bbox = np.stack(targets_bbox, axis=0)
        if targets_mask:
            targets_mask = [np.stack(x, axis=-1) for x in targets_mask]
            targets_mask = np.stack(targets_mask, axis=0)
        # Validate image sizes
        # All images in a batch MUST be of the same size
        image_shape = molded_images[0].shape
        image_metas = image_metas[:, :13]
        for g in molded_images[1:]:
            assert g.shape == image_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."
        # CHANGE: add size assertion for target
        target_shape = molded_targets[0].shape
        for g in molded_images[1:]:
            assert g.shape == target_shape, \
                "After resizing, all images must have the same size. Check IMAGE_RESIZE_MODE and image sizes."
        # Anchors
        anchors = self.get_anchors(image_shape)
        # Duplicate across the batch dimension because Keras requires it
        anchors = np.broadcast_to(anchors, (self.config.BATCH_SIZE,) + anchors.shape)

        if verbose:
            modellib.log("molded_images", molded_images)
            #             modellib.log("image_metas", image_metas)
            # CHANGE: add targets to log
            modellib.log("molded_targets", molded_targets)
            #             modellib.log("target_metas", target_metas)
            modellib.log("anchors", anchors)
        # Change: use image metas as the input fake meta, as image area of query and supports are the same in our implementation

        inputs = [molded_images, image_metas, molded_targets, targets_meta, targets_bbox, anchors]
        if self.config.MASK_LATE_FUSION:
            inputs.insert(5, targets_mask)
        if not self.config.USE_RPN_ROIS_INFERENCE:
            inputs.append(random_rois)
        if self.mode == 'inference':
            detections, _, _, _, mrcnn_mask, _, _, _ = \
                self.keras_model.predict(inputs, verbose=0)
        else:
            detections, mrcnn_class_logits, _, _, mrcnn_mask, rpn_rois, _, _, refined_rois, sup_scores = \
                self.keras_model.predict(inputs, verbose=0)
            zeros_mask = np.zeros_like(mrcnn_mask)

        if random_detections:
            # Randomly shift the detected boxes
            window_limits = utils.norm_boxes(windows, (molded_images[0].shape[:2]))[0]
            y_shifts = np.random.uniform(-detections[0, :, 0] + window_limits[0],
                                         window_limits[2] - detections[0, :, 2])
            x_shifts = np.random.uniform(-detections[0, :, 1] + window_limits[1],
                                         window_limits[3] - detections[0, :, 3])
            zeros = np.zeros(detections.shape[1])
            shifts = np.stack([y_shifts, x_shifts, y_shifts, x_shifts, zeros, zeros], axis=-1)[np.newaxis]
            detections = detections + shifts

            # Randomly permute confidence scores

            non_zero_confidences = np.where(detections[0, :, -1])[0]
            random_perm = np.random.permutation(non_zero_confidences)
            permuted_confidences = np.concatenate([detections[0, :, -1][:len(non_zero_confidences)][random_perm],
                                                   np.zeros(detections.shape[1] - len(non_zero_confidences))])
            detections = np.concatenate(
                [detections[:, :, :-1], permuted_confidences.reshape(1, detections.shape[1], 1)], axis=-1)

            # Keep the sorted order of confidence scores
            detections = detections[:, np.argsort(-detections[0, :, -1]), :]
        # Process detections
        results = []
        for i, image in enumerate(images):
            pred_mask = mrcnn_mask[i] if eval_mask else np.zeros_like(mrcnn_mask[i])
            if self.mode == 'inference':
                final_rois, final_class_ids, final_scores, final_masks = \
                    self.unmold_detections(detections[i], pred_mask,
                                           image.shape, molded_images[i].shape,
                                           windows[i])
                one_result = {
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                    "masks": final_masks,
                }
            elif self.mode == 'debug':
                final_rois, final_class_ids, final_scores, final_masks = \
                    self.unmold_detections(detections[i], pred_mask,
                                           image.shape, molded_images[i].shape,
                                           windows[i])
                num_proposals = refined_rois.shape[1]
                prop_mask = np.zeros((num_proposals,) + pred_mask.shape[1:])
                denormed_refined_rois, _, _, _ = \
                    self.unmold_detections(refined_rois[i], prop_mask,
                                           image.shape, molded_images[i].shape,
                                           windows[i])
                rpn_detection = np.concatenate([rpn_rois[i], np.sum(np.ones_like(rpn_rois[i]), keepdims=True, axis=-1)], axis=-1)            
                denormed_rpn_rois, _, _, _ = \
                    self.unmold_detections(rpn_detection, prop_mask, 
                                           image.shape, molded_images[i].shape,
                                           windows[i])  
                one_result = {
                    "rois": final_rois,
                    "class_ids": final_class_ids,
                    "scores": final_scores,
                    "masks": final_masks,
                    "logits": mrcnn_class_logits[i],
                    "all_rois": denormed_rpn_rois,
                    "all_scores": sup_scores[i],
                    "refined_rois": denormed_refined_rois
                }

            results.append(one_result)
        return results

    def get_imagenet_weights(self, pretraining='imagenet-1k'):
        """Selects ImageNet trained weights.
        Returns path to weights file.
        """
        assert pretraining in ['imagenet-1k', 'imagenet-771', 'imagenet-687']

        checkpoint_dir = self.config.CHECKPOINT_DIR

        if pretraining == 'imagenet-1k':
            weights_path = os.path.join(checkpoint_dir, 'imagenet_resnet_1k.h5')
        elif pretraining == 'imagenet-771':
            weights_path = os.path.join(checkpoint_dir, 'imagenet_resnet_771.h5')
        elif pretraining == 'imagenet-687':
            weights_path = os.path.join(checkpoint_dir, 'imagenet_resnet_687.h5')
        return weights_path

    def load_imagenet_weights(self, pretraining='imagenet-1k', weights_path=None):
        print('initializing from imagenet weights ...')
        if not weights_path:
            weights_path = self.get_imagenet_weights(pretraining=pretraining)
        resnet = self.keras_model.get_layer('resnet_model') if \
            not hasattr(self.keras_model, 'inner_model') else \
            self.keras_model.inner_model.get_layer('resnet_model')
        resnet.load_weights(weights_path, by_name=True)
        self.set_log_dir()

    def load_rpn_weights(self, weights_path=None):
        print('initializing from pretrained-rpn weights ...')
        if not weights_path:
            weights_path = os.path.join(self.config.CHECKPOINT_DIR, 'pretrained_frcnn.h5')
        # Pick excluding layers
        exclude = []
        for i in range(1, 3):
            exclude.append('mrcnn_class_conv' + str(i))
            exclude.append('mrcnn_class_bn' + str(i))
        for i in range(1, 5):
            exclude.append('mrcnn_mask_conv' + str(i))
            exclude.append('mrcnn_mask_bn' + str(i))
        exclude.append('mrcnn_mask_deconv')
        self.load_weights(weights_path, by_name=True, exclude=exclude)
        # Load other inner models (ResNet and FPN)
        resnet = self.keras_model.get_layer('resnet_model') if \
            not hasattr(self.keras_model, 'inner_model') else \
            self.keras_model.inner_model.get_layer('resnet_model')
        resnet.load_weights(weights_path, by_name=True)
        fpn = self.keras_model.get_layer('fpn_model') if \
            not hasattr(self.keras_model, 'inner_model') else \
            self.keras_model.inner_model.get_layer('fpn_model')
        fpn.load_weights(weights_path, by_name=True)
        self.set_log_dir()

    def load_frcnn_weights(self, weights_path=None):
        print('initializing from pretrained faster rcnn weights ...')
        if not weights_path:
            weights_path = os.path.join(self.config.CHECKPOINT_DIR, 'pretrained_frcnn.h5')
        self.load_weights(weights_path, by_name=True)
        if hasattr(self.keras_model, 'inner_model'):
            self.keras_model.get_layer('mask_rcnn').load_weights(weights_path, by_name=True)
        resnet = self.keras_model.get_layer('resnet_model') if \
            not hasattr(self.keras_model, 'inner_model') else \
            self.keras_model.inner_model.get_layer('resnet_model')
        resnet.load_weights(weights_path, by_name=True)
        fpn = self.keras_model.get_layer('fpn_model') if \
            not hasattr(self.keras_model, 'inner_model') else \
            self.keras_model.inner_model.get_layer('fpn_model')
        fpn.load_weights(weights_path, by_name=True)
        self.set_log_dir()

    def load_checkpoint(self, weights_path, training_schedule=None, verbose=1):
        if verbose > 0:
            print('loading', weights_path, '...')

        stage1_regex = {
            "n/a": "",
            # all layers but the backbone
            "rpn": r"(arpn\_.*)|(fpn\_.*)",
            # From a specific Resnet stage and up
            "2+": r"(res2.*)|(bn2.*)|(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            # All layers
            "all": r"(res.*)|(bn.*)|(fpn\_.*)|(arpn\_.*)"
        }
        stage2_regex = {
            "n/a": "",
            "cls": r"(mrcnn\_relation.*)|(mrcnn\_bbox.*)",
            "mask": r"(mrcnn\_mask.*)",
            "all": r"(mrcnn\_.*)"
        }
        # set layers trainable for resnet weight loading
        epoch_index = int(weights_path[-7:-3])
        if verbose > 0:
            print('starting from epoch {}'.format(epoch_index))
        if training_schedule is not None:
            # get correct schedule period
            schedule_index = min([key for key in training_schedule.keys() if epoch_index <= key])
            layers = stage1_regex[training_schedule[schedule_index]["stage1"]] \
                     + stage2_regex[training_schedule[schedule_index]["stage2"]]
            self.set_trainable(layers)
        else:
            self.set_trainable(".*")
        # load weights, dealing with multi GPU and single GPU cross loading
        # FIXME: probably single GPU is not able to load multi-GPU weights
        try:
            if weights_path:
                self.load_weights(weights_path, by_name=True)
        except:
            if weights_path:
                self.keras_model.inner_model.load_weights(weights_path, by_name=True)
        self.epoch = epoch_index

    def get_latest_checkpoint(self):
        os.path.exists(os.path.join(self.log_dir, "matching_mrcnn_0001.h5"))
        list_of_files = glob.glob(os.path.join(self.log_dir, '*.h5'))  # * means all if need specific format then *.csv
        # Add condition for not finding any training files.
        if len(list_of_files) == 0: return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file

    def load_latest_checkpoint(self, training_schedule=None):
        weights_path = self.get_latest_checkpoint()
        self.load_checkpoint(weights_path, training_schedule=training_schedule)

    def apply_mask_to_image(self, images, masks, mode='tensor'):
        """Apply mask to images in RGB channels
        :param image: list - [[i1,i2,...,in],[...],...,[...]]
                      tensor - [[h, w, c, ways], [], []]
        :param mask: same as above
        :param mode: tensor or list
        :return: images applied with mask
        """
        assert mode in ['tensor', 'list']
        results = []
        if mode == 'tensor':
            for img, msk in zip(images, masks):
                if len(np.shape(msk)) == 3:  # channel dimension is deprecated
                    msk = np.expand_dims(msk, axis=2)
                results.append(img * msk)
        else:
            for img_stack, msk_stack in zip(images, masks):
                one_result = []
                for img, msk in zip(img_stack, msk_stack):
                    if len(np.shape(msk)) == 2:  # channel dimension is deprecated
                        msk = np.expand_dims(msk, axis=2)
                    one_result.append(img * msk)
                results.append(one_result)
        return results
        
class Stage1Network(MatchingMaskRCNN):
    def build(self, mode, config):
        assert mode in ['training', 'inference']
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2 ** 6 != int(h / 2 ** 6) or w / 2 ** 6 != int(w / 2 ** 6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        input_targets_tensor = KL.Input(
            shape=config.TARGET_SHAPE.tolist() + [config.SUPPORT_NUMBER,], name='input_targets')  # targets - [b, w, h, c, s_n]
        
        input_image_meta = KL.Input(shape=[config.IMAGE_META_SIZE],
                                    name="input_image_meta")  # meta - [b, 13]
        input_supports_meta = KL.Input(shape=[config.IMAGE_META_SIZE, config.SUPPORT_NUMBER], 
                                       name='input_target_meta') # target_meta - [b, 13, sup_nb]
        input_supports_bbox = KL.Input(shape=[4, config.SUPPORT_NUMBER],
                                       name='input_support_bbox')  # tbs - [b, 4, s_n]
        mask_head_kwargs = {'train_bn': config.TRAIN_BN}
        input_targets, supports_meta, supports_bbox = self.prepare_input(mode, config, 
                                                         input_targets_tensor, input_supports_meta, input_supports_bbox)
        if mode == 'training':
            input_rpn_match = KL.Input(
                shape=[None, 1], name="input_rpn_match", dtype=tf.int32)  # rpn_match - [b, i_n, 1]
            input_rpn_bbox = KL.Input(
                shape=[None, 4], name="input_rpn_bbox", dtype=tf.float32)  # rpn_bbox - [b, i_n, 4]
        else:
            input_gt_class_ids = KL.Input(
                shape=[None], name="input_gt_class_ids", dtype=tf.int32)  # gt_class - [b, i_n]
            input_gt_boxes = KL.Input(
                shape=[None, 4], name="input_gt_boxes", dtype=tf.float32)  # gt_boxes - [b, i_n, 4]
            gt_boxes = KL.Lambda(lambda x: modellib.norm_boxes_graph(
                x, K.shape(input_image)[1:3]))(input_gt_boxes)  # gt_boxes - [b, i_n, 4]
            # This is for implementation convenience.
            if config.MODEL == 'mrcnn':
                if config.USE_MINI_MASK:
                    input_gt_masks = KL.Input(
                        shape=[config.MINI_MASK_SHAPE[0],
                               config.MINI_MASK_SHAPE[1], None],
                        name="input_gt_masks", dtype=bool)  # gt_mask - [b, 56, 56, i_n]
                else:
                    input_gt_masks = KL.Input(
                        shape=[config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1], None],
                        name="input_gt_masks", dtype=bool)  # gt_mask - [b, 512, 512, i_n] (not use)
        
        resnet = build_resnet_model(config)
        fpn = build_fpn_model(config.FPN_FEATUREMAPS)
        _, C2, C3, C4, C5 = resnet(input_image)
        P2, P3, P4, P5, P6 = fpn([C2, C3, C4, C5])
        targets_cls_featmaps = []
        targets_mask_featmaps = []
        cls_pyramid = modellib.PyramidROIAlign([config.POOL_SIZE, config.POOL_SIZE],
                                               name='cls_pyramid')  # POOL_SIZE = 7
        mask_pyramid = modellib.PyramidROIAlign([config.MASK_POOL_SIZE, config.MASK_POOL_SIZE],
                                                name='mask_pyramid')  # MASK_POOL_SIZE = 14
        for img, bbox, meta in zip(input_targets, supports_bbox, supports_meta):
            _, TC2, TC3, TC4, TC5 = resnet(img)
            TP2, TP3, TP4, TP5, TP6 = fpn([TC2, TC3, TC4, TC5])
            cls_featmap = cls_pyramid([bbox, meta, TP2, TP3, TP4, TP5])
            mask_featmap = mask_pyramid([bbox, meta, TP2, TP3, TP4, TP5])
            targets_cls_featmaps.append(cls_featmap)
            targets_mask_featmaps.append(mask_featmap)
        
        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        mrcnn_feature_maps = [P2, P3, P4, P5]
        anchors = self.get_anchors(config.IMAGE_SHAPE)  # anchors - [a_n, 4]
        # Duplicate across the batch dimension because Keras requires it
        anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)  # anchors -[b, a_n, 4]
        # A hack to get around Keras's bad support for constants
        anchors = KL.Lambda(lambda x: tf.Variable(anchors), name="anchors")(input_image)
        rpn_args = [config.RPN_ANCHOR_STRIDE, len(config.RPN_ANCHOR_RATIOS), self.config.FPN_FEATUREMAPS, config]
        rpn = build_attention_rpn_model(*rpn_args) if config.ATTENTION == 'spatial' else build_channel_attention_rpn_model(*rpn_args)

        # Loop through pyramid layers
        layer_outputs = []  # list of lists
        squeezer = KL.Lambda(lambda x: tf.squeeze(x, axis=1), name='squeezer')
        for p in rpn_feature_maps:
            level_logits, level_probs, level_bbox = [], [], []
            for sf in targets_cls_featmaps: 
                sf = squeezer(sf)
                logits, probs, bbox = rpn([p, sf])
                level_logits.append(logits)
                level_probs.append(probs)
                level_bbox.append(bbox)
            level_logits = Concat(axis=1)(level_logits)
            level_probs = Concat(axis=1)(level_probs)
            level_bbox = Concat(axis=1)(level_bbox)
            layer_outputs.append([level_logits, level_probs, level_bbox])
            # layer_outputs = [P2~P6's [[rpn_class_logits], [rpn_probs], [rpn_bbox]]]
        # Concatenate layer outputs
        # Convert from list of lists of level outputs to list of lists
        # of outputs across levels.
        # e.g. [[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_names = ["rpn_class_logits", "rpn_class", "rpn_bbox"]
        outputs = list(zip(*layer_outputs))
        outputs = [Concat(axis=1, name=n)(list(o))
                   for o, n in zip(outputs, output_names)]
        rpn_class_logits, rpn_class, rpn_bbox = outputs
        proposal_count = config.POST_NMS_ROIS_TRAINING if mode == "training" \
            else config.POST_NMS_ROIS_INFERENCE  
        if mode == 'inference':
            rpn_rois = modellib.ProposalLayer(proposal_count=proposal_count,  
                                              nms_threshold=config.RPN_NMS_THRESHOLD,  
                                              name="ROI", config=config)([rpn_class, rpn_bbox, anchors])
            rois, target_class_ids, target_bbox, target_mask, paddings = \
                DetectionTargetLayer(config, name="proposal_targets")([
                    rpn_rois, input_gt_class_ids, gt_boxes, input_gt_masks])
            clipped_rpn_rois = ROIDetectionLayer('proposal', config, name='proposal_clip')(rpn_rois)
            clipped_rois = ROIDetectionLayer('detection_target', config, name='roi_clip')(rois)
            inputs = [input_image, input_image_meta, input_target_tensor, input_supports_meta, input_supports_bbox,
                      input_gt_class_ids, input_gt_boxes, input_gt_masks]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox, clipped_rpn_rois, clipped_rois, target_class_ids] 
        else:
            rpn_class_loss = KL.Lambda(lambda x: modellib.rpn_class_loss_graph(*x), name="rpn_class_loss")(
                [input_rpn_match, rpn_class_logits])
            rpn_bbox_loss = KL.Lambda(lambda x: modellib.rpn_bbox_loss_graph(config, *x), name="rpn_bbox_loss")(
                [input_rpn_bbox, input_rpn_match, rpn_bbox])
            inputs = [input_image, input_image_meta, input_targets_tensor, input_supports_meta, input_supports_bbox,
                      input_rpn_match, input_rpn_bbox]
            outputs = [rpn_class_logits, rpn_class, rpn_bbox, rpn_class_loss, rpn_bbox_loss]
        model = KM.Model(inputs, outputs, name='mrcnn_stage1')
        if config.GPU_COUNT > 1:
            from .mrcnn.parallel_model import ParallelModel
            model = ParallelModel(model, config.GPU_COUNT)
        return model
        
    def compile(self, learning_rate, momentum):
        """ Override mm-rcnn compiling method to remove extra losses. """
        # Optimizer object
        optimizer = keras.optimizers.SGD(
            lr=learning_rate, momentum=momentum,
            clipnorm=self.config.GRADIENT_CLIP_NORM)
        self.keras_model._losses = []
        self.keras_model._per_input_losses = {}
        loss_names = [
            "rpn_class_loss", "rpn_bbox_loss"]
        for name in loss_names:
            layer = self.keras_model.get_layer(name)
            if layer.output in self.keras_model.losses:
                continue
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.add_loss(loss)

        # Add L2 Regularization
        # Skip gamma and beta weights of batch normalization layers.
        reg_losses = [
            keras.regularizers.l2(self.config.WEIGHT_DECAY)(w) / tf.cast(tf.size(w), tf.float32)
            for w in self.keras_model.trainable_weights
            if 'gamma' not in w.name and 'beta' not in w.name]
        self.keras_model.add_loss(tf.add_n(reg_losses))

        # Compile
        self.keras_model.compile(
            optimizer=optimizer,
            loss=[None] * len(self.keras_model.outputs))

        # Add metrics for losses
        for name in loss_names:
            if name in self.keras_model.metrics_names:
                continue
            layer = self.keras_model.get_layer(name)
            self.keras_model.metrics_names.append(name)
            loss = (
                    tf.reduce_mean(layer.output, keepdims=True)
                    * self.config.LOSS_WEIGHTS.get(name, 1.))
            self.keras_model.metrics_tensors.append(loss)
            
    def set_log_dir(self, model_path=None):
        time_now = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        self.epoch = 0
        self.log_dir = os.path.join(self.model_dir,
                                    "matching_{}_{}_{}".format(self.config.MODEL.lower(),
                                                               self.config.NAME.lower(),
                                                               self.config.EXPERIMENT.lower()))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        self.checkpoint_path = os.path.join(self.log_dir, "stage1_*epoch*.h5")
        self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")
        
    def train(self, train_dataset, val_dataset, learning_rate, epochs, regex, augmentation=None):
        assert self.mode == 'training', 'creating model in training mode'
        layer_regex = {
            "2+": r"(res2.*)|(bn2.*)|(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            # All layers
            "all": r"(res.*)|(bn.*)|(fpn\_.*)|(arpn\_.*)"
        }
        layers = layer_regex[regex]
        train_generator = my_utils.stage1_data_generator(train_dataset, self.config, shuffle=True,
                                                           augmentation=augmentation,
                                                           batch_size=self.config.BATCH_SIZE,
                                                           random_rois=0,
                                                           detection_targets=False,
                                                           get_mask=False
                                                           )
        val_generator = my_utils.stage1_data_generator(val_dataset, self.config, shuffle=True,
                                                           augmentation=augmentation,
                                                           batch_size=self.config.BATCH_SIZE,
                                                           random_rois=0,
                                                           detection_targets=False,
                                                           get_mask=False
                                                           )
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir,
                                        histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path,
                                            verbose=0, save_weights_only=True),
        ]

        # Train
        modellib.log("\nStarting at epoch {}. LR={}\n".format(self.epoch, learning_rate))
        modellib.log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(layers)
        self.compile(learning_rate, self.config.LEARNING_MOMENTUM)

        # Work-around for Windows: Keras fails on Windows when using
        # multiprocessing workers. See discussion here:
        # https://github.com/matterport/Mask_RCNN/issues/13#issuecomment-353124009
        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=epochs,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=val_generator,
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True,
        )
        self.epoch = max(self.epoch, epochs)
        
    def unmold_rois(self, rois, original_image_shape, image_shape, window):
        zero_ix = np.where(rois[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else rois.shape[0]
        boxes = rois[:N, :4]
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        boxes = np.divide(boxes - shift, scale)
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])
        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            N = boxes.shape[0]
        masks = np.zeros((original_image_shape[:2], N))
        class_ids = np.ones(N)
        scores = np.ones(N).astype(np.float32)
        return boxes, class_ids, scores, masks

    def pseudo_detect(self, model_inp):
        assert self.mode == 'inference' and self.config.BATCH_SIZE == 1
        rpn_class_logits, rpn_class, rpn_bbox, rpn_rois, rois, roi_class_ids = self.keras_model.predict(model_inp, verbose=0)
        image_meta = model_inp[1]
        m = modellib.parse_image_meta(image_meta)
        original_image_shape, image_shape, window = m['original_image_shape'][0],\
            m['image_shape'][0], m['window'][0]
        results = []
        for i in range(batch):
            final_rois, final_class_ids, final_scores, final_masks =\
                self.unmold_rois(rpn_rois[i], original_image_shape, image_shape, window)
            detection_rois, _, _, _ = self.unmold_rois(rois[i], original_image_shape, image_shape, window)
            result = {
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "train_rois": detection_rois, 
                "train_rois_id": roi_class_ids
            }
            results.append(result)
        return results
        
    def load_checkpoint(self, weights_path, training_schedule=None, verbose=1):
        if verbose > 0: print('loading', weights_path, '...')
        layer_regex = {
            "2+": r"(res2.*)|(bn2.*)|(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "3+": r"(res3.*)|(bn3.*)|(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "4+": r"(res4.*)|(bn4.*)|(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            "5+": r"(res5.*)|(bn5.*)|(arpn\_.*)|(fpn\_.*)",
            # All layers
            "all": r"(res.*)|(bn.*)|(fpn\_.*)|(arpn\_.*)"
        }
        epoch_index = int(weights_path[-7:-3])
        if verbose > 0:
            print('starting from epoch {}'.format(epoch_index))
        if training_schedule is not None:
            # get correct schedule period
            schedule_index = min([key for key in training_schedule.keys() if epoch_index <= key])
            layers = layer_regex[training_schedule[schedule_index]["layers"]]
            self.set_trainable(layers)
        else: self.set_trainable(".*")
        try:
            if weights_path:
                self.load_weights(weights_path, by_name=True)
        except:
            if weights_path:
                self.keras_model.inner_model.load_weights(weights_path, by_name=True)
        self.epoch = epoch_index
            
