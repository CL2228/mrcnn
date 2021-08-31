import keras.backend as K
import keras.models as KM
import keras.engine as KE
import keras.layers as KL
import tensorflow as tf
import os
import numpy as np

# from mrcnn import model as modellib
# from mrcnn import utils as utils
from lib.mrcnn import utils
from lib.mrcnn import model as modellib
from lib.mrcnn import visualize
from lib import utils as my_utils
from lib import config as config
from h5py import File

import matplotlib.pyplot as plt
import random
import numpy

config = config.Config()
config.GPU_COUNT = 1
config.IMAGES_PER_GPU = 32
config.BATCH_SIZE = config.GPU_COUNT * config.IMAGES_PER_GPU

def fpn_graph(C2, C3, C4, C5, feature_maps=128):
    P5 = KL.Conv2D(feature_maps, (1, 1), activation="relu", name='fpn_c5p5')(C5)
    P4 = KL.Add(name="fpn_p4add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p5upsampled")(P5),
        KL.Conv2D(feature_maps, (1, 1), activation="relu", name='fpn_c4p4')(C4)])
    P3 = KL.Add(name="fpn_p3add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p4upsampled")(P4),
        KL.Conv2D(feature_maps, (1, 1), activation="relu", name='fpn_c3p3')(C3)])
    P2 = KL.Add(name="fpn_p2add")([
        KL.UpSampling2D(size=(2, 2), name="fpn_p3upsampled")(P3),
        KL.Conv2D(feature_maps, (1, 1), activation="relu", name='fpn_c2p2')(C2)])
    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = KL.Conv2D(feature_maps, (3, 3), activation="relu", padding="SAME", name="fpn_p2")(P2)
    P3 = KL.Conv2D(feature_maps, (3, 3), activation="relu", padding="SAME", name="fpn_p3")(P3)
    P4 = KL.Conv2D(feature_maps, (3, 3), activation="relu", padding="SAME", name="fpn_p4")(P4)
    P5 = KL.Conv2D(feature_maps, (3, 3), activation="relu", padding="SAME", name="fpn_p5")(P5)
    # P6 is used for the 5th anchor scale in RPN. Generated by
    # subsampling from P5 with stride of 2.
    P6 = KL.MaxPooling2D(pool_size=(1, 1), strides=2, name="fpn_p6")(P5)
    return [P2, P3, P4, P5, P6]

class FPNFeatureModel: 
    def __init__(self, config, weight_path='checkpoints/pretrained_frcnn.h5', weight_check=True):
        self.config = config
        self.model = self.build(config)
        self.load_weight(weight_path, weight_check)
        
    def build(self, config):
        input_image = KL.Input(
            shape=config.IMAGE_SHAPE.tolist(), name="input_image")
        _, C2, C3, C4, C5 = modellib.resnet_graph(input_image, config.BACKBONE, stage5=True, train_bn=config.TRAIN_BN)
        P2, P3, P4, P5, _ = fpn_graph(C2, C3, C4, C5, feature_maps=config.FPN_FEATUREMAPS)
        averager = KL.Lambda(lambda x: tf.reduce_mean(x, axis=-1), name='average_layer')
        avgP2, avgP3, avgP4, avgP5 = averager(P2), averager(P3), averager(P4), averager(P5)
        return KM.Model([input_image], [avgP2, avgP3, avgP4, avgP5])
        
    def forward(self, images):
        assert len(images) == self.config.BATCH_SIZE
        molded_images = []
        for image in images:
            molded_image, window, scale, padding, crop = utils.resize_image(
                    image,
                    min_dim=self.config.IMAGE_MIN_DIM,
                    min_scale=self.config.IMAGE_MIN_SCALE,
                    max_dim=self.config.IMAGE_MAX_DIM,
                    mode=self.config.IMAGE_RESIZE_MODE)
            molded_image = my_utils.mold_image(molded_image, self.config)
            molded_images.append(molded_image)

        molded_images = np.stack(molded_images)
        p2, p3, p4, p5 = self.model.predict(molded_images)
        res = []
        for i in range(p2.shape[0]):
            res.append([p2[i], p3[i], p4[i], p5[i]])
        return res
        
    def load_weight(self, weight_path, weight_check):
        assert hasattr(self, 'model'), "model must be initialized first!"
        self.model.load_weights(weight_path, by_name=True, skip_mismatch=True)
        if weight_check:
            def assertion(a1, a2, name, judge=True):
                if judge is True: func = lambda x, y: np.all(np.equal(x, y))
                else: func = lambda x, y: np.any(np.not_equal(x, y))
                if not func(a1, a2):
                    print('first array: ', a1, '\n')
                    print('second array: ', a2, '\n')
                    print(name + ' check error')
                    
                    exit(-1)
            f = File(weight_path)
            layer_names = [str(n)[2 : -1] for n in f.attrs['layer_names']]
            res_conv_names = [n for n in layer_names if n[:3] == 'res']
            res_conv_names.append('conv1')
            for n in res_conv_names:
                if f[n].attrs['weight_names'].size == 0: continue
                file_kernel = f[n][n]['kernel:0'][:]
                file_bias = f[n][n]['bias:0'][:]
                model_kernel, model_bias = self.model.get_layer(n).get_weights()
                assertion(model_kernel, file_kernel, n + '_kernel')
                assertion(model_bias, file_bias, n + '_bias')
            res_bn_names = [n for n in layer_names if n[:2] == 'bn']
            for n in res_bn_names:
                if f[n].attrs['weight_names'].size == 0: continue
                file_gamma = f[n][n]['gamma:0'][:]
                file_beta = f[n][n]['beta:0'][:]
                file_mov_mean = f[n][n]['moving_mean:0'][:]
                file_mov_var = f[n][n]['moving_variance:0'][:]
                model_gamma, model_beta, model_mov_mean, model_mov_var = self.model.get_layer(n).get_weights()
                assertion(model_gamma, file_gamma, n + '_gamma')
                assertion(model_beta, file_beta, n + '_beta')
                assertion(model_mov_mean, file_mov_mean, n + '_mov_mean')
                assertion(model_mov_var, file_mov_var, n + '_mov_var')
            fpn_names = [n for n in layer_names if n[:3] == 'fpn']
            for n in fpn_names:
                if f[n].attrs['weight_names'].size == 0: continue
                file_kernel = f[n][n]['kernel:0'][:]
                file_bias = f[n][n]['bias:0'][:]
                model_kernel, model_bias = self.model.get_layer(n).get_weights()
                assertion(model_kernel, file_kernel, n + '_kernel')
                assertion(model_bias, file_bias, n + '_bias')
                
def plot(model, dataset, limit=32, random_seed=1):
    image_ids = np.copy(dataset.image_ids)
    np.random.seed(random_seed)
    np.random.shuffle(image_ids)
    assert limit // model.config.BATCH_SIZE != 0 and limit % model.config.BATCH_SIZE == 0, "limit is incompatible with batch size!"
    image_pairs = []
    for i in range(limit // model.config.BATCH_SIZE):
        images = []
        for j in range(model.config.BATCH_SIZE):
            image_id = image_ids[i * model.config.BATCH_SIZE + j]
            image = dataset.load_image(image_id)
            images.append(image)
        outputs = model.forward(images)
        this_pair = []
        for img, feats in zip(images, outputs):
            feats.append(img)
            this_pair.append(feats)
        image_pairs.extend(this_pair)
    for pair in image_pairs:
        ax = plt.subplot(121)
        ax.set_title('image')
        ax.imshow(pair[4])
        ax = plt.subplot(243)
        ax.set_title('P2')
        ax.imshow(pair[0])
        ax = plt.subplot(244)
        ax.set_title('P3')
        ax.imshow(pair[1])
        ax = plt.subplot(247)
        ax.set_title('P4')
        ax.imshow(pair[2])
        ax = plt.subplot(248)
        ax.set_title('P5')
        ax.imshow(pair[3])
        plt.show()
        
if __name__ == '__main__':
    COCO_DATA = '/mnt/Disk4/zbfan/coco/'
    coco_eval = my_utils.IndexedCocoDataset()
    coco_eval.load_coco(COCO_DATA, "val", year="2017")
    coco_eval.prepare()
    coco_eval.build_indices()
    
    import argparse
    parser = argparse.ArgumentParser(description="Visualize FPN Features")
    parser.add_argument('-n', '--number', type=int, default=32)
    parser.add_argument('-s', '--seed', type=int, default=233)
    parser.add_argument('-m', '--model', type=str, default='checkpoints/pretrained_frcnn.h5')
    
    args = parser.parse_args()
    LIMIT = args.number
    SEED = args.seed
    MODEL = args.model
    
    model = FPNFeatureModel(config, weight_path=MODEL, weight_check=True)
    plot(model, coco_eval, limit=LIMIT, random_seed=SEED)
    