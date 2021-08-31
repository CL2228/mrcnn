"""
Current implementation status notes:
- 3 way 1 shot
- prediction made by REPMET-like approach
- classification loss is similar to REPMET
- use SG-One style to modulate mask branch
- abandon class attention but select corresponding support given class predictions
- use random rois for training this time
Zhibo Fan
2019.08.09
"""
import numpy as np


# Base Configuration Class
# Don't use this class directly. Instead, sub-class it and override
# the configurations you need to change.

class Config(object):
    """Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.
    """
    ATTENTION = 'spatial'
    
    DETECTION_DEBUG = False

    MASK_LATE_FUSION = False

    PRE_NMS_LIMIT = 6000
    
    VALIDATION_STEPS = 50
    # Number of class of reference offered, each shot refers to a different class
    WAYS = 3
    # Number of support images to input per class
    SHOTS = 1

    # Name the configurations. For example, 'COCO', 'Experiment 3', ...etc.
    # Useful if your code needs to do things differently depending on which
    # experiment is running.
    NAME = None  # Override in sub-classes
    
    # CHANGE: Added experiment name parameter
    # Name the experiment. For example "all_classes", "experiment_3", ...etc.
    # Useful for running multiple experiments with the same basic model
    EXPERIMENT = None # Override in sub-class
    
    # CHANGE: Added MODEL parameter
    # Select wether to use Mask R-CNN or Faster R-CNN
    # Must be one of: 'mrcnn' or 'frcnn'
    MODEL = 'mrcnn'
    
    # CHANGE: Added CHECKPOINT_DIR
    # Directory with pretrained checkpoints
    CHECKPOINT_DIR = 'checkpoints/'

    # NUMBER OF GPUs to use. For CPU training, use 1
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically
    # handle 2 images of 1024x1024px.
    # Adjust based on your GPU memory and image sizes. Use the highest
    # number that your GPU can handle for best performance.
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    # This doesn't need to match the size of the training set. Tensorboard
    # updates are saved at the end of each epoch, so setting this to a
    # smaller number means getting more frequent TensorBoard updates.
    # Validation stats are also calculated at each epoch end and they
    # might take a while, so don't set this too small to avoid spending
    # a lot of time on validation stats.
    STEPS_PER_EPOCH = 1000

    # Backbone network architecture
    # Supported values are: resnet50, resnet101.
    # You can also provide a callable that should have the signature
    # of model.resnet_graph. If you do so, you need to supply a callable
    # to COMPUTE_BACKBONE_SHAPE as well
    # CHANGE: Use smaller resnet50 for efficiency
    BACKBONE = "resnet50"

    # Only useful if you supply a callable to BACKBONE. Should compute
    # the shape of each layer of the FPN Pyramid.
    # See model.compute_backbone_shapes
    COMPUTE_BACKBONE_SHAPE = None

    # The strides of each layer of the FPN Pyramid. These values
    # are based on a Resnet101 backbone.
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

    # Size of the fully-connected layers in the classification graph
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    FPN_FEATUREMAPS = 256
    TOP_DOWN_PYRAMID_SIZE = FPN_FEATUREMAPS

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1 then anchors are created for each cell in the backbone feature map.
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # ROIs kept after non-maximum suppression (training and inference)
    POST_NMS_ROIS_TRAINING = 500
    POST_NMS_ROIS_INFERENCE = 500

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = True
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Input image resizing
    # Generally, use the "square" resizing mode for training and predicting
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM or IMAGE_MIN_SCALE are not None, then it scales
    #         up before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    # crop:   Picks random crops from the image. First, scales the image based
    #         on IMAGE_MIN_DIM and IMAGE_MIN_SCALE, then picks a random crop of
    #         size IMAGE_MIN_DIM x IMAGE_MIN_DIM. Can be used in training only.
    #         IMAGE_MAX_DIM is not used in this mode.
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024
    IMAGE_MIN_SCALE = 0
    
    # CHANGE: Target resizing parameters added
    # Input target resizing
    # The same rukes apply as for image resizing
    # Currently the resizing is done through TARGET_PADDING through which the 
    # target will automatically be padded to [max_dim, max_dim]
    TARGET_PADDING = True
    TARGET_MAX_DIM = 160
    TARGET_MIN_DIM = 125
    TARGET_SIZE_LIMIT = 0.001
    

    # Image mean (RGB)
    MEAN_PIXEL = np.array([123.7, 116.8, 103.9])

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128
    ROI_POSITIVE_RATIO = 0.33

    # Pooled ROIs
    POOL_SIZE = 7
    MASK_POOL_SIZE = 14

    # Shape of output mask
    # To change this you also need to change the neural network mask branch
    MASK_SHAPE = [28, 28]

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 30

    # Bounding box refinement standard deviation for RPN and final detections.
    RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])
    BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2])

    # Max number of final detections
    DETECTION_MAX_INSTANCES = 30

    # Minimum probability value to accept a detected instance
    # ROIs below this threshold are skipped
    DETECTION_MIN_CONFIDENCE = 0.5

    # Non-maximum suppression threshold for detection
    DETECTION_NMS_THRESHOLD = 0.3

    # Learning rate and momentum
    # The Mask RCNN paper uses lr=0.02, but on TensorFlow it causes
    # weights to explode. Likely due to differences in optimizer
    # implementation.
    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9

    # Weight decay regularization
    WEIGHT_DECAY = 0.0001

    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    LOSS_WEIGHTS = {
        "rpn_class_loss": 2.,
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 2.,
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
    
    # Whether or not use random rois when training and inference
    USE_RPN_ROIS = True
    USE_RPN_ROIS_INFERENCE = True

    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when predicting
    TRAIN_BN = False

    # Gradient norm clipping
    GRADIENT_CLIP_NORM = 5.0

    # VOC's class_id and its name
    VOC_CLASS_ID = {    
	    1: "person",
	    2: "bicycle",
	    3: "car",
	    4: "motorcycle",
	    5: "airplane",
	    6: "bus",
	    7: "train",
	    9: "boat",
	    15: "bird",
	    16: "cat",
	    17: "dog",
	    18: "horse",
	    19: "sheep",
	    20: "cow",
	    40: "bottle",
	    57: "chair",
	    58: "couch",
	    59: "potted plant",
	    61: "dining table",
	    63: "tv/monitor"
    }

    # COCO's class_id and its name
    CLASS_ID = {    
        1: "person",
        2: "bicycle",
        3: "car",
        4: "motorcycle",
        5: "airplane",
        6: "bus",
        7: "train",
        8: "truck",
        9: "boat",

        10: "traffic light",
        11: "fire hydrant",
        12: "stop sign",
        13: "parking meter",
        14: "bench",

        15: "bird",
        16: "cat",
        17: "dog",
        18: "horse",
        19: "sheep",
        20: "cow",
        21: "elephant",
        22: "bear",
        23: "zebra",
        24: "giraffe",

        25: "backpack",
        26: "umbrella",
        27: "handbag",
        28: "tie",
        29: "suitcase",

        30: "frisbee",
        31: "skis",
        32: "snowboard",
        33: "sports ball",
        34: "kite",
        35: "baseball bat",
        36: "baseball glove",
        37: "skateboard",
        38: "surfboard",
        39: "tennis racket",

        40: "bottle",
        41: "wine glass",
        42: "cup",
        43: "fork",
        44: "knife",
        45: "spoon",
        46: "bowl",
        
        47: "banana",
        48: "apple",
        49: "sandwich",
        50: "orange",
        51: "broccoli",
        52: "carrot",
        53: "hot dog",
        54: "pizza",
        55: "donut",
        56: "cake",
        
        57: "chair",
        58: "couch",
        59: "potted plant",
        60: "bed",
        61: "dining table",
        62: "toilet",
        
        63: "tv/monitor",
        64: "laptop",
        65: "mouse",
        66: "remote",
        67: "keyboard",
        68: "cell phone",
        
        69: "microwave",
        70: "oven",
        71: "toaster",
        72: "sink",
        73: "refrigerator",
        74: "book",
        75: "clock",
        76: "vase",
        77: "scissors",
        78: "teddy bear",
        79: "hair drier",
        80: "toothbrush"
    }

    DATASET_TYPE = 'coco'


    def __init__(self):
        """Set values of computed attributes."""
        # Effective batch size
        self.BATCH_SIZE = self.IMAGES_PER_GPU * self.GPU_COUNT
        self.SUPPORT_NUMBER = self.WAYS * self.SHOTS
        self.NUM_CLASSES = self.WAYS + 1
            
        if self.MODEL != 'mrcnn':
            self.MASK_LATE_FUSION = False

        self.TARGET_SHAPE = np.array([self.TARGET_MAX_DIM, self.TARGET_MAX_DIM, 3])
        self.IMAGE_SHAPE = np.array([self.IMAGE_MAX_DIM, self.IMAGE_MAX_DIM, 3])

        # Image meta data length
        # See compose_image_meta() for details
        self.IMAGE_META_SIZE = 13 # do not change this, it has to be larger than 12, however, only first 12 elements are used

    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
print("\n")
