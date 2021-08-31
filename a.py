import lib.model as modellib
import keras.utils
from keras.utils.vis_utils import plot_model
from lib.config import Config
import numpy as np
import keras.backend as K
import tensorflow as tf
# modell = modellib.MatchingMaskRCNN('training', Config(), "../")
# modell.keras_model.summary()
# plot_model(modell.keras_model, to_file='training.png', show_shapes=True)
sess = tf.Session()
a = np.array([[2,4,8],[7,45,5]])
a = tf.constant(a)
a = tf.expand_dims(a, axis=-1)
b = K.argmax(a, axis=1)
print(sess.run(b))
c = tf.one_hot(b, depth=5, axis=1) #[2,5,1

# c = np.concatenate(a,axis=1)
print(sess.run(c))