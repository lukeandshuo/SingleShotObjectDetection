from preprocessing import preprocessing_infer
from util import tf_image_processing
import tensorflow as tf
import numpy as np
import os
from PIL import Image
from nets import ssd_net
import matplotlib.pyplot  as plt
from postprocessing import postprocessing_infer
from visualizations import img_with_bboxes
slim = tf.contrib.slim

import time
if __name__=="__main__":

    size = (300, 300)
    input_imgs = tf.placeholder(tf.float32, shape=(None, None, 3))
    resized_imgs = preprocessing_infer.preprocessing_fn(input_imgs, size)
    resized_imgs = tf.expand_dims(resized_imgs,0)
    with slim.arg_scope(ssd_net.ssd_arg_scope()):
        localizations, classifications, logits, end_points = ssd_net.ssd_net(resized_imgs,size)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True


    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        ##load image and transform into narray format
        imgs_path = "images/"
        names = sorted(os.listdir(imgs_path))
        img = Image.open(imgs_path + names[-2])
        np_img = np.asarray(img, dtype=np.float32)

        ##load checkpoints
        checkpoints_dir = "checkpoints/ssd_300_vgg.ckpt"
        saver = tf.train.Saver()
        saver.restore(sess,checkpoints_dir)
        t_start = time.clock()

        ##run the function
        locals,classes = sess.run([localizations,classifications], feed_dict={input_imgs: np_img})

        ## post processing
        classes, scores, bboxes = postprocessing_infer.postprocessing_fn(classes,locals,size)

        t_end = time.clock()
        print "total time", t_end-t_start

        ## visualization
        img_with_bboxes.plt_bboxes(np_img.astype(np.uint8),classes,scores,bboxes)

        ### feature visualization
        # res_shape = result.shape
        # feat_map = np.reshape(result, res_shape[1:])
        # mean_feat_map = feat_map.mean(axis=2)
        # plt.imshow(mean_feat_map)
        # plt.show()
        # tf_image_processing.image_show(result)