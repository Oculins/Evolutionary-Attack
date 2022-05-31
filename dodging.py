# "Efficient decision-based black-box adversarial attacks on face recognition" Reading Report
#  Dodging Attack
#  implementation by Python, dataset: LFW, face recognition model: FaceNet
#  Author: Oculins Jiang

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import os
import argparse
import facenet
import align.detect_face
import cv2
# import detect_face
from PIL import Image
import matplotlib.pyplot as plt

def L2_dis(img1, img2):
    d0 = np.linalg.norm(img1[:, :, 0] - img2[:, :, 0], 2)
    d1 = np.linalg.norm(img1[:, :, 1] - img2[:, :, 1], 2)
    d2 = np.linalg.norm(img1[:, :, 2] - img2[:, :, 2], 2)
    return np.sqrt(d0*d0+d1*d1+d2*d2)

def mse(img1, img2, size):
    return L2_dis(img1, img2) / (size * size * 3)

model_path = 'model'
image_path1 = 'test_data/Aaron_Peirsol/Aaron_Peirsol_0001.png'
image_path2 = 'test_data/Aaron_Peirsol/Aaron_Peirsol_0004.png'

# parameters for face recognition
image_size = 160
margin = 44
gpu_memory_fraction = 1.0

# image input
img1_input = cv2.imread(image_path1)
img2_input = cv2.imread(image_path2)

# input of optimization
i_size = 45
T = 10000
n = image_size * image_size * 3
m = i_size * i_size * 3
k = int(m/20)
MSE = []

ref_img = img1_input.copy()   # for constraint
origin_img = img2_input.copy()   # for distance
object_img = np.random.randint(0, 256, (160, 160, 3))
cv2.imwrite('./dodging_result_0.png', object_img)
MSE.append(mse(origin_img, object_img, image_size))

C = np.identity(m)   # updating
p_c = np.zeros(m)  # updating
sigma = 0.001*L2_dis(origin_img, object_img)  # updating
c_c = 0.01
c_cov = 0.001
mu = 1  # updating
P_suc = 1  # updating

with tf.Graph().as_default():
    with tf.Session() as sess:
        # Load the model
        facenet.load_model(model_path)

        L_pre = L2_dis(origin_img, object_img)

        # Get input and output tensors
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

        for t in range(T):
            z = np.random.normal(0, sigma*sigma*C)
            z = np.diagonal(z)

            C_diag = np.diagonal(C)
            rate = C_diag / np.sum(C_diag)
            cord = np.random.choice(a=m, size=k, replace=False, p=rate)
            z_r = z.copy()
            for i in range(k):
                z_r[cord[i]] = 0
            z = z - z_r

            z_hat = z.reshape(i_size, i_size, 3)
            z_hat = cv2.resize(z_hat, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

            z_hat = z_hat + mu * (origin_img - object_img)

            dis_pre = L2_dis(origin_img, object_img)
            dis_post = L2_dis(origin_img, object_img + z_hat)


            images_post = np.zeros((2, 160, 160, 3))
            images_post[0] = facenet.prewhiten(ref_img)
            images_post[1] = facenet.prewhiten(object_img + z_hat)
            feed_dict_post = {images_placeholder: images_post, phase_train_placeholder: False}
            emb_post = sess.run(embeddings, feed_dict=feed_dict_post)
            dist_post = np.sqrt(np.sum(np.square(np.subtract(emb_post[0, :], emb_post[1, :]))))

            L_post = dis_post if (dist_post > 1.2) else float('inf')

            # print(dist_post, L_pre, L_post)
            if L_post < L_pre:
                object_img = object_img + z_hat
                L_pre = L_post
                p_c = (1-c_c) * p_c + np.sqrt(c_c * (2-c_c)) * z / sigma
                for i in range(m):
                    C[i, i] = (1-c_cov) * C[i, i] + c_cov * p_c[i] * p_c[i]
                sigma = 0.001*L2_dis(origin_img, object_img)

                P_suc = (t * P_suc + 1)/(t+1) if (t > 0 and t < 20) else (20 * P_suc + 1)/21

            else:
                P_suc = (t * P_suc)/(t+1) if (t > 0 and t < 20) else (20 * P_suc)/21

            mu = mu * np.exp(P_suc - 0.2)

            if t % 10 == 0:
                print('queries:', t)

            if t == 99 or t == 999 or t == 1999:
                cv2.imwrite('./dodging_result_'+str(t)+'.png', object_img)
            MSE.append(mse(origin_img, object_img, image_size))


cv2.imwrite('./dodging_result_10000.png', object_img)


fig = plt.figure()
xx = np.arange(T+1)
yy = np.array(MSE)
plt.plot(xx, yy, color='blue')
plt.xlabel('queries')
plt.ylabel('MSE')
plt.title('Dodging Attack')
plt.savefig('./Dodging_Attack.png')

