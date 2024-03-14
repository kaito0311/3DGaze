import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from utils import *


path_base = '../data/example_images'

# load gaze estimation results
with open('../inference_results/vertex/ALL/test_0/inference/results_vertex_mode.pkl', 'rb') as f:
    results = pickle.load(f)
results = {k: v for k, v in results.items()}
paths = list(results.keys())
len(results)

# load preprocessing data
with open('../output/preprocessing/data_face68.pkl', 'rb') as f:
    data_pproc = pickle.load(f)
data_pproc = {d['name']: d for d in data_pproc}
len(data_pproc)


# Randomly loop through the results
k = np.random.randint(len(paths))
path = paths[k]

res = results[path]
path = '../' + path
image = cv2.imread(path)
lms68 = data_pproc[os.path.relpath(path, path_base)]['face']['xyz68'].astype(np.float32)
# lms5 = data_pproc[os.path.relpath(path, path_base)]['face']['xy5'].astype(np.float32)
eyes = {
    'left' : (res['eyes']['left']['P'] @ eyel_template_homo.T).T,
    'right': (res['eyes']['right']['P'] @ eyer_template_homo.T).T
}
gaze_vector = res['gaze_vec_combined']
print(f"Gaze direction: {gaze_vector}")

# visualize gaze direction
image_gaze = draw_gaze_from_vector(image.copy(), lms68, gaze_vector, colour=[255, 0, 0])
# print(lms5)
# image_gaze = draw_gaze_from_vector(image.copy(), lms5, gaze_vector, colour=[255, 0, 0])
plt.imsave("hi.jpg", image_gaze[:, :, [2, 1, 0]])


# visualize 3D eyes
image_eyes = draw_eyes(image.copy(), lms68, eyes, colour=[178, 255, 102])
# image_eyes = draw_eyes(image.copy(), lms5, eyes, colour=[178, 255, 102])
plt.imsave("hello.jpg", image_eyes[:, :, [2, 1, 0]])
image.shape