import os
import re
import sys
import cv2
import math
import time
import scipy
import argparse
import matplotlib
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

openpose_dir = Path('src/pytorch_Realtime_Multi-Person_Pose_Estimation/')
import sys
sys.path.append(str(openpose_dir))

save_dir = Path('data/results/images/1/')

import pylab as plt
import torch
#print(torch.__version__)
#print(torch.cuda.is_available())

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
#from config_reader import config_reader
from scipy.ndimage.filters import gaussian_filter
from network.rtpose_vgg import get_model
from network.post import decode_pose
from training.datasets.coco_data.preprocessing import (inception_preprocess,
												rtpose_preprocess,
												ssd_preprocess, vgg_preprocess)
from network import im_transform
from evaluate.coco_eval import get_multiplier, get_outputs
#parser = argparse.ArgumentParser()
#parser.add_argument('--t7_file', required=True)
#parser.add_argument('--pth_file', required=True)
#args = parser.parse_args()

weight_name = 'src/pytorch_Realtime_Multi-Person_Pose_Estimation/network/weight/pose_model.pth'
model = get_model('vgg19')
model.load_state_dict(torch.load(weight_name))
model.cuda()
model.float()
model.eval()

if __name__ == "__main__":

	# video_capture = cv2.VideoCapture(0)
	video_capture = cv2.VideoCapture('data/source/videos/1.avi')
	#idx = 0
	frames_num = video_capture.get(7)
	print(frames_num)

	for idx in tqdm(range(int(frames_num))):
		if video_capture.isOpened():
		#if idx > 500:
		    # Capture frame-by-frame
			ret, oriImg = video_capture.read()
		    # plt.figure("Image")
		    # plt.imshow(oriImg)
		    # plt.axis('off')
		    # plt.title('image')
		    # plt.show()

			shape_h = oriImg.shape[0]
			shape_w = oriImg.shape[1]
		    # print(shape_dst)
			back_img = cv2.imread('black.jpg')
			back_img = cv2.resize(back_img, (shape_w, shape_h), interpolation=cv2.INTER_AREA)
		    # Get results of original image
			multiplier = get_multiplier(oriImg)

			with torch.no_grad():
				paf, heatmap = get_outputs(
					multiplier, oriImg, model, 'rtpose')

			param = {'thre1': 0.1, 'thre2': 0.05, 'thre3': 0.5}
			canvas, to_plot, candidate, subset = decode_pose(
				back_img, param, heatmap, paf)

		    # Display the resulting frame
		    # cv2.imshow('Video', to_plot)
		    # plt.figure("Video")
		    # plt.imshow(to_plot)
		    # plt.axis('off')
		    # plt.title('result')
		    # plt.show()
			cv2.imwrite(str(save_dir.joinpath(f'result_{idx:04d}.png')),to_plot)
		    #idx += 1
		else:
			break

	video_capture.release()

	oriImg = cv2.imread(str(save_dir) + '/' + 'result_0000.png')
	shape_h = oriImg.shape[0]
	shape_w = oriImg.shape[1]

	filelist = os.listdir(save_dir)
	fps = 24
	size = (shape_w, shape_h)
	fourcc = cv2.VideoWriter_fourcc(*"MJPG")
	video = cv2.VideoWriter('data/results/1_result.avi', fourcc, fps, size)
	for item in tqdm(filelist):
		if item.endswith('.png'):
			item = str(save_dir) + '/' + item
			img = cv2.imread(item)
			video.write(img)
	video.release()
	print('Done')