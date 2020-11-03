import numpy as np
import os
from os.path import join
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch 
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms
from os import listdir
import pandas as pd
import reader
import load_inception_v3

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')


def my_collate_fn(batch):
	
	batch.sort(key=lambda x:len(x[0]), reverse=True)
	frame, label = zip(*batch)
	length_list = []
	frame = list(frame)
	label = list(label)
	max_length = len(frame[0])
	for ele in frame:
		length_list.append(len(ele))

	# frame = torch.stack([ele for ele in frame])
	frame = torch.cat([ele for ele in frame], dim=0)
	label = torch.stack([ele for ele in label])

	return frame, label, length_list



class DATASET(Dataset):

	def __init__(self, video_path, label_path, train=True):
		self.video_path = video_path
		self.label_path = label_path
		self.model = load_inception_v3.extractor(pretrained=True).to(device)
		self.model.eval()
		# {'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'}
		self.videos_list = reader.getVideoList(self.label_path)

	def __len__(self):
		return len(self.videos_list['Action_labels'])

	def __getitem__(self, idx):
		
		frame = reader.readShortVideo(self.video_path, self.videos_list['Video_category'][idx], self.videos_list['Video_name'][idx])
		

		label = self.videos_list['Action_labels'][idx]
		label = int(label)

		frame = frame.transpose((0, 3, 1, 2))
		frame = torch.FloatTensor(frame).to(device)
		with torch.no_grad():
			frame = self.model(frame)
		frame = torch.mean(frame, dim=0)

		label = torch.LongTensor([label])

		return frame, label


# test = DATASET('./hw4_data/TrimmedVideos/video/train', './hw4_data/TrimmedVideos/label/gt_train.csv')
# test = DataLoader(test, batch_size = 5)

# for batch in test:
# 	print(batch)