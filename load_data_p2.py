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
import torch.nn.utils as utils 
import csv
import collections

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else "cpu")

transform = transforms.Compose([
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
	])



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
		self.train = train
		self.video_path = video_path
		self.label_path = label_path
		
		#{'Action_labels', 'Nouns', 'End_times', 'Start_times', 'Video_category', 'Video_index', 'Video_name'}
		self.videos_list = reader.getVideoList(self.label_path)
	def __len__(self):
		return len(self.videos_list['Action_labels'])

	def __getitem__(self, idx):
		
		frame = reader.readShortVideo(self.video_path, self.videos_list['Video_category'][idx], self.videos_list['Video_name'][idx])
		#for f in frame:
		#	frames.append(transform(f))
		#frame = torch.stack([f for f in frames])

		frame = frame.transpose((0,3,1,2))


		frame = torch.FloatTensor(frame)
		label = int(self.videos_list['Action_labels'][idx])
		label = torch.LongTensor([label]).to(device)
		
		return frame, label
		
# if __name__ == '__main__':

# 	dataloader = DATASET('./hw4_data/TrimmedVideos/video/train', './hw4_data/TrimmedVideos/label/gt_train.csv')
# 	dataloader = DataLoader(dataloader, batch_size = 2,  shuffle=False, collate_fn=my_collate_fn)
# 	for step, batch in enumerate(dataloader):
# 	  if step == 10:
# 	      break
# 	  frame, label, len_ = batch
# 	  print(len_)
# 	  print(frame.size())