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

mean = torch.FloatTensor([0.5, 0.5, 0.5])
std = torch.FloatTensor([0.5, 0.5, 0.5])

transform = transforms.Compose([
	transforms.Resize((299, 299)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])

class DATASET(Dataset):

	def __init__(self, train_path, label_path, train=True):
		self.train_path = train_path
		self.label_path = label_path
		self.train = train
		self.img_files = listdir(join(os.getcwd(), self.train_path))
		# 50 means 25second/input
		self.seq_length = 50
		self.label_files = listdir(join(os.getcwd(), self.label_path))


	def __len__(self):
		if self.train:
			return len(self.img_files)
		else:
			return len(self.img_files)

	def __getitem__(self, idx):
		img_files_name = self.img_files[idx%len(self.img_files)]
		imgs = listdir(join(os.getcwd(), self.train_path, img_files_name))
		path = join(os.getcwd(), self.train_path, img_files_name)
		
		with open(join(os.getcwd(), self.label_path, img_files_name+'.txt')) as f:
			labels = f.read().split('\n')
			if not len(labels[-1]):
				labels = labels[:-1]
			f.close()
		
		# sample = sorted(np.random.choice(len(labels), self.seq_length, replace=False))

		if self.train :
			sample = np.random.randint(0, len(labels)-self.seq_length)
			sample = [sample + i for i in range(self.seq_length)]
			frames = torch.stack([transform(Image.open(path+'/'+imgs[num])) for num in sample])
			labels = torch.stack([torch.LongTensor([int(labels[num])]) for num in sample])
		else:
			sample = [i for i in range(len(imgs))]

			frames = torch.stack([transform(Image.open(path+'/'+imgs[num])) for num in sample])
			labels = torch.stack([torch.LongTensor([int(labels[num])]) for num in sample])
		# frames : [seq_length, 3, 299, 299]
		# labels : [seq_length, 1]

		return frames, labels

# if __name__ == '__main__':

# 	dataloader = DATASET('hw4_data/FullLengthVideos/videos/train', 'hw4_data/FullLengthVideos/labels/train')
# 	dataloader = DataLoader(dataloader, batch_size = 1,  shuffle=False)
# 	for step, batch in enumerate(dataloader):
# 	  if step == 10:
# 	      break
# 	  frame, label = batch

# 	  print(frame.size())
# 	  print(label.size())