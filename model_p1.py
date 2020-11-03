import torch.nn as nn
import math
import torch
import torch.utils.model_zoo as model_zoo
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from os.path import join
import os
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.autograd import Variable
from load_data_p1 import DATASET, my_collate_fn
import load_inception_v3
import sys
from sklearn.metrics import confusion_matrix

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
BATCH_SIZE = 1

class SimpleClassifier(nn.Module):

	def __init__(self, pretrain=True):
		super(SimpleClassifier, self).__init__()

		
		self.fc = nn.Sequential(
			nn.Linear(2048, 2048),
			nn.BatchNorm1d(2048),
			nn.ReLU(True),
		)
		self.cls = nn.Linear(2048, 11)

	def forward(self, x):
		
		feature = self.fc(x)
		
		out = self.cls(feature)

		return out, feature


if __name__ == "__main__":
	
	train = False
	if train:
		clf = SimpleClassifier().to(device)
		#clf.load_state_dict(torch.load('./p1/model_p1_v1_8_30.pth'))
		loss_fn = nn.CrossEntropyLoss()
		optimizer = optim.Adam(clf.parameters(), lr=1e-4, weight_decay=0.9)

		dataloader = DATASET('./hw4_data/TrimmedVideos/video/train', './hw4_data/TrimmedVideos/label/gt_train.csv', train=True)
		dataloader = DataLoader(dataloader, batch_size = BATCH_SIZE, shuffle=True)

		eval_dataloader = DATASET('./hw4_data/TrimmedVideos/video/valid', './hw4_data/TrimmedVideos/label/gt_valid.csv', train=True)
		eval_dataloader = DataLoader(eval_dataloader, batch_size = BATCH_SIZE, shuffle=True)
		
		plot_xy = []
		plot_ac = []
		for ep in range(40):
			print(ep)

			clf.train()
			for step, batch in enumerate(dataloader):
				optimizer.zero_grad()

				if step % 5 == 0:
					print('[%d]/[%d]' % (step, len(dataloader)))
				
				frame, label = batch
				frame = frame.to(device)
				label = label.to(device)
				label = label.view(-1)

				pred, _ = clf(frame)
				loss = loss_fn(pred, label)
				
				loss.backward()
				optimizer.step()

			clf.eval()
			total_loss = 0
			ac = 0
			my_pred, my_label = [], []
			with torch.no_grad():
				for step, batch in enumerate(eval_dataloader):
					
					frame, label = batch
					frame = frame.to(device)
					label = label.to(device)
					label = label.view(-1)
				
					pred, _ = clf(frame)
					loss = loss_fn(pred, label)
				
					total_loss += loss.item()
					my_pred.append(np.argmax(pred.cpu().detach().numpy(), axis=1).reshape(-1))
					my_label.append(label.cpu().detach().numpy())
					
					ac += np.sum(np.argmax(pred.cpu().detach().numpy(), axis=1) == label.cpu().detach().numpy())
			plot_xy.append([ep, total_loss])
			plot_ac.append([ep, ac/len(eval_dataloader) / BATCH_SIZE])
			my_pred = np.concatenate([ele for ele in my_pred])
			my_label = np.concatenate([ele for ele in my_label])
			print(confusion_matrix(my_pred, my_label))

			print('Eval Loss : [%.4f] ' % (total_loss / len(eval_dataloader)))
			print('Accuracy : [%.4f] ' % (ac / len(eval_dataloader) / BATCH_SIZE))
			torch.save(clf.state_dict(), './p1/model_p1_v1_'+str(ep)+'_'+str(int(ac/len(eval_dataloader)/BATCH_SIZE*100))+'.pth')
			np.save('./p1/p1_loss_v1_22.npy', np.array(plot_xy))
			np.save('./p1/p1_ac_v1_22.npy', np.array(plot_ac))

		plot_xy = np.array(plot_xy)
		plt.plot(plot_xy[:, 0], plot_xy[:, 1])
		plt.show()

		plot_ac = np.array(plot_ac)
		plt.plot(plot_ac[:, 0], plot_ac[:, 1])
		plt.show()

	else:
		args = sys.argv[1:]
		p1, p2, p3 = args

		clf = SimpleClassifier(False).to(device)	
		clf.load_state_dict(torch.load('./p1/model_p1_v1_15_30.pth'))
		clf.eval()
		
		dataloader = DATASET(p1, p2, train=True)
		dataloader = DataLoader(dataloader, batch_size = BATCH_SIZE, shuffle=True)
		
		total_loss = 0
		ac = 0
		all_output = []
		with torch.no_grad():
			for step, batch in enumerate(dataloader):
				if step % 10 == 0:
					print('[%d]/[%d]' % (step, len(dataloader)))
				frame, label = batch
				frame = frame.to(device)
				label = label.to(device)
				label = label.view(-1)
			
				pred = clf(frame)
				pred = pred[0]
				all_output.append(np.argmax(pred.cpu().detach().numpy().flatten()))
		
		all_output = np.array([out for out in all_output]).flatten()
		np.savetxt(os.path.join(os.getcwd(), p3, 'p1_valid.txt'), all_output, fmt='%d')
