import torch 
import numpy as np
import matplotlib.pyplot  as plt
import torch.nn as nn
import torch.optim as optim
from os.path import join
import os 
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch.autograd import Variable
from load_data_p3 import DATASET
from sklearn.metrics import confusion_matrix
import sys
import load_inception_v3 
import model_p2_batch

class lstm_model(nn.Module):

	def __init__(self, input_size, hidden_size, class_num, bidirectional=False):
		super(lstm_model, self).__init__()

		self.input_size = input_size
		self.hidden_size = hidden_size
		self.class_num = class_num
		if bidirectional:
			self.bidirectional = 2
		else:
			self.bidirectional = 1
		
		# CNN 
		self.encoder = load_inception_v3.extractor(pretrained=True)

		# LSTM
		self.lstm = nn.LSTM(self.input_size, self.hidden_size, num_layers = 2, batch_first=True, dropout=0.2, bidirectional=bidirectional)
 

		# CLF
		self.fc = nn.Sequential(
			nn.Linear(self.hidden_size*self.bidirectional , self.hidden_size),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(self.hidden_size, 11),
		)
		
	def forward(self, feature):
		# featrue : [total_num, 3, 224, 224]
		# length_list : [number of image of each data]
		
		#feature = feature[0]
		with torch.no_grad():
			self.encoder.eval()
			feature = self.encoder(feature)
		
		feature = feature.unsqueeze(0)
		lstm_out, _ = self.lstm(feature)
		
		out = lstm_out.contiguous()
		out = out.view(lstm_out.size(1) , self.hidden_size * self.bidirectional)
		out = self.fc(out)
		out = out.view(lstm_out.size(0), lstm_out.size(1), 11)

		out = out[0]
		
		return out

BATCH_SIZE = 1
cuda = torch.cuda.is_available()

device = torch.device('cuda' if cuda else 'cpu')
print(device)

def main():
	path = sys.argv[1:]
	#p1, p2, p3 = path
	train = False
	if train:

		clf = lstm_model(2048, 2048, 11, bidirectional=False)
		
		clf = clf.to(device)
		# clf.load_state_dict(torch.load('./p2/model_p2_v1.pth'))
		loss_fn = nn.CrossEntropyLoss()
		# weights= torch.FloatTensor([1, 1, 1, 0.8, 1, 0.8, 1, 1, 1, 1, 1]).to(device)
		# loss_fn = nn.CrossEntropyLoss(weight=weights)
		plot_xy = []
		plot_ac = []

		
		dataloader = DATASET('./hw4_data/FullLengthVideos/videos/train', './hw4_data/FullLengthVideos/labels/train')
		dataloader = DataLoader(dataloader, batch_size = BATCH_SIZE, shuffle=True)
		eval_dataloader = DATASET('./hw4_data/FullLengthVideos/videos/valid', './hw4_data/FullLengthVideos/labels/valid')
		eval_dataloader = DataLoader(eval_dataloader, batch_size = BATCH_SIZE)
		
		optimizer = optim.Adam(clf.parameters(), lr=0.0001)
		for ep in range(1):
			print(ep)
			
			############## train model ################
			clf.train()
			for step, batch in enumerate(dataloader):
				
				feature, label = batch
				label = label.view(-1).to(device)
				
				# feature : [total_num, 3, 224, 224]
				feature = feature.to(device)

				# pred and loss


				pred = clf(feature)
				loss = loss_fn(pred, label)


				if step % 50 == 0:
					print('[%d]/[%d]' % (step, len(dataloader)))
					print(np.argmax(pred.cpu().detach().numpy(), axis=1))
					print(loss.item())

				# backprog
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

			############## eval model ################
			clf.eval()
			total_loss, ac, total_num = 0, 0, 0
			my_pred, my_label = [], []

			for step, batch in enumerate(eval_dataloader):

				feature, label = batch
				feature = feature.to(device)
				label = label.view(-1).to(device)
				
				total_num += 1
				
				with torch.no_grad():
					pred = clf(feature)

				loss = loss_fn(pred, label)
			
				total_loss += loss.item()

				my_pred.append(np.argmax(pred.cpu().detach().numpy(), axis=1).reshape(-1))
				my_label.append(label.cpu().detach().numpy())
				
				if step % 50 == 0:
					print('[%d]/[%d]' % (step, len(eval_dataloader)))
					print(np.argmax(pred.cpu().detach().numpy(),axis=1))
				ac += np.sum(np.argmax(pred.cpu().detach().numpy(), axis=1) == label.cpu().detach().numpy())

			############## print ac & loss & save model ################
			print(ac)
			print(total_num)
			my_pred = np.concatenate([ele for ele in my_pred])
			my_label = np.concatenate([ele for ele in my_label])
			print(confusion_matrix(my_pred, my_label))
			plot_xy.append([ep, total_loss])
			plot_ac.append([ep, ac/len(eval_dataloader) / BATCH_SIZE])
			print('Eval Loss : [%.4f] ' % (total_loss))
			print('Accuracy : [%.4f] ' % (ac / total_num))
			np.save('./p2_loss_v1_2.npy', np.array(plot_xy))
			np.save('./p2_ac_v1_2.npy', np.array(plot_ac))
			torch.save(clf.state_dict(), './p2/model_p2_v1_2.pth')

		plot_xy = np.array(plot_xy)
		plt.plot(plot_xy[:, 0], plot_xy[:, 1])
		plt.show()

		plot_ac = np.array(plot_ac)
		plt.plot(plot_ac[:, 0], plot_ac[:, 1])
		plt.show()
	else:
		clf = lstm_model(2048, 2048, 11, bidirectional=False).to(device)
		clf.eval()
		clf.load_state_dict(torch.load('./p3/model_p2_v1_16_57.pth'))
		dataloader = DATASET('./hw4_data/FullLengthVideos/videos/valid', './hw4_data/FullLengthVideos/labels/valid', train=False)
		dataloader = DataLoader(dataloader, batch_size = BATCH_SIZE)
		

		total_loss = 0
		ac = 0
		
		with torch.no_grad():

			for step, batch in enumerate(dataloader):
				
				all_output = []

				frame, label = batch
				frame = frame.to(device)
				label = label.to(device)
				label = label.view(-1)
			
				pred = clf(frame)
				all_output.append(pred.cpu().detach().numpy().flatten())
		
				all_output = np.array([out for out in all_output]).flatten()
				np.savetxt(os.path.join(os.getcwd(), 'p3', 'output'+str(step)+'.txt'), all_output, fmt='%d')


if __name__ == '__main__':

	main()





