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
from load_data_p2 import DATASET, my_collate_fn
from sklearn.metrics import confusion_matrix
import sys
import load_inception_v3 
import torch.nn.functional as F

BATCH_SIZE = 1
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print(device)
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
		self.encoder = load_inception_v3.extractor(pretrained=True, mode='p2')

		# LSTM
		self.lstm = nn.LSTM(self.input_size, self.hidden_size, batch_first=True, num_layers=2, dropout=0.5, bidirectional=bidirectional)
 

		# CLF
		self.fc = nn.Sequential(
			nn.Linear(self.hidden_size*self.bidirectional , self.hidden_size * self.bidirectional),
			nn.ReLU(True),
			nn.Dropout(0.5),
			nn.Linear(self.hidden_size * self.bidirectional, 11)
		)
		
	def forward(self, feature):

		#h0 = torch.randn(2, 1, self.hidden_size).to(device)
		#c0 = torch.randn(2, 1, self.hidden_size).to(device)
		# featrue : [total_num, 3, 224, 224]
		# length_list : [number of image of each data]
		#feature = feature[0]
		with torch.no_grad():
			self.encoder.eval()
			feature = self.encoder(feature)
		
		feature = feature.unsqueeze(0)
		lstm_out, _ = self.lstm(feature)
		lstm_out = lstm_out[0]

		out = self.fc(lstm_out)

		out = out[-1]
		#out = lstm_out.contiguous()
		#out = out.view(lstm_out.size(1) , self.hidden_size * self.bidirectional)
		#out = self.fc(out)
		#out = out.view(lstm_out.size(0), lstm_out.size(1), 11)
		#out = out[:, -1, :]
		out = out.unsqueeze(0)
		return out, lstm_out[-1]


def main():
	path = sys.argv[1:]
	p1, p2, p3 = path
	train = False
	if train:

		clf = lstm_model(2048, 2048, 11, bidirectional=False)
		#print(clf)
		clf = clf.to(device)
		#clf.load_state_dict(torch.load('./p2_2/model_p2_v2_26.pth'))
		loss_fn = nn.CrossEntropyLoss()
		#weights= torch.FloatTensor([0.8, 1, 1, 0.5, 0.8, 0.5, 1, 1, 1, 1, 1]).to(device)
		#loss_fn = nn.CrossEntropyLoss(weight=weights)
		plot_xy = []
		plot_ac = []

		
		dataloader = DATASET('./hw4_data/TrimmedVideos/video/train', './hw4_data/TrimmedVideos/label/gt_train.csv', train=False)
		dataloader = DataLoader(dataloader, batch_size = BATCH_SIZE, shuffle=True)
		eval_dataloader = DATASET('./hw4_data/TrimmedVideos/video/valid', './hw4_data/TrimmedVideos/label/gt_valid.csv', train=False)
		eval_dataloader = DataLoader(eval_dataloader, batch_size = BATCH_SIZE)
		
		for ep in range(100):
			print(ep)
			train_loss = 0
			############## train model ################
			clf.train()
			preds = []
			optimizer = optim.Adam(clf.parameters(), lr=1e-4)
			for step, batch in enumerate(dataloader):
				
				feature, label = batch
				label = label.view(-1).to(device)
				
				# feature : [total_num, 3, 224, 224]
				feature = feature.to(device)

				# pred and loss
				pred, _ = clf(feature)
				loss = loss_fn(pred, label)
				train_loss += loss.item()
				preds += list(np.argmax(pred.cpu().detach().numpy(),axis=1))
				if step % 50 == 0:
					print('[%d]/[%d]' % (step, len(dataloader)))
					print(train_loss/len(dataloader))
					print(preds)
					preds = []
				# backprog
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			print('train avg loss : ', train_loss/len(dataloader))
			############## eval model ################
			clf.eval()
			total_loss, ac, total_num = 0, 0, 0
			my_pred, my_label = [], []

			with torch.no_grad():
				for step, batch in enumerate(eval_dataloader):

					feature, label = batch
					feature = feature.to(device)
					label = label.view(-1).to(device)
					
					total_num += 1
					
					pred, _ = clf(feature)

					loss = loss_fn(pred, label)
				
					total_loss += loss.item()

					my_pred.append(np.argmax(pred.cpu().detach().numpy(), axis=1).reshape(-1))
					my_label.append(label.cpu().detach().numpy())
					
					if step % 200 == 0:
						print('[%d]/[%d]' % (step, len(eval_dataloader)))
					ac += np.sum(np.argmax(pred.cpu().detach().numpy(), axis=1) == label.cpu().detach().numpy())

			############## print ac & loss & save model ################
			print(ac)
			print(total_num)
			print('eval avg loss : ', total_loss/769)
			my_pred = np.concatenate([ele for ele in my_pred])
			my_label = np.concatenate([ele for ele in my_label])
			print(confusion_matrix(my_pred, my_label))
			plot_xy.append([ep, total_loss/769])
			plot_ac.append([ep, ac/len(eval_dataloader) / BATCH_SIZE])
			print('Eval Loss : [%.4f] ' % (total_loss))
			print('Accuracy : [%.4f] ' % (ac / total_num))
			np.save('./p2_2/p2_loss_v2_2.npy', np.array(plot_xy))
			np.save('./p2_2/p2_ac_v2_2.npy', np.array(plot_ac))
			torch.save(clf.state_dict(), './p2_2/model_p2_v2_'+str(ep)+'.pth')

		plot_xy = np.array(plot_xy)
		plt.plot(plot_xy[:, 0], plot_xy[:, 1])
		plt.show()

		plot_ac = np.array(plot_ac)
		plt.plot(plot_ac[:, 0], plot_ac[:, 1])
		plt.show()
	else:
		clf = lstm_model(2048, 2048, 11, bidirectional=True).to(device)
		clf.eval()
		clf.load_state_dict(torch.load('./p2/model_p2_v2_19.pth'))
		dataloader = DATASET(p1, p2, train=True)
		dataloader = DataLoader(dataloader, batch_size = BATCH_SIZE)
		
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
				frame = frame[0]
				pred = clf(frame)
				pred = pred[0]
				all_output.append(np.argmax(pred.cpu().detach().numpy().flatten()))
		
		all_output = np.array([out for out in all_output]).flatten()
		np.savetxt(os.path.join(os.getcwd(), p3, 'p2_result.txt'), all_output, fmt='%d')


if __name__ == '__main__':

	main()





