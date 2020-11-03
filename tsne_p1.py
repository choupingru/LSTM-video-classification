import reader
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import load_inception_v3
import torch
import pandas as pd
import matplotlib.cm as cm
import model_p1
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

extractor = load_inception_v3.extractor(pretrained=True).to(device)
model = model_p1.SimpleClassifier(False).to(device)
model.load_state_dict(torch.load('./p1/model_p1_v1_15_30.pth'))
model.eval()

videos_list = reader.getVideoList('./hw4_data/TrimmedVideos/label/gt_valid.csv')
data = []
labels = []
colors = cm.rainbow(np.linspace(0, 1, 12))


for i in range(len(videos_list['Video_category'])):
	if i % 10 == 0:
		print("[%d]/[%d]" % (i, len(videos_list['Video_category'])))
		
	#frame = reader.readShortVideo('./hw4_data/TrimmedVideos/video/valid', videos_list['Video_category'][i], videos_list['Video_name'][i])	
	#frame = frame.transpose((0, 3, 1, 2))
	#frame = torch.FloatTensor(frame).to(device)

	label = videos_list['Action_labels'][i]
	label = int(label)
#	with torch.no_grad():
#		frame = extractor(frame)
#	frame = frame[0]
#	frame = torch.mean(frame, dim=0)
#	frame = frame.unsqueeze(0)
#	feature = model(frame)
#
#	data.append(frame.cpu().detach().numpy())
	labels.append(label)

#data = np.array(data)
#np.save('./tsne_data_v2.npy', data)
data = np.load('./p1/tsne_data_v2.npy')
data = data.reshape(-1, 2048)
data_tsne = TSNE(n_components=2).fit_transform(data)
x1, x2, y1, y2 = plt.axis()
plt.axis((-11, 11, -11, 11))
for index, ele in enumerate(data_tsne):
	x, y = ele
	plt.text(x, y, s=str(labels[index]), color = colors[labels[index]], fontsize=7)
plt.savefig('./tsne_p1_v2.jpg')
plt.show()
