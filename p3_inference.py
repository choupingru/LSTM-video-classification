import numpy as np
import model_p3
import os.path
import os
from os import listdir
from os.path import join
import sys
import torch
from PIL import Image
from torchvision import transforms

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

mean = torch.FloatTensor([0.5, 0.5, 0.5])
std = torch.FloatTensor([0.5, 0.5, 0.5])

transform = transforms.Compose([
	transforms.Resize((299, 299)),
	transforms.ToTensor(),
	transforms.Normalize(mean, std)
	])

if __name__ == '__main__':
	path = sys.argv[1:]
	p1, p2 = path

	clf = model_p3.lstm_model(2048, 2048, 11, bidirectional=False).to(device)
	clf.eval()
	clf.load_state_dict(torch.load('./p3/model_p2_v1_16_57.pth'))
	for file in listdir(p1):
		pred = []
		start = 0
		all_images = listdir(join(p1, file))
		prediction = []
		while start < len(all_images):

			if start == 0:
				imgs = all_images[start: start+50]
			else:
				imgs = all_images[start-5: start+50]
				
			imgs_tensor = torch.stack([transform(Image.open(join(p1, file, img_name))) for img_name in imgs]).to(device)
			with torch.no_grad():
				pred = clf(imgs_tensor)
			
			pred = torch.max(pred, dim=1)[1]
			if start ==0:
				prediction += list(pred.cpu().detach().numpy())
			else:
				prediction += list(pred[5:].cpu().detach().numpy())
			start += 50

		np.savetxt(join(p2, file+'.txt'), np.array(prediction), fmt='%d')



