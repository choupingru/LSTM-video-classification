import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision
import torch
import torchvision.models as models

__all__ = ['resnet50']

model_url = {'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth'}
# model = torchvision.models.resnet50(pretrained=False)
# print(model)


class Identity(nn.Module):
	def __init__(self, mode='p3'):
		super(Identity, self).__init__()
		# self.fc = nn.Sequential(
		# 	nn.BatchNorm1d(2048)
		# 	)
		if mode == 'p2':
			self.bn = nn.BatchNorm1d(2048, momentum=0.01)

	def forward(self, x):
		# x = self.fc(x)
		return x

def extractor(pretrained=True, mode='p3'):

	#model = resnet50(pretrained=pretrained)
		# model.avgpool = Identity()
	model = models.inception_v3(pretrained=pretrained)
	#model.load_state_dict(torch.load('./inception_v3.pth'))
	#print(model)
	if mode == 'p2':
		model.fc = Identity('p2')
	else:
		model.fc = Identity('p3')
	return model
