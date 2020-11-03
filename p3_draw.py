import os.path
import reader
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import load_inception_v3
import torch
import pandas as pd
import matplotlib.cm as cm
import model_p1
import matplotlib.patches as patches
import sys

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')

data = []
labels = []
colors = cm.rainbow(np.linspace(0, 1, 12))

if __name__ == '__main__':
	path = sys.argv[1:]
	p1, p2 = path
	# path to the pred.txt file or gt.txt file
	with open(p1) as f:
		labels = f.read().split('\n')
		if not len(labels[-1]):
			labels = labels[:-1]
		f.close()
	labels = labels[:500]
	x = [i+5 for i in range(500)]

	fig2 = plt.figure()
	ax2 = fig2.add_subplot(111, aspect='equal')
	plt.axis([0, 500, 0, 50])
	for index, (x, label) in enumerate(zip(x, labels)):

		ax2.add_patch(
			patches.Rectangle(
				(x, 0.1),
				5, 
				50,
				fill = True,
				color = colors[int(labels[index])]
				)
			)

	fig2.savefig(os.path.join(p2, 'part2.jpg'), dpi=90)