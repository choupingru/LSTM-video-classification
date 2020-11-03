import reader
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import sys






if __name__ == '__main__':
	p = sys.argv[1:][0]
	
	if p == 'p1':
		ac = np.load('./p1/p1_ac_v1_2.npy')
		ac = ac[:6]
		ac = np.concatenate([ac, np.load('./p1/p1_ac_v1_2_2.npy')], axis=0)
		ac = np.concatenate([ac, np.load('./p1/p1_ac_v1_22.npy')], axis=0)

		loss = np.load('./p1/p1_loss_v1_2.npy')
		loss = loss[:6]
		loss = np.concatenate([loss, np.load('./p1/p1_loss_v1_2_2.npy')], axis=0)
		loss = np.concatenate([loss, np.load('./p1/p1_loss_v1_22.npy')], axis=0)
		np.save('./p1/p1_ac.npy', ac)
		np.save('./p1/p1_loss.npy', loss)
		
		ep = [i for i in range(len(ac)*2)]
		ac = [[a[1], (a[1]+ac[abs(idx-1)][1])/2] for idx, a in enumerate(ac)]
		ac = np.array(ac).flatten()
		
		loss = [[a[1], (a[1]+loss[abs(idx-1)][1])/2] for idx, a in enumerate(loss)]
		loss = np.array(loss).flatten()
		
		

		plt.plot(ep, ac)
		plt.show()

		plt.plot(ep, loss)
		plt.show()

	if p == 'p2':
		ac = np.load('./p2/p2_ac_v2_2.npy')
		
		loss = np.load('./p2/p2_loss_v2_2.npy')
		
		plt.plot(ac[:, 0], ac[:, 1])
		plt.show()

		plt.plot(loss[:, 0], loss[:, 1])
		plt.show()
