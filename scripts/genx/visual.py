import h5py
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm, trange
import cv2
import torch
import sparseconvnet as scn

def visual(h5_path, i):
	i = int(i)
	with h5py.File(h5_path, 'r') as f:
		f_idx, start, end = f['indices'][i]
		xy = f['pos'][start:end]
		time = f['time'][start:end]
		p = f['events'][start:end]

	xy = torch.from_numpy(xy).long()
	p = (torch.from_numpy(p).float() - 0.5) * 2

	xy, p = xy.cuda(), p.cuda()

	height, width = 240, 304
	sparse_to_dense = scn.Sequential(
		scn.InputLayer(2, (height, width), mode=1),
		scn.SparseToDense(2, 4)
	)

	y, x = xy[:, 1], xy[:, 0]
	yxb = torch.stack((y, x, torch.zeros_like(x)), dim=1)
	p = p[:, None]

	dense = sparse_to_dense((yxb, p))

	img = dense[0, 0].cpu().numpy()
	img = (img + 1) * 255 / 2
	img = img.astype(np.uint8)
	name = h5_path.split('/')[1]+f'{i}.png'
	cv2.imwrite(name, img)

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
                    prog='ProgramName',
                    description='What the program does',
                    epilog='Text at the bottom of help')
	parser.add_argument('-f', '--filename')           # positional argument
	parser.add_argument('-i', '--index')           # positional argument
	args = parser.parse_args()
	visual(args.filename, args.index)
