import os
import cv2
import torch
import pandas as pd
import numpy as np

from torch.utils.data import Dataset
from typing import Union
from PIL import Image

class MultiLabelDataset(Dataset):
    
	def __init__(self, csv_file: Union[str, pd.DataFrame], root_dir: str, transform=None) -> None:
		"""
		Arguments:
			csv_file (string): path to the csv file with annotations or pandas' dataframe.
			root_dif (string): directory with all the images.
			transform (callable, optional): Optional transform to be applied on a sample.
		"""
		self.df = csv_file
		if (type(csv_file) != pd.DataFrame):
			self.df = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform

	def __len__(self) -> int:
		return len(self.df)
	
	def __getitem__(self, idx):
		
		if (torch.is_tensor(idx)):
			idx = idx.tolist()
		
		# load the image
		img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
		image = Image.open(img_name).convert("RGB")
		# image = np.asarray(image)
		# image = cv2.imread(img_name, flags=1)

		# get the labels
		labels = torch.Tensor(self.df.iloc[idx, 2:])

		# apply any transformations
		if (self.transform):
			image = self.transform(image)

		return image, labels