import os
import torch
import pandas as pd
import numpy as np

from torchvision import transforms
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from torch.utils.data import Dataset
from typing import Union
from PIL import Image

class MultiLabelDataset(Dataset):
    
	def __init__(self, 
			  csv_file: Union[str, pd.DataFrame], 
			  root_dir: str,
			  vectorizer: Union[CountVectorizer, TfidfVectorizer],
			  transform: transforms = None,
			  use_caption_vec: bool = False,
			  caption_vec: np.ndarray = None,
			  is_test: bool = False) -> None:
		"""
		Arguments:
			csv_file (string): path to the csv file with annotations or pandas' dataframe.
			root_dif (string): directory with all the images.
			transform (callable, optional): Optional transform to be applied on a sample.
			vectorizer (optional): vectorize based on the given class.
			caption_vec (optional): if vectorize is not used, use this instead.
		"""
		self.df = csv_file
		if (type(csv_file) != pd.DataFrame):
			self.df = pd.read_csv(csv_file)
		self.root_dir = root_dir
		self.transform = transform
		self.vectorizer = vectorizer

		# either using vectorizer or caption vec
		self.use_caption_vec = use_caption_vec
		if (self.use_caption_vec):
			self.caption_vec = caption_vec

		self.is_test = is_test

	def set_transform(self, transform: transforms):
		self.transform = transform

	def __len__(self) -> int:
		return len(self.df)
	
	def __getitem__(self, idx):
		
		if (torch.is_tensor(idx)):
			idx = idx.tolist()

		# load the image
		img_name = os.path.join(self.root_dir, self.df.iloc[idx, 0])
		image = Image.open(img_name).convert("RGB")

		# load the caption
		caption = self.df.iloc[idx, 1]
		if (self.vectorizer):
			caption = self.vectorizer.transform([caption])
			caption = torch.from_numpy(caption.todense()).int()
		if (self.use_caption_vec): # not using vectorizer
			caption = self.caption_vec[idx]
			caption = torch.from_numpy(caption).int()

		# apply any transformations
		if (self.transform):
			image = self.transform(image)

		# get the labels
		labels = None
		if (not self.is_test):
			labels = torch.Tensor(self.df.iloc[idx, 2:].to_list())
			return img_name, image, caption, labels

		return img_name, image, caption