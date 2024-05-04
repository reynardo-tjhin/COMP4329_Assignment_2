import torch
import torch.nn as nn
import argparse

from torch.utils.data import DataLoader
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, ResNet18_Weights
from dataset import MultiLabelDataset
from tqdm import tqdm
from tools import get_data, load_data, remove_class

# global variables/constant
N_CLASSES = 19
IMG_SIZE = 256 # can change depending on model (maybe to parse args?)

# main
def main(args: argparse.Namespace):

	# for logging
	log = open(args.log_file, "w")

	# import data
	train_data = get_data("./dataset/train.csv")
	test_data = get_data("./dataset/test.csv")

	# perform text cleaning and get the pandas' dataframe
	train_data = load_data(train_data)
	test_data = load_data(test_data, has_label=False)

	# remove an imbalanced class
	train_data = remove_class(train_data, class_no=1)

	print(f"Number of training instances: {train_data.shape[0]}")
	print(f"Number of testing instances:  {test_data.shape[0]}")

	# define the image transformation: currently following resnet18
	transform = transforms.Compose([
		transforms.Resize((256, 256)),
		transforms.CenterCrop(224),
		transforms.ToTensor(), # converts images to [0, 1]
		transforms.Normalize(
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225],
		)
	])

	# perform pre-processing and initialize the dataset
	train_dataset = MultiLabelDataset(
		csv_file=train_data,
		root_dir='./dataset/data/',
		transform=transform,
	)
	test_dataset = MultiLabelDataset(
		csv_file=test_data,
		root_dir='./dataset/data/',
		transform=transform,
	)

	# load the dataset into batches 
	batch_size = args.batch_size
	train_dataloader = DataLoader(
		dataset=train_dataset,
		batch_size=batch_size,
		shuffle=True,
	)
	test_dataloader = DataLoader(
		dataset=test_dataset,
		batch_size=batch_size,
		shuffle=False,
	)

	# create the model
	if (args.model == 'resnet50'):
		model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
	elif (args.model == 'resnet18'):
		model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

	# TODO: CHANGE THE CLASSIFIER LAYER TO IMPROVE MODEL!
	# Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
	for name, params in model.named_parameters():
		if ("fc" not in name):
			params.requires_grad = False
	n_features = model.fc.in_features
	n_out = N_CLASSES
	model.fc = nn.Sequential(
		nn.Linear(in_features=n_features, out_features=n_out),
		# nn.Sigmoid(), no need for sigmoid as BCE with logits loss already uses Sigmoid!
	)

	# utilise GPU
	if torch.cuda.is_available():
		print('using GPU')
		model = model.to('cuda')

	# loss func and optimizer
	loss_fn = nn.BCEWithLogitsLoss()
	optimizer = torch.optim.Adam(
		params=model.parameters(),
		lr=args.learning_rate,
	)

	log.write("epoch,train loss,train acc,test acc\n")

	# the minimum probability of assigning one class
	threshold = args.threshold
	n_epochs = args.epochs
	train_losses = []
	train_accs = []
	for epoch in range(n_epochs):

		i = 0
		# train the model
		n_total = 0
		n_correct = 0
		train_loss = 0.
		model.train()
		for (images, labels) in tqdm(train_dataloader, desc=f"Epoch {epoch+1} Training: "):

			if torch.cuda.is_available():
				images = images.to('cuda')
				labels = labels.to('cuda')

			# set gradient to zero
			optimizer.zero_grad()
			
			# forward
			outputs = model(images)

			# backward
			loss = loss_fn(outputs, labels)
			loss.backward()

			# update
			optimizer.step()

			# compare
			predicted = (outputs > threshold).int()

			# if (n_total == 0): print(labels, predicted)

			train_loss += loss.item()
			n_correct += torch.all(torch.eq(predicted, labels), dim=1).sum()
			n_total += labels.shape[0]

		train_losses.append(train_loss / len(train_dataloader))
		train_accs.append(n_correct / n_total)

		print("Epoch {:d}, Train Loss: {:.7f}, Train Accuracy: {:.3f}%".format(epoch+1, train_losses[-1], train_accs[-1]*100))
		log.write("{:d},{:.7f},{:.5f}\n".format(epoch+1, train_losses[-1], train_accs[-1]))

	# close log file
	log.close()

	# create a file for test submission
	# f = open('submission ' + args.log_file, "w")
	# f.write("ImageID,Labels\n")
	# for (images, labels) in test_dataloader:
	# 	if torch.cuda.is_available():
	# 		images = images.to('cuda')
	# 		labels = labels.to('cuda')
	# 	outputs = model(images)
	# 	predicted = (torch.sigmoid(outputs).data > threshold).int()

	# 	# NOTE: add 1 to the output of predicted!
	# 	# write the output
	# 	for predicted_label in predicted:
	# 		label = (predicted_label == torch.max(predicted_label)).nonzero().flatten()
	# 		label += 1
	# 		label = label.tolist()
	# 		label = " ".join(str(x + 1) for x in label)
	# 	f.write("")
	# f.close()


if (__name__ == "__main__"):

	parser = argparse.ArgumentParser()
	parser.add_argument('-lr', '--learning_rate',
					 default=0.01,
					 type=float)
	parser.add_argument('-b', '--batch_size',
					 default=16,
					 type=int)
	parser.add_argument('-m', '--model',
					 choices=['resnet50', 'resnet18'],
					 default='resnet50',
					 type=str)
	parser.add_argument('-e', '--epochs',
					 default=20,
					 type=int)
	parser.add_argument('-t', '--threshold',
					 default=0.5,
					 type=float)
	parser.add_argument('-lf', '--log_file',
					 default='log',
					 type=str)
	args = parser.parse_args()

	main(args)