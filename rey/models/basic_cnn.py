import torch.nn as nn
import torch.nn.functional as F

class BasicCNN(nn.Module):
	
	def __init__(self, num_classes=10):
		super(BasicCNN, self).__init__()
		
		# Convolutional layers
		self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
		self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

		# Pooling layer
		self.pool = nn.MaxPool2d(2, 2)  # Kernel size 2 and stride 2

		# Linear layers
		self.fc1 = nn.Linear(128 * 4 * 4, 512)  # Adjust size according to the output of the last conv layer
		self.fc2 = nn.Linear(512, num_classes)

	def forward(self, x):
		
		# Apply convolutional layers with ReLU and pooling
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = self.pool(F.relu(self.conv3(x)))

		# Flatten the layer before passing to the fully connected layers
		x = x.view(-1, 128 * 4 * 4)  # Flatten all dimensions except batch

		# Fully connected layers with ReLU activation and final layer
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return x
