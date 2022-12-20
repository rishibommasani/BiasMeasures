# Draws from John Hewitt's probing work
import torch.nn as nn
import torch
import numpy
from tqdm import tqdm


class LinearProbe(nn.Module):
	def __init__(self, model_dim, label_space_size):
		super(LinearProbe, self).__init__()
		self.model_dim = model_dim
		self.label_space_size = label_space_size
		self.linear = nn.Linear(self.model_dim, self.label_space_size)
		self.loss = torch.nn.CrossEntropyLoss()
		# self.linear2 = nn.Linear(self.label_space_size, self.label_space_size)
		# self.print_param_count()
		# dropout = .0
		# self.dropout = nn.Dropout(p=dropout)
		# print('Applying dropout {}'.format(dropout))
		# self.zero_features = zero_features
		# self.to(args['device'])


	def forward(self, representation, gold_label, target_concept, train=True):
		logits = self.linear(representation)
		reshaped_logits = logits.unsqueeze(0)
		predicted_label = torch.argmax(logits).item()
		gold_label_tensor = torch.tensor([gold_label])
		example_loss = self.loss(reshaped_logits, gold_label_tensor)

		if train:
			return example_loss, predicted_label, logits.detach().numpy() 
		else:
			return example_loss.detach().numpy(), predicted_label