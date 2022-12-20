import torch
import random
import torch.nn as nn
from torch import optim
import numpy
from tqdm import tqdm
from probe import LinearProbe


def partition(examples, target2counts, split_ratio = 0.7):
	train, test = [], []
	target2trainsize = {}
	for target, v in target2counts.items():
		train_size = 0
		for count in v.values():
			train_size += count
		target2trainsize[target] = int(split_ratio * train_size)
	for example in examples:
		target = example['target_concept'] 
		if target2trainsize[target] > 0:
			train.append(example)
			target2trainsize[target] = target2trainsize[target] - 1
		else:
			test.append(example)
	return train, test


def predict(model, test, num_classes):
	target2counts = {}
	total_loss = None
	correct, predictions = 0, 0
	for example in test:
		representation = example['representation']
		gold_label_id = example['gold_label_id']
		target_concept = example['target_concept']	
		example_loss, predicted_label_id = model(representation, gold_label_id, target_concept, train=False)
		if total_loss is None:
			total_loss = example_loss
		else:
			total_loss += example_loss
		if gold_label_id == predicted_label_id:
			correct += 1
		predictions += 1
		if target_concept in target2counts:
			target2counts[target_concept]['gold'][gold_label_id] = target2counts[target_concept]['gold'].get(gold_label_id, 0) +  1
			target2counts[target_concept]['predicted'][predicted_label_id] = target2counts[target_concept]['predicted'].get(predicted_label_id, 0) +  1
		else:
			target2counts[target_concept] = {'gold' : {label_id : 0 for label_id in range(num_classes)}, 'predicted': {label_id : 0 for label_id in range(num_classes)}}
			target2counts[target_concept]['gold'][gold_label_id] = target2counts[target_concept]['gold'].get(gold_label_id, 0) +  1
			target2counts[target_concept]['predicted'][predicted_label_id] = target2counts[target_concept]['predicted'].get(predicted_label_id, 0) +  1
		predictions += 1
	target2results = {}
	for target in target2counts:
		gold_scores = target2counts[target]['gold']
		gold_Z = sum(gold_scores.values())
		gold_ratios = {label : round(score / gold_Z, 4) for label, score in gold_scores.items()}
		predicted_scores = target2counts[target]['predicted']
		predicted_Z = sum(predicted_scores.values())
		predicted_ratios = {label : round(score / predicted_Z, 4) for label, score in predicted_scores.items()}
		target2results[target] = {'gold' : gold_ratios, 'predicted' : predicted_ratios}
	accuracy = correct / predictions
	return target2counts, target2results, accuracy, total_loss
	

def train_probe(examples, target2counts, model_dim=768, label_space_size=3, epochs=10, batch_size=10, split_ratio=0.7):
	train, test = partition(examples, target2counts, split_ratio=split_ratio)
	n_train, n_test = len(train), len(test)
	print(n_train, n_test)
	model = LinearProbe(model_dim=model_dim, label_space_size=label_space_size)
	optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0)
	
	eval_index = 0
	min_dev_loss_eval_index = -1
	eval_dev_losses = []
	for epoch_index in tqdm(range(1, epochs + 1), desc='[training]'):
		random.shuffle(train)
		epoch_train_loss = None
		for batch_index in range(n_train // batch_size):
		# Take a train step
			model.train()
			optimizer.zero_grad()
			batch = train[batch_index * batch_size : batch_index * batch_size + batch_size]
			batch_loss = None
			for example in batch:
				representation = example['representation']
				gold_label = example['gold_label_id']
				target_concept = example['target_concept']
				example_loss, predicted_label, logits = model(representation, gold_label, target_concept, train=True)
				if batch_loss is None:
					batch_loss = example_loss
				else:
					batch_loss += example_loss
			batch_loss.backward()
			if epoch_train_loss is None:
				epoch_train_loss = batch_loss.detach().cpu().numpy()
			else:
				epoch_train_loss += batch_loss.detach().cpu().numpy()
			optimizer.step()
		target2counts, target2results, test_accuracy, test_loss = predict(model, test, label_space_size)
		print('Epoch {}: Train Loss: {}, Test Loss: {}, Test Accuracy: {}'.format(epoch_index, epoch_train_loss, test_loss, test_accuracy))
		for target in target2results:
			print('Target {}: {}'.format(target, target2results[target]))
	return target2counts, target2results
