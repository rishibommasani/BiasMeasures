import torch
from transformers import AutoModel, AutoTokenizer
from pathlib import Path 
from tqdm import tqdm 
import pickle
cwd = Path.cwd()


def get_word2positions(ids, tokenizer, model_name):
	tokens = tokenizer.convert_ids_to_tokens(ids)
	n = len(tokens)
	i = 0
	word2positions = {}
	if 'bert-' in model_name:
		delimiter = '##'
		while i < n:
			token = tokens[i]
			word = token 
			start = i 
			i += 1
			while (i < n and tokens[i].startswith(delimiter)):
				word += tokens[i].replace(delimiter, '')
				i += 1
			end = i
			word = word.lower()
			if word in word2positions:
				word2positions[word].append((start, end))
			else:
				word2positions[word] = [(start, end)]
	elif 'gpt2' in model_name:
		delimiter = 'Ġ'
		while i < n:
			token = tokens[i]
			word = token 
			start, end = i, i + 1 
			i += 1
			while (i < n and not(tokens[i].startswith(delimiter))):
				wordpiece = tokens[i]
				if wordpiece not in {',', '.', '!', ':', ';', '?'}:
					word += tokens[i]
					end += 1
				i += 1
			word= word.replace(delimiter, '')
			word = word.lower()
			if word in word2positions:
				word2positions[word].append((start, end))
			else:
				word2positions[word] = [(start, end)]
	return word2positions


def format_probing_dataset(inputs, layer, model_name, file_descriptor, pickled=False):
	N = len(inputs)
	file_name = cwd / "data" / "contextualized" / "{}_{}examples.pkl".format(file_descriptor, N)
	if pickled:
		inputs = pickle.load(open(file_name, 'rb'))
	else:
		model, tokenizer = AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name, use_fast = True) 
		# outputs = []
		for i, example in tqdm(list(enumerate(inputs))):
			sequence = example['context']
			target_words = set(example['target_words'])
			ids_tensor, ids = tokenizer(sequence, return_tensors = 'pt')['input_ids'], tokenizer(sequence)['input_ids'] 
			word2positions = get_word2positions(ids, tokenizer, model_name)
			if any(word in target_words for word in word2positions):
				sequence_representations = model(ids_tensor, output_hidden_states=True)[2][layer].detach()
				representation_per_position = []
				for word, positions in word2positions.items():
					if word in target_words:
						for start, end in positions:
							representation_per_position.append(torch.mean(sequence_representations[:, start : end, :], 1).squeeze())
				representation = torch.mean(torch.stack(representation_per_position), 0)
				example['representation'] = representation.detach()
			else:
				example['representation'] = None
			# outputs.append(example)
		pickle.dump(inputs, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	return inputs


def fetch_canonical_embeddings(sequences, vocab, layer, model_name, file_descriptor, pickled=False):
	N, V = len(sequences), len(vocab)
	file_name = cwd / "data" / "contextualized" / "{}_{}conts_layer{}_{}_{}vocab.pkl".format(file_descriptor, N, layer, model_name, V)
	if pickled:
		representations, word2count, missing_words = pickle.load(open(file_name, 'rb'))
	else:
		model, tokenizer = AutoModel.from_pretrained(model_name), AutoTokenizer.from_pretrained(model_name, use_fast = True) 
		representations = {word : None for word in vocab}
		word2count = {word : 0 for word in vocab} # Number of contexts/sequences a word appears in
		missing_words = set()
		documents = []
		for sequence in tqdm(sequences):
			ids_tensor, ids = tokenizer(sequence, return_tensors = 'pt')['input_ids'], tokenizer(sequence)['input_ids'] 
			word2positions = get_word2positions(ids, tokenizer, model_name)
			if any(word in vocab for word in word2positions):
				sequence_representations = model(ids_tensor, output_hidden_states=True)[2][layer].detach()
				for word, positions in word2positions.items():
					if word in vocab:
						word2count[word] += 1
						representation_per_position = [torch.mean(sequence_representations[:, start : end, :], 1).squeeze() for start, end in positions]
						representation = torch.mean(torch.stack(representation_per_position), 0).detach().numpy()
						if representations[word] is None:
							representations[word] = representation
						else:
							representations[word] += representation 
		for word, representation in representations.items():
			if representation is None:
				assert word2count[word] == 0
				missing_words.add(word)
				# Recall that the word is the sequence in the decontextualized setting (Bommasani et al., 2020)
				sequence = word
				ids_tensor, ids = tokenizer(sequence, return_tensors = 'pt')['input_ids'], tokenizer(sequence)['input_ids']
				word2positions = get_word2positions(ids, tokenizer, model_name)
				# assert word in get_word2positionsns and len(word2positions[word]) == 1
				start, end = word2positions[word][0]
				sequence_representations = model(ids_tensor, output_hidden_states=True)[2][layer].detach()
				decontextualized_representation = torch.mean(sequence_representations[:, start : end, :], 1).squeeze()
				representations[word] = decontextualized_representation.detach().numpy()
				print(word2positions, representations[word].shape)
			else:
				assert word2count[word] > 0
				representations[word] = representation / word2count[word]
		pickle.dump((representations, word2count, missing_words), open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	return representations, word2count, missing_words


if __name__ == '__main__':
	s = "The existential crisis is mitigated by a shortage of reserves from pacific northwest, conflated with hydroelectric damns."
	model_name = 'gpt2-medium'
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
	ids = tokenizer(s)['input_ids'] 
	word2positions = get_word2positions(ids, tokenizer, model_name)
	model_name = 'bert-base-uncased'
	tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = True)
	ids = tokenizer(s)['input_ids'] 
	word2positions = get_word2positions(ids, tokenizer, model_name)