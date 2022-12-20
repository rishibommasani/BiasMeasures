from pathlib import Path
import nltk
from tqdm import tqdm
import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from data_loader import *
cwd = Path.cwd()


def valid_targets(prof2ratios, left, right, increment):
	valid_set = set()
	for profession in prof2ratios:
		flag = True
		for year in range(left, right, increment):
			if year not in prof2ratios[profession]:
				flag = False
		if flag:
			valid_set.add(profession)
	return valid_set


def make_contexts(documents, context_length, resource_name, n=False, pickled=False):
	if n:
		file_name = cwd / "data" / "contexts" / "{}_{}contexts-len={}.pkl".format(resource_name, n, context_length)
	else:
		file_name = cwd / "data" / "contexts" / "{}_contexts-len={}.pkl".format(resource_name, context_length)
	if pickled:
		contexts = pickle.load(open(file_name, 'rb'))
	else:
		contexts = []
		for doc in tqdm(documents):
			sents = sent_tokenize(doc)
			sents = [word_tokenize(sent) for sent in sents]
			sents = [sent for sent in sents if len(sent) in range(5, 41)]
			N = len(sents)
			for i in range(N - context_length):
				contexts.append(sents[i : i + context_length])
			if n:
				if len(contexts) >= n:
					contexts = contexts[:n]
					break
		del documents
		pickle.dump(contexts, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	print("{} contexts of sentence length {} for resource: {}".format(len(contexts), context_length, resource_name))
	return contexts


if __name__ == '__main__':
	wikipedia = fetch_wikipedia(pickled=True)
	for context_length in range(1, 6):
		print('Making contexts of length: {}'.format(context_length))
		make_contexts(wikipedia, context_length, 'wikipedia', n = 2 * (10**6), pickled=False)

