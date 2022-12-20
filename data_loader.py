from tqdm import tqdm
from pathlib import Path
import pickle
import numpy as np
from gensim.models import KeyedVectors
import io
import nltk
import wget
import csv
import json
cwd = Path.cwd()


def fetch_wikipedia(pickled=False):
	wikipedia_file_name = cwd / "data" / "sources" / "documents_utf8_filtered_20pageviews.csv"
	file_name = cwd / "data" / "sources" / "wikipedia.pkl"
	if pickled:
		documents = pickle.load(open(file_name, 'rb'))
	else:
		with open(wikipedia_file_name, 'r') as f:
			raw_documents = list(f)
		documents = []
		for doc in tqdm(raw_documents):
			left = doc.index('  ')
			doc = doc[left + 2: -2]
			doc = doc.strip()
			documents.append(doc)
		pickle.dump(documents, open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	return documents


def fetch_w2v_GloVe_embeddings(vocab, pickled=False):
	file_name = cwd / "data" / "embeddings" / "w2v-GloVe_vocab_size={}.pkl".format(len(vocab))
	if pickled:
		w2v, GloVe, missing_words = pickle.load(open(file_name, 'rb'))
	else:
		w2v, GloVe = {}, {}
		w2v_file_name = cwd / "data" / "embeddings" / "GoogleNews-vectors-negative300.bin"
		# tempr = "data\\embeddings\\GoogleNews-vectors-negative300.bin"
		GloVe_file_name = cwd / "data" / "embeddings" / "glove.6B.300d.txt"
		dimension = 300
		full_wv = KeyedVectors.load_word2vec_format(w2v_file_name, binary=True)
		with open(GloVe_file_name, mode='r', encoding='utf-8') as f:
			for i, line in tqdm(enumerate(f)):
				tokens = line.split()
				word = tokens[0].lower()
				embedding = np.array([float(val) for val in tokens[1:]])
				if word in vocab and word in full_wv:
					assert len(embedding) == dimension
					w2v[word] = full_wv[word]
					GloVe[word] = embedding	
		assert set(w2v.keys()) == set(GloVe.keys())
		missing_words = vocab - set(w2v.keys())	
		pickle.dump((w2v, GloVe, missing_words), open(file_name, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
	print("Fetched w2v and GloVe embeddings")
	return w2v, GloVe, missing_words


def fetch_coha_word2vec_embeddings(vocab, pickled=False):
	file_name = cwd / "data" / "embeddings" / "coha_vocab_size={}.pkl".format(len(vocab))
	if pickled:
		final_embeddings, missing_words = pickle.load(open(file_name, 'rb'))
		return final_embeddings, missing_words
	else:
		missing_words = set()
		embeddings = {year : {} for year in range(1900, 2000, 10)}
		for year in tqdm(list(range(1900, 2000, 10))):
			vocab_file = cwd / "data" / "embeddings" / "{}-vocab.pkl".format(year)
			embedding_file = cwd / "data" / "embeddings" / "{}-w.npy".format(year)
			word2vec_vocab = pickle.load(open(vocab_file, 'rb'))
			w2i = {w.lower() : i for i, w in enumerate(word2vec_vocab)}
			wv = np.load(embedding_file)
			for w in vocab:
				if w in w2i:
					embedding = wv[w2i[w]]
					embeddings[year][w] = embedding
				else:
					missing_words.add(w)
		final_embeddings = {year : {} for year in range(1900, 2000, 10)}
		for w in vocab - missing_words:
			for year in tqdm(list(range(1900, 2000, 10))):
				final_embeddings[year][w] = embeddings[year][w]
		pickle.dump((final_embeddings, missing_words), open(file_name, 'wb'))
		return final_embeddings, missing_words


def fetch_gpt2_data(pickled=False):
	pretraining_file_name = cwd / "data" / "sources" / "gpt-2" / "data" / "webtext.train.jsonl"
	synthetic_file_name = cwd / "data" / "sources" / "gpt-2" / "data" / "medium-345M.train.jsonl"
	pickle_file_name = cwd / "data" / "sources" / "gpt2.pkl"
	if pickled:
		documents = pickle.load(open(pickle_file_name, 'rb'))
	else:
		with open(pretraining_file_name, 'r') as f:
			pretraining_documents = [json.loads(line)['text'] for line in f]
		with open(synthetic_file_name, 'r') as f:
			synthetic_documents = [json.loads(line)['text'] for line in f]
		documents = {'pretraining' : pretraining_documents, 'synthetic' : synthetic_documents}
		pickle.dump(documents, open(pickle_file_name, 'wb'))
	return documents


def fetch_occupation_statistics():
	gender_file_name = cwd / "data" / "sources" / "occupation_percentages_gender_occ1950.csv"
	race_file_name = cwd / "data" / "sources" / "occupation_percentages_race_occ1950.csv"
	prof2gender_ratio, prof2race_ratio = {}, {}
	data = []
	with open(gender_file_name) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for row in readCSV:
			data.append(row)
	data = data[1:]
	for year, profession, _, female, male in data:
		profession = profession.lower()
		year = int(year)
		female, male = float(female), float(male)
		normalization_constant = female + male
		if normalization_constant:
			female, male = female / normalization_constant, male / normalization_constant
			if profession in prof2gender_ratio:
				prof2gender_ratio[profession][year] = [female, male]
			else:
				prof2gender_ratio[profession] = {year: [female, male]}
	data = []
	with open(race_file_name) as csvfile:
		readCSV = csv.reader(csvfile, delimiter=',')
		for row in readCSV:
			data.append(row)
	data = data[1:]
	for year, profession, _, _, _, white, hispanic, asian, _ in data:
		profession = profession.lower()
		year = int(year)
		white, hispanic, asian = float(white), float(hispanic), float(asian)
		normalization_constant = white + hispanic + asian
		if normalization_constant:
			white, hispanic, asian = white / normalization_constant, hispanic / normalization_constant, asian / normalization_constant
			if profession in prof2race_ratio:
				prof2race_ratio[profession][year] = [white, hispanic, asian]
			else:
				prof2race_ratio[profession] = {year: [white, hispanic, asian]}
	return prof2gender_ratio, prof2race_ratio