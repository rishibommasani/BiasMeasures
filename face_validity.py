from data_loader import *
from preprocess import *
from our_word_list import *
from our_estimator import *
from primitive_operations import *
from annotate import *
from convergent_validity import *
from preprocess_contextualized_representations import *
from train import *


def face_validity_text(documents, targets, groups, group2wordlist, hyperparams):
	human_bias = {}
	automated_bias = {}
	normalizer, distance, prior_distribution = hyperparams['normalizer'], hyperparams['distance'], hyperparams['prior']

	human_annotation_n = 250
	automated_annotation_n = 2 * (10 ** 6)
	preamble_length = 4

	attributes = set()
	for word_list in group2wordlist.values():
		attributes = attributes | {word.lower() for word in word_list}
	context_length = 3
	resource_name = 'wikipedia'
	contexts = make_contexts(documents, context_length, resource_name, automated_annotation_n, pickled=True)
	target2contexts = context_sampler(contexts, targets, attributes, automated_annotation_n)

	for target in targets:
		human_annotation_file_name = cwd / "data" / "annotation" / "gender" / "target={}_n={}.txt".format(target, human_annotation_n)
		context_length = 5
		human_annotation_list = parse_annotation_file(groups=groups, preamble_length=preamble_length, context_length=context_length, file_name=human_annotation_file_name)
		human_annotation_counts = count(human_annotation_list, groups, 'human-label')	
		human_association_vector = [human_annotation_counts[group] for group in groups]
		human_bias[target] = compute_our_bias_given_SoA(human_association_vector, normalizer, distance, prior_distribution)
		
		automated_annotation_list = [{'context' : context, 'automated-argmax-label' : annotate_context(context, group2wordlist, exclusive = True)} for context in target2contexts[target]]
		automated_annotation_counts = count(automated_annotation_list, groups, 'automated-argmax-label')	
		automated_association_vector = [automated_annotation_counts[group] for group in groups]
		automated_bias[target] = compute_our_bias_given_SoA(automated_association_vector, normalizer, distance, prior_distribution)
	return human_bias, automated_bias


def face_validity_embeddings(w2v, GloVe, targets, attribute_list, missing_words, hyperparams):
	similarity = cosine
	normalizer, distance, prior_distribution = hyperparams['normalizer'], hyperparams['distance'], hyperparams['prior']
	embeddings_attribute_list = [word_list - missing_words for word_list in attribute_list]
	_, w2v_bias = compute_our_bias(w2v, embeddings_attribute_list, targets, normalizer, distance, similarity, prior_distribution, verbose=True)
	_, GloVe_bias = compute_our_bias(GloVe, embeddings_attribute_list, targets, normalizer, distance, similarity, prior_distribution, verbose=True)
	return w2v_bias, GloVe_bias


def face_validity_contextualized_probing(contexts, targets, attribute_list, groups, group2wordlist):
	n = 1000
	model_name = 'bert-base-uncased'
	layer = 12
	file_descriptor = 'face_validity_wikipedia'
	attributes = set()
	for word_list in group2wordlist.values():
		attributes = attributes | {word.lower() for word in word_list}
	target2contexts = context_sampler(contexts, targets, attributes, n)
	examples = []
	label2index = {group : i for i, group in enumerate(groups + ['N/A'])}
	index2label = {i : group for group, i in label2index.items()}
	for target, contexts in target2contexts.items():
		for context in contexts:
			example = {}
			gold_label = annotate_context(context, group2wordlist, exclusive = True)
			example['gold_label'] = gold_label
			example['gold_label_id'] = label2index[gold_label]
			example['context'] = ' '.join(context[0])
			example['target_concept'] = target
			example['target_words'] = {target}
			examples.append(example)
	examples = format_probing_dataset(examples, layer, model_name, file_descriptor, pickled=True)
	target2counts = {}
	for example in examples:
		target = example['target_concept']
		gold_label = example['gold_label']
		if target in target2counts:
			target2counts[target][gold_label] = target2counts[target].get(gold_label, 0) + 1
		else:
			target2counts[target] = {gold_label : 1}
	target2counts, target2results = train_probe(examples, target2counts, model_dim=768, label_space_size=len(label2index), epochs=10, batch_size=10)
	target2SoA = {}
	for target, counts in target2counts.items():
		SoA = []
		for group in groups:
			index = label2index[group]
			group_association = counts['predicted'][index]
			SoA.append(group_association)
		target2SoA[target] = SoA
		print('{} : {}'.format(target, SoA))
		print('{} : {}'.format(target, counts))
	return target2results, target2SoA


def face_validity_contextualized(documents, vocab, targets, attribute_list, groups, group2wordlist, hyperparams):
	normalizer, distance, prior_distribution = hyperparams['normalizer'], hyperparams['distance'], hyperparams['prior']
	
	context_length = 1
	resource_name = 'wikipedia'
	N_probing = 150000
	documents = documents[:N_probing]
	contexts = make_contexts(documents, context_length, resource_name, pickled=True)
	target2results, target2SoA = face_validity_contextualized_probing(contexts, targets, attribute_list, groups, group2wordlist)
	BERT_probing_bias = {}
	for target, SoA in target2SoA.items():
		BERT_probing_bias[target] = compute_our_bias_given_SoA(SoA, normalizer, distance, prior_distribution)
	# sequences = [' '.join(context[0]) for context in contexts]
	# N_reduction = 2000000
	# sequences = sequences[:N_reduction]
	# model_name = 'bert-base-uncased'
	# layer = 12
	# BERT_canonical_embeddings, word2count, missing_words = fetch_canonical_embeddings(sequences, vocab, layer, model_name, file_descriptor = 'general', pickled=True)
	# print(missing_words)
	# similarity = cosine
	# embeddings_attribute_list = [word_list - missing_words for word_list in attribute_list]
	# _, BERT_reduction_bias = compute_our_bias(BERT_canonical_embeddings, embeddings_attribute_list, targets, normalizer, distance, similarity, prior_distribution, verbose=True)
	BERT_reduction_bias = None
	return BERT_probing_bias, BERT_reduction_bias	


def face_validity_results(male_word_list, female_word_list, vocab, normalizer, distance):
	hyperparams = {}
	prior_distribution = get_uniform_distribution(2)
	hyperparams['normalizer'], hyperparams['distance'], hyperparams['prior'] = normalizer, distance, prior_distribution 

	targets = face_validity_targets
	groups = ['male', 'female']
	attribute_list = [male_word_list, female_word_list]
	group2wordlist = {'male' : male_word_list, 'female' : female_word_list}

	documents = fetch_wikipedia(pickled=True)
	# human_bias, automated_bias = face_validity_text(documents, targets, groups, group2wordlist, hyperparams)

	# w2v, GloVe, missing_words = fetch_w2v_GloVe_embeddings(vocab, pickled=False)
	# w2v_bias, GloVe_bias = face_validity_embeddings(w2v, GloVe, targets, attribute_list, missing_words, hyperparams)

	BERT_probing_bias, BERT_reduction_bias = face_validity_contextualized(documents, vocab, targets, attribute_list, groups, group2wordlist, hyperparams)

	target2scores = {target : {} for target in targets}
	for target in ordered_face_validity_targets:
		# target2scores[target]['text-human'] = human_bias[target]['direction'][0]
		# target2scores[target]['text-automated'] = automated_bias[target]['direction'][0]
		# target2scores[target]['embeddings-w2v'] = w2v_bias[target]['direction'][0]
		# target2scores[target]['embeddings-GloVe'] = GloVe_bias[target]['direction'][0]
		# target2scores[target]['contextualized-BERT-reduction'] = BERT_reduction_bias[target]['direction'][0]
		target2scores[target]['contextualized-BERT-probing'] = BERT_probing_bias[target]['direction'][0]
		target_bias_scores = target2scores[target]
	return target2scores


def write_face_validity_results(target2scores):
	file_name = cwd / "results" / "face_validity_contextualized.table.tex"
	file = open(file_name, "w")
	ordered_targets = ordered_face_validity_targets
	# columns = ['text-human', 'text-automated', 'embeddings-w2v', 'embeddings-GloVe', 'contextualized-BERT-reduction', 'contextualized-BERT-probing']
	columns = ['contextualized-BERT-reduction', 'contextualized-BERT-probing']
	file.write('& ' + ' & '.join(columns) + ' \\\\ \n')
	for target in ordered_face_validity_targets:
		s = target2scores[target]
		line = ' & '.join([target] + [str(round(s[column_name], 3)) for column_name in columns]) + ' \\\\ \n'
		file.write(line)


if __name__ == '__main__':
	vocab = face_validity_targets | gender_all | race
	normalizer = sum_normalization
	distance = L1_distance
	target2scores = face_validity_results(male, female, vocab, normalizer, distance)
	write_face_validity_results(target2scores)