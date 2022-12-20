from data_loader import *
from preprocess import *
from our_word_list import *
from our_estimator import *
from primitive_operations import *
from annotate import *
from convergent_validity import *
from preprocess_contextualized_representations import *
from train import *


def compute_textual_SoA(documents, targets, groups, group2wordlist, resource_name, pickled=False):
	print('Computing Textual SoA in Overamplification Experiment for: {}'.format(resource_name))
	target2SoA = {}
	num_contexts = 10 ** 6
	attributes = set()
	for word_list in group2wordlist.values():
		attributes = attributes | {word.lower() for word in word_list}
	context_length = 3
	contexts = make_contexts(documents, context_length, resource_name, num_contexts, pickled=pickled)
	target2contexts = context_sampler(contexts, targets, attributes, num_contexts)

	for target in targets:
		annotation_list = [{'context' : context, 'automated-argmax-label' : annotate_context(context, group2wordlist, exclusive = True)} for context in target2contexts[target]]
		annotation_counts = count(annotation_list, groups, 'automated-argmax-label')	
		association_vector = [annotation_counts[group] for group in groups]
		target2SoA[target] = association_vector 
	return target2SoA


def compute_representation_SoA(documents, targets, groups, group2wordlist, resource_name, pickled=False):
	print('Computing Representation SoA in Overamplification Experiment for: {}'.format(resource_name))
	n = 1000
	model_name = 'gpt2-medium'
	layer = 24
	hidden_dim = 1024
	context_length = 1
	attributes = set()
	for word_list in group2wordlist.values():
		attributes = attributes | {word.lower() for word in word_list}
	context_length = 1
	contexts = make_contexts(documents, context_length, resource_name, pickled=pickled)
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
			example['context'] = ' '.join([' '.join(context[i]) for i in range(context_length)])
			example['target_concept'] = target
			example['target_words'] = {target}
			examples.append(example)
	file_descriptor = 'overamplification_gpt2_representations'
	examples = format_probing_dataset(examples, layer, model_name, file_descriptor, pickled=pickled)
	target2counts = {}
	for example in examples:
		target = example['target_concept']
		gold_label = example['gold_label']
		if target in target2counts:
			target2counts[target][gold_label] = target2counts[target].get(gold_label, 0) + 1
		else:
			target2counts[target] = {gold_label : 1}
	target2counts, target2results = train_probe(examples, target2counts, model_dim=hidden_dim, label_space_size=len(label2index), epochs=30, batch_size=50)
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


def overamplification_results(male_word_list, female_word_list, vocab, normalizer, distance):
	all_documents = fetch_gpt2_data(pickled=True)
	
	targets = face_validity_targets
	groups = ['male', 'female']
	attribute_list = [male_word_list, female_word_list]
	group2wordlist = {'male' : male_word_list, 'female' : female_word_list}

	pretrained_target2SoA = compute_textual_SoA(all_documents['pretraining'], targets, groups, group2wordlist, resource_name = 'gpt-2_pretraining', pickled=True)
	_, representations_target2SoA = compute_representation_SoA(all_documents['pretraining'], targets, groups, group2wordlist, resource_name = 'gpt-2_representations', pickled=True) 
	synthetic_target2SoA = compute_textual_SoA(all_documents['synthetic'], targets, groups, group2wordlist, resource_name = 'gpt-2_synthetic', pickled=True)

	target2scores = {target : {} for target in targets}
	for target in targets:
		pretrained_SoA = pretrained_target2SoA[target] # L1
		representation_SoA = representations_target2SoA[target] # L2
		synthetic_SoA = synthetic_target2SoA[target] # L3
		U = get_uniform_distribution(2)
		target2scores[target]['L2;L1'] = compute_our_bias_given_SoA(representation_SoA, normalizer, distance, normalizer(pretrained_SoA))
		target2scores[target]['L3;L2'] = compute_our_bias_given_SoA(synthetic_SoA, normalizer, distance, normalizer(representation_SoA))
		target2scores[target]['L3;L1'] = compute_our_bias_given_SoA(synthetic_SoA, normalizer, distance, normalizer(pretrained_SoA))
		target2scores[target]['L1;U'] = compute_our_bias_given_SoA(pretrained_SoA, normalizer, distance, U)
		target2scores[target]['L2;U'] = compute_our_bias_given_SoA(representation_SoA, normalizer, distance, U)
		target2scores[target]['L3;U'] = compute_our_bias_given_SoA(synthetic_SoA, normalizer, distance, U)
		target2scores[target]['L2;U-L1;U'] = target2scores[target]['L2;U']['magnitude'] - target2scores[target]['L1;U']['magnitude']
		target2scores[target]['L3;U-L2;U'] = target2scores[target]['L3;U']['magnitude'] - target2scores[target]['L2;U']['magnitude']
		target2scores[target]['L3;U-L1;U'] = target2scores[target]['L3;U']['magnitude'] - target2scores[target]['L1;U']['magnitude']
		print(target)
		for key in target2scores[target]:
			print(key, target2scores[target][key])
		# print('{} : {}'.format(target, target2scores[target]))
	return target2scores


def write_overamplification_results(target2scores):
	file_name = cwd / "results" / "overamplification_relative.table.tex"
	file = open(file_name, "w")
	ordered_targets = ordered_face_validity_targets
	columns = ['L2;L1', 'L3;L2', 'L3;L1']
	file.write('& ' + ' & '.join(columns) + ' \\\\ \n')
	for target in ordered_face_validity_targets:
		s = target2scores[target]
		line = ' & '.join([target] + [str(round(s[column_name]['direction'][0], 3)) for column_name in columns]) + ' \\\\ \n'
		file.write(line)

	file_name = cwd / "results" / "overamplification_uniform.table.tex"
	file = open(file_name, "w")
	ordered_targets = ordered_face_validity_targets
	columns = ['L2;L1', 'L3;L2', 'L1;U', 'L2;U', 'L3;U']
	file.write('& ' + ' & '.join(columns) + ' \\\\ \n')
	for target in ordered_face_validity_targets:
		s = target2scores[target]
		line = ' & '.join([target] + [str(round(s[column_name]['direction'][0], 3)) for column_name in columns]) + ' \\\\ \n'
		file.write(line)


if __name__ == '__main__':
	vocab = face_validity_targets | gender_all | race
	normalizer = sum_normalization
	distance = L1_distance
	target2scores = overamplification_results(male, female, vocab, normalizer, distance)
	write_overamplification_results(target2scores)