import numpy


# [context] : list of sentences, which themselves are lists of words or whitespace-tokenized strings
# [group_word_list] : list of words for each group
# Optional [exclusive] : indicates whether a context can be labelled g_i if it doesn't refer any group for any g_j
# Returns [output] : group label or 'N/A' assigned to [context]
def annotate_context(context, group2wordlist, exclusive = True):
	k = len(group2wordlist)
	context_length = len(context)
	assert k > 0 and context_length > 0

	context_words = set()
	context_counts = {}
	for sentence in context:
		if type(sentence) is str:
			sentence = sentence.split()
		for word in sentence:
			word = word.lower()
			context_words.add(word)
			context_counts[word] = context_counts.get(word, 0) + 1
	
	seen_groups = {}
	output = None
	for group, word_list in group2wordlist.items():
		word_set = {w.lower() for w in word_list}
		if len(word_set & context_words) > 0:
			# seen_groups[group] = len(word_set & context_words)
			seen_groups[group] = sum([context_counts.get(word, 0) for word in word_set])
	if exclusive:
		if len(seen_groups) == 1:
			output = list(seen_groups.keys())[0]
		else:
			output = 'N/A'
	else:
		if len(seen_groups) > 0:
			ranked_groups = sorted([group for group in seen_groups.keys()], key = lambda g: seen_groups[g])
			output = ranked_groups[-1]
		else:
			output = 'N/A'
	return output 


def automated_annotate_contexts(annotation_list, group2wordlist, exclusive = True, subcontext_length = None):
	ordered_groups = sorted(list(group2wordlist.keys()))
	group2index = {group : index for index, group in enumerate(ordered_groups)}
	index2group = {index : group for group, index in group2index.items()}
	updated_annotation_list = []
	for example in annotation_list:
		context = example['context']
		if subcontext_length:
			context_length = len(context)
			assert context_length >= subcontext_length
			subcontext_labels = [] 
			for index in range(context_length - subcontext_length + 1):
				subcontext = context[index : index + subcontext_length]
				label = annotate_context(subcontext, group2wordlist, exclusive = exclusive)
				subcontext_labels.append(label)
			num_subcontexts = len(subcontext_labels)
			if subcontext_labels == (['N/A'] * num_subcontexts):
				label = 'N/A'
				labels_dict = {group : 0.0 for group in ordered_groups}
				labels_dict['N/A'] = 1.0
			else:
				counts_vector = [0] * len(ordered_groups)
				normalization_constant = 0
				for label in subcontext_labels:
					if label != 'N/A':
						counts_vector[group2index[label]] += 1
						normalization_constant += 1
				label = index2group[numpy.argmax(counts_vector)]
				labels_dict = {group : counts_vector[group2index[group]] / normalization_constant for group in ordered_groups}
				labels_dict['N/A'] = 0.0
		else:
			label = annotate_context(context, group2wordlist, exclusive = exclusive)
			labels_dict = {group : 0.0 for group in ordered_groups + ['N/A']}
			labels_dict[label] = 1.0

		example['automated-argmax-label'] = label
		assert len(labels_dict) == len(ordered_groups) + 1
		example['automated-group-scores'] = labels_dict
		updated_annotation_list.append(example)
	# for index, example in enumerate(updated_annotation_list[:90]):
	# 	print('Index: {} \n'.format(index))
	# 	for key, value in example.items():
	# 		print('{} : {}'.format(key, value))
	# 	print('\n')
	# exit()
	return updated_annotation_list