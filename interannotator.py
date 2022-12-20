from pathlib import Path 
from tqdm import tqdm 
from our_word_list import *
cwd = Path.cwd()


def parse_IAA_files(folder_names, social_axis, k, targets, n_2):
	N = len(folder_names)
	preamble_length = 3
	context_length = 5
	padding_length = 1
	n_1 = len(targets)
	n = n_1 * n_2
	folder2annotations = {}
	for folder_name in folder_names:
		annotations = []
		for target in sorted(targets):
			target_annotations = []
			file_name = cwd / 'data' / 'annotation' / 'IAA' / folder_name / social_axis / '{}_{}.txt'.format(target, social_axis)
			with open(file_name, 'r') as f:
				data = list(f)
			i = preamble_length
			for j in range(1, n_2 + 1):
				i += (context_length + padding_length + 1)
				annotation_str = data[i].strip()
				print(target)
				annotation_label = int(annotation_str.split(':')[-1].strip())
				assert 0 < annotation_label <= k
				i += 1
				if i < len(data):
					feedback = data[i].strip()
					if feedback:
						i += 1	
				else:
					feedback = ''
				annotation = {'target' : target, 'example_id' : j, 'label' : annotation_label, 'feedback' : feedback}
				target_annotations.append(annotation)
				i += 1
			annotations += target_annotations
		folder2annotations[folder_name] = annotations
	assert len(folder2annotations) == N
	for annotations in folder2annotations.values():
		assert len(annotations) == n 
		for annotation in annotations:
			assert 0 < annotation['label'] <= k
	return folder2annotations


def compute_fleiss_kappa(folder2annotations, k):
	n = len(folder2annotations)
	annotations_list = [[annotation['label'] for annotation in annotations] for annotations in folder2annotations.values()]
	N = len(annotations_list[0])
	print('N: {}, n: {}, k: {}'.format(N, n, k))
	assert all([len(annotations) == N for annotations in annotations_list])
	n_ij_dict = {(i, j) : sum([annotations[i] == j for annotations in annotations_list]) for i in range(N) for j in range(1, k + 1)}
	print('Annotations')
	for annotations in annotations_list:
		print(annotations)
	for j in range(1, k + 1):
		print([n_ij_dict[(i, j)] for i in range(N)])
	p_j_dict = {}
	for j in range(1, k + 1):
		p_j_unnormalized = sum([n_ij_dict[(i, j)] for i in range(N)])
		p_j = p_j_unnormalized / (N * n)
		p_j_dict[j] = p_j 
	assert sum(p_j_dict.values()) == 1
	print('p_j dict: {} \n \n'.format(p_j_dict))
	P_i_dict = {i : None for i in range(N)}
	for i in range(N):
		P_i_unnormalized = 0
		for j in range(1, k + 1):
			n_ij = n_ij_dict[(i, j)] 
			matching_pairs = n_ij * (n_ij - 1)
			P_i_unnormalized += matching_pairs
		P_i = P_i_unnormalized / (n * (n - 1))
		P_i_dict[i] = P_i 
	print('P_i dict: {} \n \n'.format(P_i_dict))
	P_bar = sum(P_i_dict.values()) / N 
	P_e_bar = sum([p_j ** 2 for p_j in p_j_dict.values()]) 
	fleiss_kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
	print('P_bar: {}, P_e_bar: {}, fleiss_kappa: {}'.format(P_bar, P_e_bar, fleiss_kappa))
	return fleiss_kappa





if __name__ == '__main__':
	folder_names = ['claire', 'tianyi_annotated', 'xikun']
	social_axis = 'gender'
	k = 3 
	targets = ordered_face_validity_targets
	n_2 = 5
	folder2annotations = parse_IAA_files(folder_names, social_axis, k, targets, n_2)
	# for f1, f2 in [('claire', 'tianyi_annotated'), ('claire', 'xikun'), ('tianyi_annotated', 'xikun')]:
	# 		a1, a2 = folder2annotations[f1], folder2annotations[f2]
	# 		assert len(a1) == len(a2)
	# 		print(f1, f2, len(a1))
	# 		print([x['label'] - y['label'] for x, y in zip(a1, a2)])
	for folder, annotations in folder2annotations.items():
		print(folder)
		print(annotations)
		print('\n')
	compute_fleiss_kappa(folder2annotations, k)

