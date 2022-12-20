from our_word_list import *
from our_estimator import *
from primitive_operations import *
from automated_text_associations import *
from annotate import *
from tqdm import tqdm
from sklearn.metrics import r2_score
from scipy.stats import linregress, spearmanr


def count(annotation_list, groups, label_type):
	extended_groups = groups + ['N/A']
	annotation_counts = {group : 0 for group in extended_groups}
	for example in annotation_list:
		label = example[label_type]
		annotation_counts[label] += 1
	return annotation_counts


def score(annotation_list):
	correct, true_positives, human_predictions, automated_predictions, double_predictions = 0, 0, 0, 0, 0
	for example in annotation_list:
		human_label = example['human-label']
		automated_label = example['automated-argmax-label']
		if human_label == automated_label:
			correct += 1
		if human_label != 'N/A':
			human_predictions += 1
			if human_label == automated_label:
				true_positives += 1
		if automated_label != 'N/A':
			automated_predictions += 1
		if human_label != 'N/A' and automated_label != 'N/A':
			double_predictions += 1
	accuracy = correct / len(annotation_list)
	double_prediction_accuracy = true_positives / double_predictions
	precision, recall = true_positives / automated_predictions, true_positives / human_predictions
	scores = {'accuracy' : accuracy, 'precision' : precision, 'recall' : recall, 'double-prediction-accuracy' : double_prediction_accuracy}
	scores['f1'] = (2 * precision * recall) / (precision + recall)
	return scores


def compute_convergent_validity_bias_scores(human_annotation_counts, automated_annotation_counts, groups):
	normalizer = sum_normalization
	distance = L1_distance
	prior_distribution = get_uniform_distribution(len(groups))
	human_association_vector = [human_annotation_counts[group] for group in groups]
	automated_association_vector = [automated_annotation_counts[group] for group in groups]
	human_bias = compute_our_bias_given_SoA(human_association_vector, normalizer, distance, prior_distribution)
	automated_bias = compute_our_bias_given_SoA(automated_association_vector, normalizer, distance, prior_distribution)
	output = {'human' : {'bias_score' : human_bias['magnitude'], 'difference_vector' : human_bias['direction'], 'directed_bias_score' : human_bias['direction'][0]}}
	output['automated'] = {'bias_score' : automated_bias['magnitude'], 'difference_vector' : automated_bias['direction'], 'directed_bias_score' : automated_bias['direction'][0]}
	return output


def compute_convergent_validity_results(target2human_annotation_file_name, group2wordlist, groups):
	results = {}
	for subcontext_length in (list(range(1, 6))):
		scores = {'accuracy' : 0.0, 'precision' : 0.0, 'recall' : 0.0, 'f1' : 0.0, 'double-prediction-accuracy' : 0.0}
		aggregate_precision, aggregate_recall, aggregate_f1 = 0, 0, 0
		human_bias_list, automated_bias_list = [], []
		for target, human_annotation_file_name in target2human_annotation_file_name.items():
			preamble_length = 4
			context_length = 5
			annotation_list = parse_annotation_file(groups=groups, preamble_length=preamble_length, context_length=context_length, file_name=human_annotation_file_name)
			annotation_list = automated_annotate_contexts(annotation_list, group2wordlist, exclusive = True, subcontext_length = subcontext_length)
			human_annotation_counts = count(annotation_list, groups, 'human-label')
			automated_annotation_counts = count(annotation_list, groups, 'automated-argmax-label')
			bias_scores = compute_convergent_validity_bias_scores(human_annotation_counts, automated_annotation_counts, groups)
			# print(target)
			# print(annotation_counts)
			# print(bias_scores)
			human_bias_list.append(bias_scores['human']['directed_bias_score'])
			automated_bias_list.append(bias_scores['automated']['directed_bias_score'])
			if subcontext_length == 3:
				print("{} & {} & {}".format(target, round(bias_scores['human']['directed_bias_score'], 3), round(bias_scores['automated']['directed_bias_score'], 3)))
			target_scores = score(annotation_list)
			for key in scores:
				scores[key] += target_scores[key]
			# print('Target: {}, Subcontext Length: {}, # Annotations: {}'.format(target, subcontext_length, len(annotation_list)))
			# print('Scores: {}'.format({key : round(value, 3) for key, value in target_scores.items()}))
		slope, intercept, r_value, linear_p_value, std_err = linregress(human_bias_list, automated_bias_list)
		rho, spearman_p_value = spearmanr(human_bias_list, automated_bias_list)
		# print("R^2: {}, linear p: {}, std_err: {}, rho: {}, spearman p: {}".format(r_value ** 2, linear_p_value, std_err, rho, spearman_p_value))
		for key, value in scores.items():
			scores[key] = value / len(target2human_annotation_file_name)
		results[subcontext_length] = scores
	return results


if __name__ == '__main__':
	# Gender Results 
	targets = face_validity_targets
	target2human_annotation_file_name = {}
	n = 250
	for target in targets:
		human_annotation_file_name = cwd / "data" / "annotation" / "gender" / "target={}_n={}.txt".format(target, n)
		target2human_annotation_file_name[target] = human_annotation_file_name
	group2wordlist = {'male' : male_all, 'female' : female_all}
	groups = ['male', 'female']
	gender_results = compute_convergent_validity_results(target2human_annotation_file_name, group2wordlist, groups)
	# print(gender_results)

	# Race Results
	# target2human_annotation_file_name = {}
	# for target in targets:
	# 	human_annotation_file_name = 
	# 	target2human_annotation_file_name[target] = human_annotation_file_name
	# group2wordlist = {'white' : white, 'hispanic' : hispanic, 'asian' : asian}
	# groups = ['white', 'hispanic', 'asian']
	# race_results = compute_convergent_validity_results(target2human_annotation_file_name, group2wordlist, groups)
	# print(race_results)
