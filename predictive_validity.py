from data_loader import *
from our_word_list import *
from preprocess import *
from aligned_estimators import *
from unaligned_estimators import *
from our_estimator import *
from scratch import *
from primitive_operations import *
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import linregress, spearmanr
cwd = Path.cwd()


def census_bias(ratios):
	N = len(ratios)
	uniform_distribution = get_uniform_distribution(N)
	bias = L1_distance(ratios, uniform_distribution)
	return bias 		


def write_finegrained_census_correlations(file_descriptor, census, prior_estimators, ours):
	ordered_professions = sorted(list(census.keys()))  
	census_list = [census[prof] for prof in ordered_professions]
	for embedding_descriptor in ['w2v', 'GloVe']:
		file_name = cwd / "results" / "finegrained_census" / "{}.{}.table.tex".format(file_descriptor, embedding_descriptor)
		file = open(file_name, "w")
		file.write(embedding_descriptor + '\n \n')
		for estimator_name, estimator_scores_dict in prior_estimators[embedding_descriptor].items():
			estimator_scores_list = [estimator_scores_dict[prof] for prof in ordered_professions]
			slope, intercept, r_value, linear_p_value, std_err = linregress(census_list, np.array(estimator_scores_list))
			rho, spearman_p_value = spearmanr(census_list, estimator_scores_list)
			file.write("{} & {} & {} & {} & {} \\\\".format(estimator_name, round(rho, 3), round(spearman_p_value, 3), round(r_value ** 2, 3), round(linear_p_value, 3)))
			file.write('\n')
		file.write('\\midrule \n')
		for params, estimator_scores_dict in ours[embedding_descriptor].items():
			estimator_scores_list = [estimator_scores_dict[prof] for prof in ordered_professions]
			slope, intercept, r_value, linear_p_value, std_err = linregress(census_list, np.array(estimator_scores_list))
			rho, spearman_p_value = spearmanr(census_list, estimator_scores_list)
			file.write("{} & {} & {} & {} & {} \\\\".format(params, round(rho, 3), round(spearman_p_value, 3), round(r_value ** 2, 3), round(linear_p_value, 3)))
			file.write('\n')
		file.write('\\bottomrule \n')


def write_finegrained_census_bias(file_descriptor, census, prior_estimators, ours):
	raise NotImplementedError
	# file_name = cwd / "results" / "averaged_historical" / "{}.txt".format(file_descriptor)
	# file = open(file_name, "w")  
	# file.write("Census: {}".format(census))
	# file.write('\n')
	# for estimator_name, estimator_scores in prior_estimators.items():
	# 	file.write("{}: {}".format(estimator_name, estimator_scores))
	# 	file.write('\n')
	# file.write('\n')
	# for params, estimator_scores in ours.items():
	# 	file.write("{}: {}".format(params, estimator_scores))
	# 	file.write('\n')
	# file.write('\n')


def finegrained_census_bias():
	prof2gender_ratio, prof2race_ratio = fetch_occupation_statistics()
	vocab = gender_2010_targets | race_2010_targets | gender | race 
	w2v, GloVe, missing_words = fetch_w2v_GloVe_embeddings(vocab, pickled=True)
	print("Missing words in pretrained embeddings: {}".format(missing_words))
	updated_gender_census_targets = gender_2010_targets - missing_words
	updated_race_census_targets = race_2010_targets - missing_words
	updated_race = race - missing_words
	updated_gender = gender - missing_words

	census_gender_bias = {}
	for profession in updated_gender_census_targets:
		bias = census_bias(prof2gender_ratio[profession][2010])
		census_gender_bias[profession] = bias

	prior_keys = ['garg-cosine', 'garg-euclidean', 'caliskan', 'manzini', 'bolukbasi-conventional', 'ethayarajh-conventional']
	
	all_prior_bias = {'w2v': None, 'GloVe': None}
	for embedding_descriptor, embeddings in [('w2v', w2v), ('GloVe', GloVe)]:
		prior_bias = {key : [] for key in prior_keys}
		aligned_estimates = compute_all_aligned_estimates(embeddings, aligned_male_female, updated_gender_census_targets, abs, verbose = True)
		gender_A_list = [male - missing_words, female - missing_words]
		unaligned_estimates = compute_all_unaligned_estimates(embeddings, gender_A_list, updated_gender_census_targets, abs, verbose = True)
		for key in aligned_estimates:
			prior_bias[key] = aligned_estimates[key][1]
		for key in unaligned_estimates:
			prior_bias[key] = unaligned_estimates[key][1]
		all_prior_bias[embedding_descriptor] = prior_bias
	
	all_our_bias = {'w2v': None, 'GloVe': None}
	for embedding_descriptor, embeddings in [('w2v', w2v), ('GloVe', GloVe)]:
		our_bias = {key : [] for key in parameter_triples}
		gender_A_list = [male - missing_words, female - missing_words]
		prior_distribution = get_uniform_distribution(2)
		for parameter_descriptor, parameter_triple in parameter_triples.items():
			distance, normalizer, similarity = parameter_triple 
			_, bias_per_target_word = compute_our_bias(embeddings, gender_A_list, updated_gender_census_targets, normalizer, distance, similarity, prior_distribution, verbose=True)
			our_bias[parameter_descriptor] = {target_word : value['magnitude'] for target_word, value in bias_per_target_word.items()}
		all_our_bias[embedding_descriptor] = our_bias			

	# write_finegrained_census_bias('gender', census_gender_bias, all_prior_bias, all_our_bias)
	write_finegrained_census_correlations('gender_correlations', census_gender_bias, all_prior_bias, all_our_bias)

	census_race_bias = {}
	for profession in updated_race_census_targets:
		bias = census_bias(prof2race_ratio[profession][2010])
		census_race_bias[profession] = bias

	prior_keys = ['manzini']
	
	all_prior_bias = {'w2v': None, 'GloVe': None}
	for embedding_descriptor, embeddings in [('w2v', w2v), ('GloVe', GloVe)]:
		prior_bias = {key : [] for key in prior_keys}
		race_A_list = [white - missing_words, hispanic - missing_words, asian - missing_words]
		unaligned_estimates = compute_all_unaligned_estimates(embeddings, race_A_list, updated_race_census_targets, abs, verbose = True)
		for key in unaligned_estimates:
			prior_bias[key] = unaligned_estimates[key][1]
		all_prior_bias[embedding_descriptor] = prior_bias
	
	all_our_bias = {'w2v': None, 'GloVe': None}
	for embedding_descriptor, embeddings in [('w2v', w2v), ('GloVe', GloVe)]:
		our_bias = {key : [] for key in parameter_triples}
		race_A_list = [white - missing_words, hispanic - missing_words, asian - missing_words]
		prior_distribution = get_uniform_distribution(3)
		for parameter_descriptor, parameter_triple in parameter_triples.items():
			distance, normalizer, similarity = parameter_triple 
			_, bias_per_target_word = compute_our_bias(embeddings, race_A_list, updated_race_census_targets, normalizer, distance, similarity, prior_distribution, verbose=True)
			our_bias[parameter_descriptor] = {target_word : value['magnitude'] for target_word, value in bias_per_target_word.items()}
		all_our_bias[embedding_descriptor] = our_bias			

	# write_finegrained_census_bias('race', census_race_bias, all_prior_bias, all_our_bias)
	write_finegrained_census_correlations('race_correlations', census_race_bias, all_prior_bias, all_our_bias)


def write_averaged_historical_correlations(file_descriptor, census, prior_estimators, ours):
	file_name = cwd / "results" / "averaged_historical" / "{}.table.tex".format(file_descriptor)
	file = open(file_name, "w")  
	census = np.array(census)
	for estimator_name, estimator_scores in prior_estimators.items():
		slope, intercept, r_value, linear_p_value, std_err = linregress(census, np.array(estimator_scores))
		rho, spearman_p_value = spearmanr(census, estimator_scores)
		file.write("{} & {} & {} & {} & {} \\\\".format(estimator_name, round(rho, 3), round(spearman_p_value, 3), round(r_value ** 2, 3), round(linear_p_value, 3)))
		file.write('\n')
	file.write('\\midrule \n')
	for params, estimator_scores in ours.items():
		slope, intercept, r_value, linear_p_value, std_err = linregress(census, np.array(estimator_scores))
		rho, spearman_p_value = spearmanr(census, estimator_scores)
		file.write("{} & {} & {} & {} & {} \\\\".format(params, round(rho, 3), round(spearman_p_value, 3), round(r_value ** 2, 3), round(linear_p_value, 3)))
		file.write('\n')
	file.write('\\bottomrule \n')


def write_averaged_historical_bias(file_descriptor, census, prior_estimators, ours):
	file_name = cwd / "results" / "averaged_historical" / "{}.txt".format(file_descriptor)
	file = open(file_name, "w")  
	file.write("Census: {}".format(census))
	file.write('\n')
	for estimator_name, estimator_scores in prior_estimators.items():
		file.write("{}: {}".format(estimator_name, estimator_scores))
		file.write('\n')
	file.write('\n')
	for params, estimator_scores in ours.items():
		file.write("{}: {}".format(params, estimator_scores))
		file.write('\n')
	file.write('\n')


def averaged_historical_bias():
	prof2gender_ratio, prof2race_ratio = fetch_occupation_statistics()
	vocab = gender_census_targets | race_census_targets | gender | race 
	all_embeddings, missing_words = fetch_coha_word2vec_embeddings(vocab, pickled=False)
	print("Missing words in COHA embeddings: {}".format(missing_words))
	updated_gender_census_targets = gender_census_targets - missing_words
	updated_race_census_targets = race_census_targets - missing_words
	updated_race = race - missing_words
	updated_gender = gender - missing_words

	historical_gender_bias = []
	for year in range(1900, 2000, 10):
		bias = 0.0
		for profession in updated_gender_census_targets:
			bias += census_bias(prof2gender_ratio[profession][year])
		bias = bias / len(updated_gender_census_targets)
		historical_gender_bias.append(bias)

	prior_keys = ['garg-cosine', 'garg-euclidean', 'caliskan', 'manzini', 'bolukbasi-conventional', 'ethayarajh-conventional']
	historical_prior_bias = {key : [] for key in prior_keys}
	for year in range(1900, 2000, 10):
		embeddings = all_embeddings[year]
		aligned_estimates = compute_all_aligned_estimates(embeddings, aligned_male_female, updated_gender_census_targets, abs, verbose = False)
		gender_A_list = [male - missing_words, female - missing_words]
		unaligned_estimates = compute_all_unaligned_estimates(embeddings, gender_A_list, updated_gender_census_targets, abs, verbose = False)
		for key in aligned_estimates:
			historical_prior_bias[key].append(aligned_estimates[key])
		for key in unaligned_estimates:
			historical_prior_bias[key].append(unaligned_estimates[key])
	
	historical_our_bias = {key : [] for key in parameter_triples}
	for year in range(1900, 2000, 10):
		embeddings = all_embeddings[year]
		gender_A_list = [male - missing_words, female - missing_words]
		prior_distribution = get_uniform_distribution(2)
		for parameter_descriptor, (distance, normalizer, similarity) in parameter_triples.items():
			our_estimate = compute_our_bias(embeddings, gender_A_list, updated_gender_census_targets, normalizer, distance, similarity, prior_distribution, verbose=False)
			historical_our_bias[parameter_descriptor].append(our_estimate)

	write_averaged_historical_bias('gender', historical_gender_bias, historical_prior_bias, historical_our_bias)
	write_averaged_historical_correlations('gender_correlations', historical_gender_bias, historical_prior_bias, historical_our_bias)

	historical_race_bias = []
	for year in range(1900, 2000, 10):
		bias = 0.0
		for profession in updated_race_census_targets:
			bias += census_bias(prof2race_ratio[profession][year])
		bias = bias / len(updated_race_census_targets)
		historical_race_bias.append(bias)

	prior_keys = ['manzini']
	historical_prior_bias = {key : [] for key in prior_keys}
	for year in range(1900, 2000, 10):
		embeddings = all_embeddings[year]
		race_A_list = [white - missing_words, hispanic - missing_words, asian - missing_words]
		unaligned_estimates = compute_all_unaligned_estimates(embeddings, race_A_list, updated_race_census_targets, abs, verbose = False)
		for key in unaligned_estimates:
			historical_prior_bias[key].append(unaligned_estimates[key])

	historical_our_bias = {key : [] for key in parameter_triples}
	for year in range(1900, 2000, 10):
		embeddings = all_embeddings[year]
		race_A_list = [white - missing_words, hispanic - missing_words, asian - missing_words]
		prior_distribution = get_uniform_distribution(3)
		for parameter_descriptor, (distance, normalizer, similarity) in parameter_triples.items():
			our_estimate = compute_our_bias(embeddings, race_A_list, updated_race_census_targets, normalizer, distance, similarity, prior_distribution, verbose=False)
			historical_our_bias[parameter_descriptor].append(our_estimate)
	
	write_averaged_historical_bias('race', historical_race_bias, historical_prior_bias, historical_our_bias)
	write_averaged_historical_correlations('race_correlations', historical_race_bias, historical_prior_bias, historical_our_bias)



if __name__ == '__main__':
	averaged_historical_bias()
	finegrained_census_bias()