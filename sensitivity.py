from our_word_list import *
from primitive_operations import *
from face_validity import *
from tqdm import tqdm
from random import sample 
from sklearn.metrics import r2_score
from scipy.stats import linregress, spearmanr


def generate_inputs():
	inputs = {}
	inputs['default'] = {'male_words' : male, 'female_words' : female, 'normalizer' : sum_normalization, 'distance' : L1_distance}
	length_3_male, length_3_female = set(sample(male, 3)), set(sample(female, 3))
	length_5_male, length_5_female = set(sample(male, 5)), set(sample(female, 5)) 
	# inputs['L3'] = {'male_words' : length_3_male, 'female_words' : length_3_female, 'normalizer' : sum_normalization, 'distance' : L1_distance}
	# inputs['L5'] = {'male_words' : length_5_male, 'female_words' : length_5_female, 'normalizer' : sum_normalization, 'distance' : L1_distance}
	# inputs['names'] = {'male_words' : male_names, 'female_words' : female_names, 'normalizer' : sum_normalization, 'distance' : L1_distance}
	# inputs['ell-2'] = {'male_words' : male, 'female_words' : female, 'normalizer' : sum_normalization, 'distance' : L2_distance}
	# inputs['ell-inf'] = {'male_words' : male, 'female_words' : female, 'normalizer' : sum_normalization, 'distance' : Linfinity_distance}
	inputs['softmax'] = {'male_words' : male, 'female_words' : female, 'normalizer' : softmax, 'distance' : L1_distance}
	# ordered_columns = ['L3', 'L5', 'names', 'ell-2', 'ell-inf', 'softmax']
	ordered_columns = ['softmax']
	return inputs, ordered_columns


def sensitivity_results(inputs, ordered_rows, ordered_columns, ordered_targets, vocab):
	results = {}
	default_params = inputs['default']
	male_words, female_words, normalizer, distance = default_params['male_words'], default_params['female_words'], default_params['normalizer'], default_params['distance']
	default_results = face_validity_results(male_words, female_words, vocab, normalizer, distance)

	for alternative_name in tqdm(ordered_columns):
		alternative_params = inputs[alternative_name]
		male_words, female_words, normalizer, distance = alternative_params['male_words'], alternative_params['female_words'], alternative_params['normalizer'], alternative_params['distance']
		alternative_results = face_validity_results(male_words, female_words, vocab, normalizer, distance)
		for setting_name in ordered_rows:
			default_list = [default_results[target][setting_name] for target in ordered_targets]
			alternative_list = [alternative_results[target][setting_name] for target in ordered_targets]
			slope, intercept, r_value, linear_p_value, std_err = linregress(default_list, alternative_list)
			rho, spearman_p_value = spearmanr(default_list, alternative_list)
			results[(setting_name, alternative_name)] = {'rho' : rho, 'spearman_p' : spearman_p_value, 'R2': r_value ** 2, 'linear_p': linear_p_value, 'std_err' : std_err}
	return results	


def write_sensitivity_results(results, ordered_rows, ordered_columns):
	file_name = cwd / "results" / "sensitivity_probing.table.tex"
	file = open(file_name, "w")
	file.write(' & ' + ' & '.join(ordered_columns) + ' \\\\ \n')
	for row in ordered_rows:
		line = row 
		for column in ordered_columns:
			line = line + ' & '
			line = line + '({}, {})'.format(round(results[(row, column)]['rho'], 2), round(results[(row, column)]['R2'], 2))
		line = line + ' \\\\ \n'
		file.write(line)

	file_name = cwd / "results" / "sensitivity-p_values_probing.table.tex"
	file = open(file_name, "w")
	file.write(' & ' + ' & '.join(ordered_columns) + ' \\\\ \n')
	for row in ordered_rows:
		line = row 
		for column in ordered_columns:
			line = line + ' & '
			line = line + '({}, {})'.format(round(results[(row, column)]['spearman_p'], 3), round(results[(row, column)]['linear_p'], 3))
		line = line + ' \\\\ \n'
		file.write(line)


if __name__ == '__main__':
	vocab = face_validity_targets | gender_all | race
	# _, _, missing_words = fetch_w2v_GloVe_embeddings(vocab, pickled=False)
	# print(missing_words)
	# exit()
	ordered_rows = ['contextualized-BERT-probing']
	inputs, ordered_columns = generate_inputs()
	results = sensitivity_results(inputs, ordered_rows, ordered_columns, ordered_face_validity_targets, vocab)
	print(results)
	write_sensitivity_results(results, ordered_rows, ordered_columns)
