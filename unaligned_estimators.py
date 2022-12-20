import numpy as np
from primitive_operations import *


# Reference: Garg et al., 2018 - https://www.pnas.org/content/115/16/E3635
# Binary, unaligned bias estimator
# [embeddings] - Map from words to vectors
# [A1], [A2] - Lists of words for the two social groups of interest (e.g. male, female)
# [target_words] - The words bias will be computed with respect to (e.g. professions)
# [pooling_operation] - Generally abs(); absolute value encodes intuition that if X is male biased and Y is female-biased, these bias should not "cancel"
# [verbose] - Additionally return per-word bias scores
def compute_garg_cosine_bias(embeddings,
							 A1,
							 A2,
							 target_words,
							 pooling_operation,
							 verbose=False):
	mu_1, mu_2 = average([embeddings[w] for w in A1]), average([embeddings[w] for w in A2])
	bias_per_target_word = {}
	for target_word in target_words:
		target_embedding = embeddings[target_word]
		target_word_bias = pooling_operation(cosine(target_embedding, mu_1) - cosine(target_embedding, mu_2))
		bias_per_target_word[target_word] = target_word_bias
	averaged_bias = average(list(bias_per_target_word.values()))
	if verbose:
		return averaged_bias, bias_per_target_word
	else:
		return averaged_bias


# Reference: Garg et al., 2018 - https://www.pnas.org/content/115/16/E3635
# Binary, unaligned bias estimator
# [embeddings] - Map from words to vectors
# [A1], [A2] - Lists of words for the two social groups of interest (e.g. male, female)
# [target_words] - The words bias will be computed with respect to (e.g. professions)
# [pooling_operation] - Generally abs(); absolute value encodes intuition that if X is male biased and Y is female-biased, these bias should not "cancel"
# [verbose] - Additionally return per-word bias scores
def compute_garg_euclidean_bias(embeddings,
								A1,
								A2,
								target_words,
								pooling_operation,
								verbose=False):
	mu_1, mu_2 = average([embeddings[w] for w in A1]), average([embeddings[w] for w in A2])
	bias_per_target_word = {}
	for target_word in target_words:
			target_embedding = embeddings[target_word]
			target_word_bias = pooling_operation(np.linalg.norm(target_embedding - mu_1) - np.linalg.norm(target_embedding - mu_2))
			bias_per_target_word[target_word] = target_word_bias
	averaged_bias = average(list(bias_per_target_word.values()))
	if verbose:
		return averaged_bias, bias_per_target_word
	else:
		return averaged_bias


# Reference: Caliskan et al., 2017 - https://science.sciencemag.org/content/356/6334/183
# Binary, unaligned bias estimator
# [embeddings] - Map from words to vectors
# [A1], [A2] - Lists of words for the two social groups of interest (e.g. male, female)
# [target_words] - The words bias will be computed with respect to (e.g. professions)
# [pooling_operation] - Generally abs(); absolute value encodes intuition that if X is male biased and Y is female-biased, these bias should not "cancel"
# [verbose] - Additionally return per-word bias scores
def compute_caliskan_bias(embeddings,
							A1,
							A2,
							target_words,
							pooling_operation,
							verbose=False):
	bias_per_target_word = {}
	for target_word in target_words:
		target_embedding = embeddings[target_word]
		avg_sim_1 = average([cosine(target_embedding, embeddings[attribute_word]) for attribute_word in A1])
		avg_sim_2 = average([cosine(target_embedding, embeddings[attribute_word]) for attribute_word in A2])
		bias_per_target_word[target_word] = pooling_operation(avg_sim_1 - avg_sim_2)
	averaged_bias = average(list(bias_per_target_word.values()))
	if verbose:
		return averaged_bias, bias_per_target_word
	else:
		return averaged_bias


# Reference: Manzini et al., 2019 - https://www.aclweb.org/anthology/N19-1062/
# Multiclass, unaligned bias estimator
# [embeddings] - Map from words to vectors
# [A_list] - List of list of words. Each inner list designates a social group (e.g. white, black, asian)
# [target_words] - The words bias will be computed with respect to (e.g. professions)
# [pooling_operation] - Generally abs(); absolute value encodes intuition that if X is male biased and Y is female-biased, these bias should not "cancel"
# [verbose] - Additionally return per-word bias scores
def compute_manzini_bias(embeddings,
						 A_list,
						 target_words,
						 pooling_operation,
						 verbose=False):
	attribute_union = [a for Ai in A_list for a in Ai]
	bias_per_target_word = {}
	for target_word in target_words:
		target_embedding = embeddings[target_word]
		bias_per_target_word[target_word] = pooling_operation(average([cosine(target_embedding, embeddings[attribute_word]) for attribute_word in attribute_union]))
	averaged_bias = average(list(bias_per_target_word.values()))
	if verbose:
		return averaged_bias, bias_per_target_word
	else:
		return averaged_bias


def compute_all_unaligned_estimates(embeddings, A_list, target_words, pooling_operation, verbose = False):
	unaligned_estimates = {}
	if len(A_list) == 2:
		A1, A2 = A_list 
		garg_cosine_bias = compute_garg_cosine_bias(embeddings, A1, A2, target_words, pooling_operation, verbose=verbose)
		garg_euclidean_bias = compute_garg_euclidean_bias(embeddings, A1, A2, target_words, pooling_operation, verbose=verbose)
		caliskan_bias = compute_caliskan_bias(embeddings, A1, A2, target_words, pooling_operation, verbose=verbose)
		manzini_bias = compute_manzini_bias(embeddings, A_list, target_words, pooling_operation, verbose=verbose)
		unaligned_estimates['garg-cosine'] = garg_cosine_bias
		unaligned_estimates['garg-euclidean'] = garg_euclidean_bias
		unaligned_estimates['caliskan'] = caliskan_bias
		unaligned_estimates['manzini'] = manzini_bias
	else:
		manzini_bias = compute_manzini_bias(embeddings, A_list, target_words, pooling_operation, verbose=verbose)
		unaligned_estimates['manzini'] = manzini_bias
	return unaligned_estimates