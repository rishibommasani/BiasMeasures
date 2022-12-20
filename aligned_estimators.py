import numpy as np
import sklearn
from sklearn.decomposition import PCA
from primitive_operations import *


# Reference: https://github.com/tolga-b/debiaswe/blob/master/debiaswe/we.py
def compute_bolukbasi_bias_direction_bolukbasi_implementation(embeddings, seed_pairs):
	diff_embeddings = []
	for x, y in seed_pairs:
		center = (embeddings[x] + embeddings[y]) / 2
		diff_embeddings.append(embeddings[x] - center)
		diff_embeddings.append(embeddings[y] - center)
	X = np.array(diff_embeddings)
	pca = PCA(n_components=1)
	pca.fit(X)
	return pca.components_[0], pca.explained_variance_ratio_[0]


# Seems like the more conventional implementation than the above
# However, may not be desirable, since we may actually prefer to *not* center before running PCA in this setting
def compute_bolukbasi_bias_direction_conventional_implementation(embeddings, seed_pairs):
	diff_embeddings = [embeddings[x] - embeddings[y] for x, y in seed_pairs]
	X = np.array(diff_embeddings)
	pca = PCA(n_components=1)
	pca.fit(X)
	return pca.components_[0], pca.explained_variance_ratio_[0]


# Reference: Bolukbasi et al., 2016 - https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings
# Binary, aligned bias estimator
# [embeddings] - Map from words to vectors
# [bias direction] - First principal component of bias subspace
# [target_words] - The words bias will be computed with respect to (e.g. professions)
# [pooling_operation] - Generally abs(); absolute value encodes intuition that if X is male biased and Y is female-biased, these bias should not "cancel"
# [verbose] - Additionally return per-word bias scores
def compute_bolukbasi_bias(embeddings, bias_direction, target_words, pooling_operation, verbose = False):
	bias_per_target_word = {}
	for target_word in target_words:
		if target_word in embeddings.keys():
			bias_per_target_word[target_word] = pooling_operation(cosine(embeddings[target_word], bias_direction))
	averaged_bias = average(list(bias_per_target_word.values()))
	if verbose:
		return averaged_bias, bias_per_target_word
	else:
		return averaged_bias
	

# Reference: Ethayarajh et al., 2019 - https://www.aclweb.org/anthology/P19-1166/
# Binary, aligned bias estimator
# [embeddings] - Map from words to vectors
# [bias direction] - First principal component of bias subspace
# [target_words] - The words bias will be computed with respect to (e.g. professions)
# [pooling_operation] - Generally abs(); absolute value encodes intuition that if X is male biased and Y is female-biased, these bias should not "cancel"
# [verbose] - Additionally return per-word bias scores
def compute_ethayarajh_bias(embeddings, bias_direction, target_words, pooling_operation, verbose = False):
	bias_per_target_word = {}
	for target_word in target_words:
		if target_word in embeddings.keys():
			bias_per_target_word[target_word] = pooling_operation(inner_product(embeddings[target_word], bias_direction))
	averaged_bias = average(list(bias_per_target_word.values()))
	if verbose:
		return averaged_bias, bias_per_target_word
	else:
		return averaged_bias


def compute_all_aligned_estimates(embeddings, seed_pairs, target_words, pooling_operation, verbose = False):
	bias_direction_bolukbasi_implementation, _ = compute_bolukbasi_bias_direction_bolukbasi_implementation(embeddings, seed_pairs)
	bias_direction_conventional_implementation, _  = compute_bolukbasi_bias_direction_conventional_implementation(embeddings, seed_pairs)

	bolukbasi_bias_bolukbasi_implementation = compute_bolukbasi_bias(embeddings, bias_direction_bolukbasi_implementation, target_words, pooling_operation, verbose = verbose)
	bolukbasi_bias_conventional_implementation = compute_bolukbasi_bias(embeddings, bias_direction_conventional_implementation, target_words, pooling_operation, verbose = verbose)
	
	ethayarajh_bias_bolukbasi_implementation = compute_ethayarajh_bias(embeddings, bias_direction_bolukbasi_implementation, target_words, pooling_operation, verbose = verbose)
	ethayarajh_bias_conventional_implementation = compute_ethayarajh_bias(embeddings, bias_direction_conventional_implementation, target_words, pooling_operation, verbose = verbose)

	return {
	# 'bolukbasi-bolukbasi' : bolukbasi_bias_bolukbasi_implementation,
	'bolukbasi-conventional' : bolukbasi_bias_conventional_implementation,
	# 'ethayarajh-bolukbasi' : ethayarajh_bias_bolukbasi_implementation,
	'ethayarajh-conventional' : ethayarajh_bias_conventional_implementation
	}