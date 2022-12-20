from primitive_operations import *
import numpy as np


parameter_triples = {}
descriptor2distance = {"L1" : L1_distance, "L2" : L2_distance, "Linf" : Linfinity_distance, "KL": KL_divergence, "JS": JS_divergence}
descriptor2normalizer = {"sum" : sum_normalization, "softmax" : softmax}
descriptor2similarity = {"cosine" : cosine, "inner_product" : inner_product}
for distance_descriptor, distance in descriptor2distance.items():
	for normalizer_descriptor, normalizer in descriptor2normalizer.items():
		for similarity_descriptor, similarity in descriptor2similarity.items():
			parameter_triples[(distance_descriptor, normalizer_descriptor, similarity_descriptor)] = (distance, normalizer, similarity)


def compute_our_bias_given_SoA(association_vector, normalizer, distance, prior_distribution):
	association_vector, prior_distribution = np.array(association_vector), np.array(prior_distribution)
	association_distribution = normalizer(association_vector)
	bias_score = distance(association_distribution, prior_distribution)
	difference_vector = association_distribution - prior_distribution
	return {'magnitude' : bias_score, 'direction' : difference_vector}


# Estimator introduced in this work (hence the descriptor "our")
# Multiclass, unaligned bias estimator
# [embeddings] - Map from words to vectors
# [A_list] - List of list of words. Each inner list designates a social group (e.g. white, black, asian)
# [target_words] - The words bias will be computed with respect to (e.g. professions)
# [distance] - distance function between prior distribution and embedding-derived distribution
# [similarity] - Similarity function used in computing (generally cosine simalrity or inner product)
# [prior_distribution] - Distribution from which bias estimates are taken respect to; uniform distribution encodes all social groups are "valued" equally
# [verbose] - Additionally return per-word bias scores
def compute_our_bias(embeddings,
						A_list,
						target_words,
						normalizer,
						distance,
						similarity,
						prior_distribution,
						verbose=False):
	bias_per_target_word = {}
	for target_word in target_words:
		target_embedding = embeddings[target_word]
		group_embeddings = [average([embeddings[attribute_word] for attribute_word in Ai]) for Ai in A_list]
		score_vector = [similarity(target_embedding, group_embedding) for group_embedding in group_embeddings]
		embedding_distribution = normalizer(score_vector)
		target_word_bias = distance(embedding_distribution, prior_distribution)
		difference_vector = embedding_distribution - prior_distribution
		bias_per_target_word[target_word] = {'magnitude' : target_word_bias, 'direction' : difference_vector}
	averaged_bias = average([v['magnitude'] for v in bias_per_target_word.values()])
	if verbose:
		return averaged_bias, bias_per_target_word
	else:
		return averaged_bias