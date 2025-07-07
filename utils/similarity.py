from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(user_vector, item_matrix):
    """
    user_vector: 1D array of shape (n_features,)
    item_matrix: 2D array of shape (n_items, n_features)
    returns: 1D array of similarity scores of length n_items
    """
    return cosine_similarity([user_vector], item_matrix)[0] 