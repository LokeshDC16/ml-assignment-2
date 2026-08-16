from sklearn.neighbors import KNeighborsClassifier


def create_model():
    """Create the K-Nearest Neighbors classifier."""
    return KNeighborsClassifier(n_neighbors=5)
