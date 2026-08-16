from sklearn.linear_model import LogisticRegression


def create_model():
    """Create the Logistic Regression classifier."""
    return LogisticRegression(max_iter=1000, random_state=42)
