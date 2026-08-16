from sklearn.tree import DecisionTreeClassifier


def create_model():
    """Create the Decision Tree classifier."""
    return DecisionTreeClassifier(random_state=42)
