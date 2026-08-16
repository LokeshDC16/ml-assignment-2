from sklearn.ensemble import RandomForestClassifier


def create_model():
    """Create the Random Forest ensemble classifier."""
    return RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
