from sklearn.ensemble import RandomForestClassifier

def get_model(n_estimators=100, max_depth=None):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
