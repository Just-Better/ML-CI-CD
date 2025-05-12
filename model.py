from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_and_predict():
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.3, random_state=42)
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return preds, y_test

def get_accuracy():
    preds, y_test = train_and_predict()
    return accuracy_score(y_test, preds)
