import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model(data_path="iris.csv"):
    df = pd.read_csv(data_path)
    X = df.drop("Species", axis=1)
    y = df["Species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def predict(model, data):
    return model.predict(data)

if __name__ == "__main__":
    model = train_model()
    joblib.dump(model, "model.joblib") # Save the trained model