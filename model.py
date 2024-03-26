from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pickle import dump, load
import pandas as pd


def split_data(df: pd.DataFrame):
    y = df['Outcome']
    X = df.drop(columns=['Outcome'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test


def open_data(path="data/diabetes.csv"):
    df = pd.read_csv(path)
    df = df[['Outcome', "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI","DiabetesPedigreeFunction", "Age"]]

    return df


def preprocess_data(df: pd.DataFrame, test=True):
    df = df.dropna()
    X_train, X_test, y_train, y_test = split_data(df)
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, X_test, y_train, y_test



def fit_and_save_model(X_train, y_train, X_test, y_test, path="data/model_weights.mw"):
    model = KNeighborsClassifier()
    
    model.fit(X_train, y_train)
    test_prediction = model.predict(X_test)
    accuracy = accuracy_score(test_prediction, y_test)
    
    print(f"Model accuracy is {accuracy:.2f}")
    
    with open(path, "wb") as file:
        dump(model, file)
    
    print(f"Model was saved to {path}")


def load_model_and_predict(df, path="data/model_weights.mw"):
    with open(path, "rb") as file:
        model = load(file)

    prediction = model.predict(df)[0]
 
    prediction_proba = model.predict_proba(df)[0]


    encode_prediction_proba = {
        0: "Вам повезло с вероятностью",
        1: "Вам не повезло с вероятностью"
    }

    encode_prediction = {
        0: "Вероятность диабета низкая",
        1: "Вероятность диабета высокая"
    }

    prediction_data = {}
    for key, value in encode_prediction_proba.items():
        prediction_data.update({value: prediction_proba[key]})

    prediction_df = pd.DataFrame(prediction_data, index=[0])
    prediction = encode_prediction[prediction]

    return prediction, prediction_df


if __name__ == "__main__":
    df = open_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    fit_and_save_model(X_train, y_train, X_test, y_test)
