import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


def predictResult(answers):
# Import the datasets
    datasets_train = pd.read_csv("train_data9010.csv")
    X_Train = datasets_train.iloc[:, [3, 4, 5, 6]].values
    Y_Train = datasets_train.iloc[:, 14].values

    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)
    X_Test = sc_X.transform([answers])

    # Fitting the classifier into the Training set
    # classifier = SVC(kernel='rbf', random_state=0)
    # classifier.fit(X_Train, Y_Train)

    classifier = joblib.load('svmModel.pkl')

    Y_Pred = classifier.predict(X_Test)
    print(Y_Pred)
    return Y_Pred