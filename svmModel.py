import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from sklearn.metrics import accuracy_score

def train_svm_model(train_data_path, model_save_path):
    # Import the datasets
    datasets_train = pd.read_csv(train_data_path)
    X_Train = datasets_train.iloc[:, [3, 4, 5, 6]].values
    Y_Train = datasets_train.iloc[:, 14].values

    # datasets_test = pd.read_csv("test_data9010.csv")
    # X_Test = datasets_test.iloc[:, [3, 4, 5, 6]].values
    # Y_Test = datasets_test.iloc[:, 14].values

    # Feature Scaling
    sc_X = StandardScaler()
    X_Train = sc_X.fit_transform(X_Train)

    # Fitting the classifier into the Training set
    classifier = SVC(kernel='rbf', random_state=0)
    classifier.fit(X_Train, Y_Train)

    # X_Test = sc_X.transform(X_Test)
    # Y_Pred = classifier.predict(X_Test)
    # print(accuracy_score(Y_Test, Y_Pred))

    # Save the trained model
    joblib.dump(classifier, model_save_path)

    return classifier

if __name__ == '__main__':
    train_svm_model("train_data9010.csv", "svmModel.pkl")
