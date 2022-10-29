import sklearn.preprocessing
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd

def model_columns(train,validate,test):
    # features that will be used for x columns
    features = ['tech_support_Yes','contract_type_One year', 'contract_type_Two year','internet_service_type_Fiber optic']
    
    # setting the x and y values for my train, validate sets 
    X_train = train[features]
    y_train = train['has_churned']

    X_validate = validate[features]
    y_validate = validate['has_churned']

    X_test = test[features]
    y_test = test['has_churned']

    
    return X_train, X_validate, y_train, y_validate, X_test, y_test

def baseline(train):
    ''' this function will generate the baseline model and print its performance '''
    y_train = train['has_churned']
    baseline = y_train.mode()

    # creating boolean to match where y_train is no 
    matches_baseline_prediction = (y_train == 0)

    #calculating baseline and printing it
    baseline_accuracy = matches_baseline_prediction.mean()
    print(f"Baseline accuracy: {round(baseline_accuracy, 2)}")

def decision_tree_model(X_train, X_validate, y_train, y_validate):
    '''this function will create my decision tree model and print its performance'''

    # Make the model
    tree1 = DecisionTreeClassifier(max_depth=3, random_state=100)

    # Fit the model (on train and only train)
    tree_train = tree1.fit(X_train, y_train)

    # Fit the model (on validate and only validate)
    tree_validate = tree1.fit(X_validate, y_validate)

    # print out of evaluation results 
    print('Accuracy of Decision Tree classifier on training set: {:.2f}'
      .format(tree1.score(X_train, y_train)))

    print('Accuracy of Decision Tree classifier on validate set: {:.2f}'
      .format(tree1.score(X_validate, y_validate)))
