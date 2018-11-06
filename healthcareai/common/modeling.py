from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier
from sklearn.ensemble     import GradientBoostingClassifier
from sklearn.naive_bayes  import GaussianNB
from sklearn.tree         import DecisionTreeClassifier
from sklearn.neighbors    import KNeighborsClassifier
from sklearn.svm          import SVC

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from operator import itemgetter, attrgetter
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

## using "Recursive Feature Elimination" RFE as the feature selection method.
from sklearn.feature_selection import RFECV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def kFold(dataframe, feature, output):
    X = dataframe[feature]
    y = dataframe.Outcome
    # Initial model selection process
    models = []
    models.append(('LR',  LogisticRegression(solver='lbfgs', max_iter=4000)))
    models.append(('RF',  RandomForestClassifier(n_estimators=100)))
    models.append(('GB',  GradientBoostingClassifier()))
    models.append(('GNB', GaussianNB()))
    models.append(('DT',  DecisionTreeClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVC', SVC(gamma=0.001)))

    strat_k_fold = StratifiedKFold(n_splits=10, random_state=10)

    names = []
    scores = []
    algorithms = []

    for name, model in models:
        score = cross_val_score(model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
        names.append(name)
        algorithms.append(model)
        scores.append(score)

    kf_cross_val = pd.DataFrame({'Name': names, 'Algorithms': algorithms, 'Score': scores})
    print(kf_cross_val)
    return strat_k_fold

def modelSelectionByKfold(dataframe, outputDataFrame):
    # Initial model selection process
    models = []
    models.append(('LR',  LogisticRegression(solver='lbfgs', max_iter=4000)))
    models.append(('RF',  RandomForestClassifier(n_estimators=100)))
    models.append(('GB',  GradientBoostingClassifier()))
    models.append(('GNB', GaussianNB()))
    models.append(('DT',  DecisionTreeClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVC', SVC(gamma=0.001)))

    # Using K-Fold cross validation
    strat_k_fold = StratifiedKFold(n_splits=10, random_state=10)

    names = []
    scores = []
    algorithms = []

    for name, model in models:
        score = cross_val_score(model, dataframe, outputDataFrame, cv=strat_k_fold, scoring='accuracy').mean()
        names.append(name)
        algorithms.append(model)
        scores.append(score)

    kf_cross_val = pd.DataFrame({'Name': names, 'Algorithm': algorithms, 'Score': scores})
    # show plot chart
    # Plot the accuracy scores using "seaborn"
    axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
    axis.set(xlabel='Classifier Algorithm', ylabel='Accuracy')

    for p in axis.patches:
        height = p.get_height()
        axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
        
    plt.show()
    kf_cross_val = kf_cross_val.sort_values(by='Score', ascending=False)
    return kf_cross_val

def train_test(dataframe, feature, output):
    X = dataframe[feature]
    y = dataframe.output

    # Initial model selection process
    models = []
    models.append(('LR',  LogisticRegression(solver='lbfgs', max_iter=4000)))
    models.append(('RF',  RandomForestClassifier(n_estimators=100)))
    models.append(('GB',  GradientBoostingClassifier()))
    models.append(('GNB', GaussianNB()))
    models.append(('DT',  DecisionTreeClassifier()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('SVC', SVC(gamma=0.001)))
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = dataframe.output, random_state=0)

    names = []
    scores = []

    for name, model in models:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
        names.append(name)

    tr_split = pd.DataFrame({'Name': names, 'Score': scores})
    return tr_split

def featureSelectionByModel(dataframe, outcome, feature_names, dfModels):
    # dfModels = modelSelectionByKfold(dataframe, outcome)
    strat_k_fold = StratifiedKFold(n_splits=10, random_state=10)
    scores = []
    models = []
    modelFeatures = []
    for index, row in dfModels.iterrows():
        modelSelection = row['Algorithm']
        print('model', modelSelection)
        rfecv = RFECV(estimator=modelSelection, step=1, cv=strat_k_fold, scoring='accuracy')
        rfecv.fit(dataframe, outcome)

        plt.figure()
        plt.title(row['Name'] + 'score vs No of Features')
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
        plt.show()

        feature_importance = list(zip(feature_names, rfecv.support_))

        new_features = []

        for key, value in enumerate(feature_importance):
            if(value[1]) == True:
                new_features.append(value[0])
                
        print(new_features)
        # Calculate accuracy scores
        X_new = dataframe[new_features]

        initial_score = cross_val_score(modelSelection, dataframe, outcome, cv=strat_k_fold, scoring='accuracy').mean()
        print("Initial accuracy : {} ".format(initial_score))

        fe_score = cross_val_score(modelSelection, X_new, outcome, cv=strat_k_fold, scoring='accuracy').mean()
        print("Accuracy after Feature Selection : {} ".format(fe_score))

        importFeature = {'score': fe_score - initial_score, 'feature': new_features, 'model': rfecv}
        modelFeatures.append(importFeature)

    sortedImportFeature = sorted(modelFeatures, key=lambda obj: obj['score'], reverse=True)
    # choose feature with max score
    sortedImportFeature = sortedImportFeature[0]
    featureImportances = pd.DataFrame({'features': sortedImportFeature['feature'], 'model': sortedImportFeature['model']})
    return featureImportances

def featureImportance(dataframe, feature_names, output, model):
    dataframe_mod  = clean.clean_data(dataframe)
    X = dataframe_mod[feature_names]
    y = dataframe_mod.output
    strat_k_fold = StratifiedKFold(n_splits=10, random_state=10)
    rfecv = RFECV(estimator=model, step=1, cv=strat_k_fold, scoring='accuracy')
    rfecv.fit(X, y)

    plt.figure()
    plt.title('Logistic Regression CV score vs No of Features')
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()

    feature_importance = list(zip(feature_names, rfecv.support_))

    new_features = []

    for key, value in enumerate(feature_importance):
        if(value[1]) == True:
            new_features.append(value[0])
            
    print(new_features)

    # Calculate accuracy scores 
    X_new = dataframe[new_features]

    initial_score = cross_val_score(model, X, y, cv=strat_k_fold, scoring='accuracy').mean()
    print("Initial accuracy : {} ".format(initial_score))

    fe_score = cross_val_score(model, X_new, y, cv=strat_k_fold, scoring='accuracy').mean()
    print("Accuracy after Feature Selection : {} ".format(fe_score))
    return X_new

def modelTuning(dataframe, feature_names, output, model):
    importanceFeature = featureImportance(dataframe, feature_names, output, model)
    strat_k_fold = StratifiedKFold(n_splits=10, random_state=10)
    # Specify parameters
    c_values = list(np.arange(1, 10))

    param_grid = [
        {'C': c_values, 'penalty': ['l1'], 'solver' : ['liblinear'], 'multi_class' : ['ovr']},
        {'C': c_values, 'penalty': ['l2'], 'solver' : ['liblinear', 'newton-cg', 'lbfgs'], 'multi_class' : ['ovr']}
    ]
    ## fit the data to the GridSearchCV, which performs a K-fold cross validation on the data for the given combinations of the parameters.
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=strat_k_fold, scoring='accuracy', iid=False)
    grid.fit(importanceFeature, output)
    ## After training & scoring, GridSearchCV provides some useful attributes to find the best parameters and the best estimator.
    print(grid.best_params_)
    print(grid.best_estimator_)
    ## feed the best parameters to the Logistic Regression model and observe whether it’s accuracy has increased.
    logreg_new = LogisticRegression(grid.best_params_)
    initial_score = cross_val_score(logreg_new, importanceFeature, output, cv=strat_k_fold, scoring='accuracy').mean()
    print("Final accuracy : {} ".format(initial_score))
