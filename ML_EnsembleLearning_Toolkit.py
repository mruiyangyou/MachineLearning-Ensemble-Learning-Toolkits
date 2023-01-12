from sklearn import *
from sklearn.model_selection import cross_validate
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# build a data frame for having an overview of algothrims
MLA = [
    # Ensemble Methods
    ensemble.AdaBoostClassifier(),
    ensemble.BaggingClassifier(),
    ensemble.ExtraTreesClassifier(),
    ensemble.GradientBoostingClassifier(),
    ensemble.RandomForestClassifier(),

    # fier(),
    #     ensemble.Gaussian Processes
    gaussian_process.GaussianProcessClassifier(),

    # GLM
    linear_model.LogisticRegressionCV(),
    linear_model.PassiveAggressiveClassifier(),
    linear_model.RidgeClassifierCV(),
    linear_model.SGDClassifier(),
    linear_model.Perceptron(),

# Navies Bayes
    naive_bayes.BernoulliNB(),
    naive_bayes.GaussianNB(),

    # Nearest Neighbor
    neighbors.KNeighborsClassifier(),

    # SVM
    svm.SVC(probability=True),
    svm.NuSVC(probability=True),
    svm.LinearSVC(),

    # Trees
    tree.DecisionTreeClassifier(),
    tree.ExtraTreeClassifier(),

    # Discriminant Analysis
    discriminant_analysis.LinearDiscriminantAnalysis(),
    discriminant_analysis.QuadraticDiscriminantAnalysis()]

cv_split = model_selection.ShuffleSplit(n_splits=10,test_size=0.3, train_size=0.6)

def mla_dataframe(x_train, y_train, x_valid,vectorizer = None, nlp = False,
                  scaler = False, scale = None):
    if nlp == True:
        vectorizer.fit(list(x_train)+list(x_valid))
        x_train = vectorizer.transform(x_train)

    if scaler == True:
        x_train = scale.fit_transofrm(x_train)

    MLA_cols = ['MLA_Name', 'Parameters', 'MLA training accuray', 'MLA valid accuracy', 'MLA accuracy * std',
                'MLA time']
    mla_dataframe = pd.DataFrame(columns=MLA_cols)
    row_index = 0
    for alg in MLA:
        mla_dataframe.loc[row_index, 'MLA_Name'] = alg.__class__.__name__
        mla_dataframe.loc[row_index, 'Parameters'] = str(alg.get_params())
        cv_result = cross_validate(alg, x_train, y_train, cv=cv_split, scoring='roc_auc', return_train_score=True)
        mla_dataframe.loc[row_index, 'MLA training accuracy'] = cv_result['train_score'].mean()
        mla_dataframe.loc[row_index, 'MLA valid accuracy'] = cv_result['test_score'].mean()
        mla_dataframe.loc[row_index, 'MLA accuracy * std'] = cv_result['test_score'].std() * 3
        mla_dataframe.loc[row_index, 'MLA time'] = cv_result['fit_time'].mean()
        row_index += 1

    MLA_compare = mla_dataframe.sort_values(by='MLA valid accuracy', ascending=False)
    print(MLA_compare)

    plt.figure(dpi=30)
    sns.barplot(x='MLA valid accuracy', y='MLA_Name', data=MLA_compare, color='m')
    plt.show()

    return MLA_compare


###################### Next step is fit the hyperparameter by grid seach
cv_split = model_selection.ShuffleSplit(n_splits = 10, test_size = .3, train_size = .6, random_state = 0 )

def fine_tunning(model, params, names, x_train, y_train, scoring):
    cv_split = model_selection.ShuffleSplit(n_splits=10, test_size=.3, train_size=.6, random_state=0)
    base_results = model_selection.cross_validate(model, x_train, y_train, cv=cv_split,)
    print('Before', names, 'the parameter:', model.get_params())
    print('Before training score mean: {:.2f}'.format(base_results['test_score'].mean()))
    print('-' * 10)
    tune_model = GridSearchCV(model, param_grid=params, scoring=scoring, cv=cv_split)
    tune_model.fit(x_train,y_train)
    print('After', names, 'the best parameters:', tune_model.best_params_)
   #print('the training score: {:.2f}'.format(tune_model.cv_results_['mean_train_score']))
    print('the best index', tune_model.best_index_)
    print('the test score: {:.2f}'.format(tune_model.cv_results_['mean_test_score']))
    print('-' * 10)
    return tune_model.best_estimator_

# apply the function to a for loop with their parameters

##################### feature selection

def feature_selection(model_selection, x_train, y_train, df_x, model_classfication):
    model_selection.fit(x_train,y_train) # choose features
    x_train_new = df_x.columns.values[model_selection.get_support()] # to get the features
    rfe_results = model_selection.cross_validate(model_classfication,
                                                 x_train_new, y_train, cv=cv_split)  # corss vaidate to see the improved result
    print("AFTER DT RFE Test w/bin score mean: {:.2f}".format(rfe_results['test_score'].mean() * 100))
    print("AFTER DT RFE Test w/bin score 3*std: +/- {:.2f}".format(rfe_results['test_score'].std() * 100 * 3))
    print('-' * 10)


##################### model ensembling

# voting classifier
def performance_clf(y, y_predict, multiclass = None, average = None):
    #print('accuracy score: %0.4lf' % (metrics.accuracy_score(y, y_predict,multi_class=multiclass, average=average)))
    #print('precision score: %0.4lf'% (metrics.precision_score(y, y_predict,multi_class=multiclass, average=average)))
    #print('recall score: %0.4lf' % (metrics.recall_score(y, y_predict,multi_class=multiclass, average=average)))
    print('auc: %0.4lf' % (metrics.roc_auc_score(y, y_predict,multi_class=multiclass, average=average)))

def voting(vote_set, voting,x_train,y_train):
    vote = ensemble.VotingClassifier(vote_set, voting)
    vote_cv = model_selection.cross_validate(vote, x_train, y_train, cv  = cv_split, return_train_score = True)
    print('{} Working training score: {:.2f}'.format(voting, vote_cv['train_score'].mean()))
    print('{} Working testing score: {:.2f}'.format(voting, vote_cv['test_score'].mean()))

def valid_vote(vote,x_valid,y_valid):
    y_predict = vote.predit(x_valid)
    performance_clf(y_valid,y_predict)

# stacking classification with blending
import numpy as np
def classfication_bledning(x,y,clf_list,split):
    x,y, x_valid, y_valid = model_selection.train_test_split(x,y,test_size=0.3)
    n_splits = split
    skf = model_selection.StratifiedKFold(n_splits)
    data_blend_train = np.zeros((x.shape[0], len(clf_list)))
    data_blend_valid = np.zeros((x_valid.shape[0], len(clf_list)))

    for j, clf in enumerate(clf_list):
        data_blend_test_j = np.zeros((x_valid.shape[0],split))
        for i, (train,test) in enumerate(skf):
            x_train, y_train, x_test, y_test = x[train], y[train], x[test], y[test]
            clf.fit(x_train,y_train)
            y_submisiion = clf.predict(x_test)
            data_blend_train[test,j] = y_submisiion
            data_blend_test_j[:,i] = clf.predict(x_valid)
        data_blend_valid[:,j] = data_blend_test_j.mean(1)
        print('Val auc score: %.2f' % metrics.roc_auc_score(y_valid, data_blend_valid[:,j]), clf_list[j])

    stacking = linear_model.LogisticRegression()
    stacking.fit(data_blend_train, y)
    y_predict = stacking.predict(data_blend_valid)
    performance_clf(y_valid, y_predict)

#mean regression
def mean_stacking(test_predict,y_valid):
    mean_result = test_predict.mean(axis = 1)
    print('The mean square error: %.2f'  % metrics.mean_absolute_error(y_valid ,mean_result))

# fit regression
import pandas as pd
from sklearn import linear_model
def Stacking_method(model_L2,train_reg1,train_reg2,train_reg3,y_train_true,test_pre1,test_pre2,test_pre3, y_valid):
    model_L2.fit(pd.concat([pd.Series(train_reg1),pd.Series(train_reg2),pd.Series(train_reg3)],axis=1).values,y_train_true)
    Stacking_result = model_L2.predict(pd.concat([pd.Series(test_pre1),pd.Series(test_pre2),pd.Series(test_pre3)],axis=1).values)
    print('The mean square error: %.2f' % metrics.mean_absolute_error(y_valid, Stacking_result))
    return Stacking_result





