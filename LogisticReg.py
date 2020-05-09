from pandas.plotting import scatter_matrix
import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB

# def load_dataset():
url = "prsaf1.csv"
names = ['pm2.5', 'dew_point', 'temperature', 'pressure', 'windspeed', 'result']
dataset = pandas.read_csv(url, names=names)

# Split-out validation dataset
array = dataset.values
X = array[:, 0:5]
Y = array[:, 5]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size,
                                                                                random_state=seed)
seed = 7
scoring = 'accuracy'


def display():
    # load_dataset()
    # shape

    dis1 = dataset.shape
    # column names

    dis2 = dataset.columns
    # head

    dis3 = str(dataset.head(2))
    # tail

    dis4 = dataset.tail(2)
    # descriptions

    dis5 = dataset.describe()

    # box and whisker plots
    dataset.plot(kind='box', subplots=True)
    plt.savefig("/Users/chandanadeshmukh/fy/frn/static/box.png")
    plt.close()

    # histograms
    dataset.hist(figsize=(8, 8))
    plt.savefig("/Users/chandanadeshmukh/fy/frn/static/hist.png")
    plt.close()

    # scatter plot matrix
    scatter_matrix(dataset[['pm2.5', 'dew_point', 'temperature', 'pressure', 'windspeed']], figsize=(8, 8))
    plt.savefig("/Users/chandanadeshmukh/fy/frn/static/scat.png")
    plt.close()

    return [dis1, dis2, dis3, dis4, dis5]


def algo_comp():
    # Spot Check Algorithms
    models = [('LR', LogisticRegression()), ('LDA', LinearDiscriminantAnalysis()), ('KNN', KNeighborsClassifier()),
              ('CART', DecisionTreeClassifier()), ('NB', GaussianNB())]
    # evaluate each model in turn
    results = []
    names = []

    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)

    # Compare Algorithms
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.savefig("/Users/chandanadeshmukh/fy/frn/static/algo.png")
    plt.close()


def log_reg(pm, dew_point, temperature, pressure, windspeed):
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression()))
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(LogisticRegression(), X_train, Y_train, cv=kfold, scoring=scoring)
    example_measures = np.array([pm, dew_point, temperature, pressure, windspeed])
    example_measures = example_measures.reshape(1, -1)
    lr = LogisticRegression()
    lr.fit(X_train, Y_train)
    predictions = lr.predict(X_validation)
    prediction = float(lr.predict(example_measures))
    return prediction
