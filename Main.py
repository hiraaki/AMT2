from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

#carrega o arquivo do disco
data_frame = pd.read_csv("Sonar - Maurício.csv")
x = data_frame.iloc[:, 0:60].values
y = data_frame.iloc[:, 60].values


#função para medir a acurácia da predição
def accuracy(y_true: object, y_pred: object) -> object:
    return np.mean(np.equal(y_true, y_pred))


KnnAcur = []
TreeAcu = []
NbAcu = []
SvnACu = []
MlAcu = []
for i in range(20):
    train_x, aux_x, train_y, aux_y = train_test_split(x, y, test_size=0.5)
    validation_x, test_x, validation_y, test_y = train_test_split(aux_x, aux_y, test_size=0.5)
    nbrs = KNeighborsClassifier(n_neighbors=1, algorithm='auto')
    nbrs.fit(train_x, train_y)
    # print(nbrs.fit(train_x, train_y))
    # print(accuracy(test_y, nbrs.predict(test_x)))
    KnnAcur.append(accuracy(test_y, nbrs.predict(test_x)))

    decision = DecisionTreeClassifier(min_samples_leaf=6)
    decision = decision.fit(train_x, train_y)
    # print(accuracy(test_y, decision.predict(test_x)))
    TreeAcu.append(accuracy(test_y, decision.predict(test_x)))

    nb = BernoulliNB()
    nb.fit(train_x, train_y)
    NbAcu.append(accuracy(test_y, nb.predict(test_x)))

    svm = SVC(C=1.62, kernel='poly', gamma='scale')
    svm.fit(train_x, train_y)
    SvnACu.append(accuracy(test_y, svm.predict(test_x)))

    mlp = MLPClassifier(hidden_layer_sizes=(8,), random_state=1, learning_rate='constant', learning_rate_init=0.0007, max_iter=380)
    mlp.fit(train_x, train_y)
    MlAcu.append(accuracy(test_y, mlp.predict(test_x)))

with open("TableladeComparacao.csv", "w") as fp:

    fp.write("KNN; AD; NB; SVM; MLP\n")
    for index in range(20):
        fp.write("%f; %f; %f; %f; %f\n" % (KnnAcur[index], TreeAcu[index], NbAcu[index], SvnACu[index], MlAcu[index]))

    fp.write("%f; %f; %f; %f; %f\n" % (np.mean(KnnAcur), np.mean(TreeAcu), np.mean(NbAcu), np.mean(SvnACu), np.mean(MlAcu)))
