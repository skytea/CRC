import tools
import time
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn import svm
from mlxtend.classifier import StackingClassifier
from sklearn.model_selection import train_test_split,RepeatedKFold
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
import random
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve, auc
import matplotlib.pyplot as plt


def cacu_dis(x, y):
    ret = 0
    for i in range(len(x)):
        ret += (x[i] - y[i]) ** 2
    return ret


def smote(data, n, k, y):
    #data = list(data)
    ret = []
    ret_y = []
    g1 = []
    g2 = []
    for i in range(len(y)):
        if y[i] == 1:
            g1.append(i)
        else:
            g2.append(i)
    data_d = [list(data[g2,]), list(data[g1,])]

    random.seed(0)
    for i in range(len(data)):
        k_list = [10000000 for ii in range(k)]
        k_dis = [10000000 for ii in range(k)]
        for j in range(len(data_d[y[i]])):
            qaq = j
            dis = cacu_dis(data[i], data_d[y[i]][j])
            for kk in range(len(k_list)):
                if dis < k_dis[kk]:
                    temp = k_list[kk]
                    temp_dis = k_dis[kk]
                    k_list[kk] = qaq
                    k_dis[kk] = dis
                    qaq = temp
                    dis = temp_dis

        random.shuffle(k_list)
        k_list = k_list[:n]
        for kk in range(len(k_list)):
            new_line = []

            for j in range(len(data[i])):
                c = random.random()
                new_line.append(data[i][j] + c * (data_d[y[i]][k_list[kk]][j] - data[i][j]))
            ret.append(new_line)
            ret_y.append(y[i])

    return np.array(ret, dtype=float), np.array(ret_y)


def single_test(train_x, train_y, test_x, test_y, model):

    model = tools.train(train_x, train_y, model)
    return model.predict(test_x), test_y
    accuracy = tools.test(test_x, test_y, model)

    '''
    qaq = -1
    plt.tick_params(labelsize=15)
    for clf, c_name in zip([svm, lr, nb, stacking_clf],
                           ['svm',
                            'Logistic Regression',
                            'Naive Bayes',
                            'Stacking Classifier']):
        color = [(255 / 255, 225 / 255, 150 / 255), (100 / 255, 180 / 255, 246 / 255), (76 / 255, 175 / 255, 79 / 255),
                 (229 / 255, 115 / 255, 115 / 255)]
        name = ['SVM', 'LR', 'Naive Bayes', 'Ensemble']
        model_ = tools.train(train_x, train_y, clf)
        y_predict = model_.predict_proba(test_x)
        fpr, tpr, thresholds = roc_curve(test_y, y_predict[:, 1], pos_label=1)
        roc_auc = auc(fpr, tpr)
        qaq += 1
        plt.plot(fpr, tpr, lw=1, color = color[qaq], label=name[qaq]+' (AUC = %0.2f)' % roc_auc)
    plt.ylabel("True positive rate", fontsize=15)
    plt.xlabel("False positive rate", fontsize=15)
    plt.title("receiver operating characteristic curve", fontsize=15)
    plt.legend(loc='lower right', fontsize=10)
    plt.show()
    '''

    return accuracy


def multiple_test(times, model, label_x, y):
    accuracy = []
    kf = RepeatedKFold(n_splits=4, n_repeats=times, random_state=0)
    precision = []
    sensitivity = []
    specificity = []
    for train_index, test_index in kf.split(label_x):
        train_x = label_x[train_index]
        train_y = y[train_index]
        test_x = label_x[test_index]
        test_y = y[test_index]
        pre_y, y_true = single_test(train_x, train_y, test_x, test_y, model)
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(len(pre_y)):
            if y_true[i] == 1:
                if pre_y[i] == 1:
                    TP += 1
                else:
                    FN += 1
            else:
                if pre_y[i] == 1:
                    FP += 1
                else:
                    TN += 1
        accuracy.append((TP + TN) / (TP+FP+TN+FN))
        precision.append((TP) / (TP + FP))
        sensitivity.append((TP) / (TP+FN))
        specificity.append((TN) / (FP + TN))
        #accuracy.append(single_test(train_x, train_y, test_x, test_y, model, unlabel_x))
    print(accuracy)
    print(round(np.mean(accuracy), 4), end='')
    print('??', end='')
    print(round(np.std(accuracy), 4))
    print(precision)
    print(round(np.mean(precision), 4), end='')
    print('??', end='')
    print(round(np.std(precision), 4))
    print(sensitivity)
    print(round(np.mean(sensitivity), 4), end='')
    print('??', end='')
    print(round(np.std(sensitivity), 4))
    print(specificity)
    print(round(np.mean(specificity), 4), end='')
    print('??', end='')
    print(round(np.std(specificity), 4))


def data_exam(data, y):
    data = data.T
    std = np.std(data, axis=1)
    data = list(data)
    ua = 1.96
    ub = 1.2816
    u = -1
    maxn = 0
    for line in data:
        u += 1
        div = [[0, 0], [0, 0]]
        for i in range(len(data[0])):
            div[y[i]][0] += line[i]
            div[y[i]][1] += 1
        det = div[0][0]/div[0][1] - div[1][0]/div[1][1]
        sample_size = ((std[u] * (ua + ub) / det) ** 2) * ((div[0][1]+div[1][1])/div[0][1] + (div[0][1]+div[1][1])/div[1][1])
        maxn = max(maxn, sample_size)
    print(maxn)


def discriminant(data, y):
    data = data.T
    data = list(data)
    f_f = 0
    y = list(y)
    for line in data:
        line = list(line)
        div = [[], []]
        for i in range(len(data[0])):
            div[int(y[i])].append(line[i])
        x1 = np.array(div[0])
        x2 = np.array(div[1])
        f = (x1.mean() - x2.mean()) ** 2 / (x1.std() ** 2 + x2.std() ** 2)
        f_f = max(f, f_f)
    print(f_f)


def F_pic(x, y):
    pca_model = PCA(n_components=2)
    x = pca_model.fit_transform(x)
    colors = ['r', 'b']
    color = []
    for i in y:
        color.append(colors[i])
    plt.xticks([])
    plt.yticks([])
    plt.scatter(x[:,0], x[:,1], c=color)
    plt.show()


label_ = tools.labelRefine(open('label.txt','r'))
clinic = tools.clinicRead(open('clinic.txt','r'))
somatic = tools.somaticRead(open('somatic.txt','r'))
protein = tools.ppRead(open('protein.txt','r'))
phosphosites = tools.ppRead(open('phosphosites.txt','r'))

label = []
none_label = []
for i in label_:
    if i[1] == 'None':
        none_label.append(i)
    else:
        label.append(i)

useful_data = ['A0A024R4G1_509_Y', 'A0A024R4Z6_441_Y', 'A0A024RAM4_1817_S', 'A0A087WVT6_320_S', 'A0A140VJN8_130_S', 'B4DPP8_314_T', 'B4DPP8_320_S', 'B4DPP8_325_T', 'P08670_436_T', 'P62263_137_S', 'Q6WKZ4_206_S', 'Q6WKZ4_338_S', 'Q8IZ21_358_T', 'Q92625_628_S', 'A0A024R046', 'A0A024R056', 'A0A024R0Y5', 'A0A024R1S8', 'A0A024R2U7', 'A0A024R3B5', 'A0A024R5K1', 'A0A024R7I3', 'A0A024R9G4', 'A0A024RCX8', 'A0A024RCY1', 'A0A087WTA8', 'A0A0A0MRF6', 'A0A0A0MSM0', 'A0A0S2Z3J9', 'A0A140VJC9', 'A2RUA4', 'A4D1X5', 'A6NHQ2', 'A8K2L4', 'A8K878', 'B2R4P9', 'B2R5J1', 'B2RDF2', 'B2RDW1', 'B2ZDQ1', 'B4DEH0', 'B7ZB78', 'C9K0I4', 'D3DT27', 'E9PMC9', 'G1EPM2', 'O14917', 'O60493', 'O75554', 'P02452', 'P02461', 'P02749', 'P05164', 'P11908', 'P14780', 'P17066', 'P17600', 'P18077', 'P20585', 'P23378', 'P24158', 'P29350', 'P31947', 'P35637', 'P36268', 'P37802', 'P40763', 'P42224', 'P48061', 'P49327', 'P51572', 'P52209', 'P52630', 'P52732', 'P54707', 'P63261', 'P68371', 'P78346', 'Q13151', 'Q13287', 'Q13884', 'Q15029', 'Q2TAM5', 'Q53SW3', 'Q546E0', 'Q6DN03', 'Q6FIA3', 'Q7Z434', 'Q8N8A2', 'Q8WUM0', 'Q8WXH0', 'Q92734', 'Q96A33', 'Q96BP3', 'Q96IS6', 'Q99497', 'Q9BTE1', 'Q9BTT0', 'Q9NZ08', 'Q9UI15', 'Q9UNS2', 'Q9Y2Z0', 'Q9Y426']

clinic = tools.clinicRefine(clinic, label_)
clinic = np.array(clinic, dtype=int)
somatic = np.array(tools.somaticRefine(somatic, label_), dtype=int)
protein = np.array(tools.ppRefine(protein, label_, useful_data), dtype=float).T
phosphosites = np.array(tools.ppRefine(phosphosites, label_, useful_data), dtype=float).T
y = []
for i in label_:
    if i[1] == 'Yes':
        y.append(1)
    if i[1] == 'No':
        y.append(0)
unlabeled = [101, 137]
clinic = np.delete(clinic, unlabeled, 0)
somatic = np.delete(somatic, unlabeled, 0)
protein = np.delete(protein, unlabeled, 0)
phosphosites = np.delete(phosphosites, unlabeled, 0)

data_exam(clinic, y)
data_exam(somatic, y)
data_exam(protein, y)
data_exam(phosphosites, y)

s = preprocessing.StandardScaler()
protein = s.fit_transform(protein)
phosphosites = s.fit_transform(phosphosites)
pca_model = PCA(n_components=2)
protein = pca_model.fit_transform(protein)
phosphosites = pca_model.fit_transform(phosphosites)
k1 = np.hstack((somatic, clinic))
k2 = np.hstack((protein, phosphosites))
x = np.hstack((k1, k2))
y = np.array(y)

#tools.generate_data_pca(x, y)

lr = LogisticRegression(max_iter=10000)
svm = svm.SVC(max_iter=10000, probability=True)
nb = GaussianNB()

ada_lr = AdaBoostClassifier(lr, n_estimators=50, learning_rate=0.1)
ada_svm = AdaBoostClassifier(svm, n_estimators=50, learning_rate=0.1)
ada_nb = AdaBoostClassifier(nb, n_estimators=50, learning_rate=0.1)
stacking_clf = StackingClassifier(classifiers=[ada_svm, ada_lr, ada_nb], meta_classifier=LogisticRegression(),
                                  use_probas=True,
                                  average_probas=False)

model = stacking_clf
augmented_data, data_y = smote(x, 1, 5, y)

#tools.generate_data_augmentation(augmented_data, data_y)

discriminant(x, y)
#F_pic(x, y)
discriminant(augmented_data, data_y)
#F_pic(augmented_data, data_y)

label_x = np.vstack((augmented_data, x))
y = np.hstack((y, data_y))

times_r = 20
multiple_test(times_r, model, label_x, y)
multiple_test(times_r, lr, label_x, y)
multiple_test(times_r, svm, label_x, y)
multiple_test(times_r, nb, label_x, y)