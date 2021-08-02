import random
from sklearn.metrics import roc_auc_score, plot_roc_curve, roc_curve
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr
from sklearn.feature_selection import f_classif, chi2
from sklearn import preprocessing
import numpy as np
import math


def ExtractData(label, data):
    extracted_data = []
    for line in data[1:]:
        for j in range(len(data[0])):
            if label[2] != data[0][j]:
                continue
            extracted_data.append(line[j])
    return extracted_data


def extract_data(data):
    ret = []
    for line in data[1:]:
        ret.append(line[1:])
    return ret


def train(train_x, train_y, model):
    model.fit(train_x, train_y)
    return model


def test(test_x, test_y, model):
    y_predict = model.predict_proba(test_x)
    y_0 = list(y_predict[:, 1])
    auc = roc_auc_score(test_y, y_0)
    '''
    plot_roc_curve(model, test_x, test_y)
    plt.show()
    exit(0)
    '''
    return auc


def reverse(data):
    reversed_data = []
    for i in range(max(len(data), len(data[0]))):
        new_line = []
        if i >= len(data[0]):
            continue
        for j in data:
            new_line.append(j[i])
        reversed_data.append(new_line)
    return reversed_data


def clinicRead(clinic_file):
    clinic = []
    for line in clinic_file:
        data = line[:-2].split('\t')
        clinic.append(data)
    return clinic


def clinicRefine(clinic, label):
    if clinic == 'null':
        return clinic, 'null'

    used = []
    for i in label:
        used.append(i[2])

    useful_line = [9, 20]#18, , 23
    refined_data = []
    name = ["ID"]
    for i in useful_line:
        #print(clinic[0][i], end=', ')
        name.append(clinic[0][i])
    c_mean = [[0, 0], [0, 0]]
    for line in clinic[1:]:
        if line[0] not in used:
            continue
        temp = []
        for j in range(len(line[9])):
            if line[9][j] == 'N' or line[9][j] == 'M':
                temp.append(int(line[9][j+1]))

        temp.append(2 if line[20] == 'None' else int(line[20]))
        refined_data.append(temp)

    return refined_data


def ppRead(row_data):
    refined_data = []
    flag = 0
    for line in row_data:
        data = line[:-2].split('\t')
        lines = [data[0]]
        for i in data[1:]:
            if i == 'nan' or flag == 0:
                if i == 'nan':
                    i = np.nan
                if flag == 0 and i[0:4] == 'CCRC':
                    i = i[0:4] + '-' + i[5:]
                lines.append(i)
            else:
                lines.append(float(i))
        flag = 1
        refined_data.append(lines)
    return refined_data


def ppRefine(row_data, label, useful_data):
    if row_data == 'null':
        return 'null'
    used = []
    for i in label:
        used.append(i[2])

    refined_data = []
    for line in row_data:
        if line[0] not in useful_data:
            continue
        new_line = []
        sum = 0
        n = 0
        for i in range(1, len(row_data[0])):
            if row_data[0][i] not in used:
                continue
            else:
                if line[i] is not np.nan:
                    sum += line[i]
                    n += 1

        for i in range(1, len(row_data[0])):
            if row_data[0][i] not in used:
                continue
            else:
                if line[i] is not np.nan:
                    new_line.append(line[i])
                else:
                    new_line.append(sum/n)
        refined_data.append(new_line)
    return refined_data


def labelRefine(label_file):
    refined_data = []
    for line in label_file:
        data = line[:-2].split('\t')
        refined_data.append((int(data[0]), data[1], data[2]))
    return refined_data


def somaticRead(somatic_file):
    somatic = []
    for line in somatic_file:
        data = line[:-2].split('\t')
        somatic.append(data)
    return somatic


def somaticRefine(somatic, label):
    if somatic == 'null':
        return somatic, 'null'

    used = []
    for i in label:
        used.append(i[2])

    useful_data = ['ID', 'COL6A3', 'OTOG', 'KAL1']

    ret = []
    h = {}
    name = []
    all_symbol = ['ID']
    for line in somatic[1:]:
        if line[0] not in used:
            continue
        if line[1] not in useful_data:
            continue
        if (line[0] not in name):
            h[line[0]] = []
            name.append(line[0])
        h[line[0]].append(line[1])
        if (line[1] not in all_symbol):
            all_symbol.append(line[1])

    all_symbol = useful_data
    for i in used:
        if i not in h:
            ret.append([0, 0, 0])
            continue
        new_line = []
        for j in all_symbol[1:]:
            if j not in h[i]:
                new_line.append(0)
            else:
                new_line.append(1)
        ret.append(new_line)

    return ret


def feature_selection(label, X, p = 0.05):
    target = []
    for line in label:
        target.append(0 if (line[1][0] == 'Y') else 1)

    sp = SelectFpr(f_classif, alpha=p)
    new_X = sp.fit_transform(X, target)
    tail = []
    for i in range(len(sp.pvalues_)):
        if sp.pvalues_[i] <= p:
            tail.append(i)

    return new_X, sp


def generate_data_pca(x, y):
    import xlsxwriter as xw
    workbook = xw.Workbook('Table S2.xlsx')

    worksheet = workbook.add_worksheet("Data")
    worksheet.activate()
    features = [('feature '+str(i)) for i in range(1, 11)]
    worksheet.write_row('A1', ['sample'] + features + ['label'])

    line = 2
    x = list(x)
    for i in range(len(x)):
        insertData = [i+1] + list(x[i]) + [y[i]]
        row = 'A' + str(line)
        worksheet.write_row(row, insertData)
        line += 1

    workbook.close()


def generate_data_augmentation(x, y):
    import xlsxwriter as xw
    workbook = xw.Workbook('Table S3.xlsx')

    worksheet = workbook.add_worksheet("Data")
    worksheet.activate()
    features = [('feature ' + str(i)) for i in range(1, 11)]
    worksheet.write_row('A1', ['sample'] + features + ['label'])

    line = 2
    x = list(x)
    for i in range(len(x)):
        insertData = [i + 1] + list(x[i]) + [y[i]]
        row = 'A' + str(line)
        worksheet.write_row(row, insertData)
        line += 1

    workbook.close()