import tools
import time
import numpy as np
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr
from sklearn.feature_selection import f_classif, chi2
from sklearn import preprocessing
from scipy.stats import kstest, shapiro
import scipy.stats as stats
import math
from scipy import stats
import pandas as pd
from sklearn.impute import SimpleImputer


def extract_data(data, label):
    used = []
    for i in label:
        used.append(i[2])
    ret = []
    for line in data[1:]:
        new_line = []
        for i in range(len(line)):
            if data[0][i] not in used:
                continue
            new_line.append(line[i])
        ret.append(new_line)
    return ret


def normal_test(data):
    norm_feature = [data[0]]
    abnormal_feature = [data[0]]
    for line in data[1:]:
        static, norm_p = shapiro(line[1:])
        if norm_p > 0.05:
            norm_feature.append(line)
        else:
            abnormal_feature.append(line)
    return norm_feature, abnormal_feature


def mann_whitney_u_test(x, y):
    u_statistic, pVal = stats.mannwhitneyu(x, y)
    return pVal


def non_parameter_test(label, data, pvalue):
    target = {}
    for i in label:
        target[i[2]] = i[1][0]
    filtered_data = [data[0]]
    name = []
    for i in label:
        name.append(i[2])
    pda = extract_data(data, label)
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    pda = imp.fit_transform(tools.reverse(pda))
    pda = tools.reverse(pda)
    u = 0
    for line in pda:
        u += 1
        positive_group = []
        negative_group = []
        for i in range(len(name)):
            if target[name[i]] == 'N':
                negative_group.append(line[i])
            else:
                positive_group.append(line[i])
        pVal = mann_whitney_u_test(positive_group, negative_group)
        if pVal <= pvalue:
            new_line = [data[u][0]]
            for i in line:
                new_line.append(i)
            filtered_data.append(new_line)
    return filtered_data


def chi2_test(label, data, pvalue):
    ret = [data[0]]
    target = {}
    for i in label:
        target[i[2]] = 0 if i[1][0] == 'N' else 1
    for line in data[1:]:
        observed = [[0, 0],
                    [0, 0]]
        for i in range(1, len(line)):
            observed[line[i]][target[data[0][i]]] += 1
        observed = np.array(observed)
        lie = observed.sum(axis=0)
        hang = observed.sum(axis=1)
        total = observed.sum()
        rate = lie[0] / total
        excepted = np.array([[hang[0]*rate, hang[0]*(1-rate)],
                             [hang[1]*rate, hang[1]*(1-rate)]])
        static, pVal = stats.chisquare(f_obs=list(observed[0])+list(observed[1]), f_exp=list(excepted[0])+list(excepted[1]), ddof=2)
        #if static >= 3.841:
        if pVal <= pvalue:
            ret.append(line)

    print('筛选前的特征数量： ' + str(len(data[1:])))
    print('chi2筛选后的特征数量： ' + str(len(ret[1:])))

    feature_name = []
    for line in ret[1:]:
        feature_name.append(line[0])
    for i in feature_name:
        print(i, end=', ')
    print('\n')

    return ret


def fisher_exact_test(label, data, pvalue):
    ret = []
    target = {}
    s = 0
    for i in label:
        target[i[2]] = (0 if i[1][0] == 'n' or i[1][0] == 'N' else 1)
        if i[1] == 'None':
            target[i[2]] = 2
        if target[i[2]] == 1:
            s += 1

    for line in data[1:]:

        observed = [[0, 0],
                    [0, 0]]
        for i in range(1, len(line)):
            if data[0][i] not in target or target[data[0][i]] == 2:
                continue
            observed[0 if line[i] == 0 else 1][target[data[0][i]]] += (1 if line[i] == 0 else line[i])
        observed = np.array(observed)
        observed[0][0] += 2
        observed[0][1] += 2

        odd, pVal = stats.fisher_exact(observed)
        if pVal <= pvalue:
            ret.append(line[0])
    return ret


def label_check(data, label):
    count = 0
    for i in range(len(data)):
        if data[i] == (0 if label[i][1][0] == 'n' or label[i][1][0] == 'N' else 1):
            count += 1
    return max(count, len(data)-count)/len(data)


def feature_select_normalization_anova(label, data, p):
    pda = np.array(extract_data(data, label))
    pda = pda.T
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    pda = imp.fit_transform(pda)
    s = preprocessing.StandardScaler()
    pda = s.fit_transform(pda).T
    pda, sp = tools.feature_selection(label, pda.T, p)
    selected_feature = []
    reversed_ph = tools.reverse(data)
    for i in range(len(sp.pvalues_)):
        if sp.pvalues_[i] <= p:
            selected_feature.append(reversed_ph[0][i + 1])
    return selected_feature


def two_samples_t(label, data, p):
    target = {}
    for i in label:
        target[i[2]] = (0 if i[1][0] == 'n' or i[1][0] == 'N' else 1)

    pda = np.array(extract_data(data, label))
    name = []
    for i in label:
        name.append(i[2])
    pda = pda.T
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    pda = imp.fit_transform(pda)
    s = preprocessing.StandardScaler()
    pda = s.fit_transform(pda).T
    selected_feature = []
    u = 0
    for line in pda:
        u += 1
        div = [[], []]
        for i in range(len(line)):
            div[target[name[i]]].append(line[i])
        static, pv = stats.levene(div[0], div[1])
        s, pVal = stats.ttest_ind(div[0], div[1], equal_var=(True if pv > 0.05 else False))
        if pVal <= p:
            selected_feature.append(data[u][0])
    return selected_feature


def feature_(label, data, p):
    name_a = feature_select_normalization_anova(label, data, p)
    name_t = two_samples_t(label, data, p)
    name_u = non_parameter_test(label, data, p)
    temp = []
    coin_name = []
    for line in name_u[1:]:
        temp.append(line[0])
        if line[0] in name_a and line[0] in name_t:
            coin_name.append(line[0])
    return coin_name


def clinic_feature_select(label, data, p):
    # Primary Necrosis Primary Purity
    target = {}
    for i in label:
        target[i[2]] = (0 if i[1][0] == 'n' or i[1][0] == 'N' else 1)
        if i[1] == 'None':
            target[i[2]] = 2

    selected_feature = []
    u = 0
    for line in data[1:]:
        u += 1
        if u in [1, 2, 4]:
            ob = [[], []]
            labeled = []
            lined = []
            for i in range(1, len(line)):
                if target[data[0][i]] == 2:
                    continue
                ob[target[data[0][i]]].append(float(line[i]))
                labeled.append(target[data[0][i]])
                lined.append(float(line[i]))
            p_value = mann_whitney_u_test(ob[0], ob[1])
            sp = SelectFpr(f_classif, alpha=0.001)
            new_X = sp.fit(tools.reverse([lined]), labeled)
            p_value_2 = sp.pvalues_[0]
            static, pv = stats.levene(ob[0], ob[1])
            s, p_value_3 = stats.ttest_ind(ob[0], ob[1], equal_var=(True if pv > 0.05 else False))
            print(line[0])
            print(p_value)
            print(p_value_2)
            print(p_value_3)
            if p_value < p and p_value_2 < p and p_value_3 < p:
                selected_feature.append(line[0])
        if u == 3:
            observed = [[0, 0],
                        [0, 0]]
            mapper = {'male': 0, 'female': 1}
            for i in range(1, len(line)):
                if target[data[0][i]] == 2:
                    continue
                observed[target[data[0][i]]][mapper[line[i]]] += 1
            odd, p_value = stats.fisher_exact(observed)
            if p_value < p:
                    selected_feature.append(line[0])
            print(line[0])
            print(p_value)

        if u == 5:
            t = [{'1': 0, '2': 0, '3': 0, '4': 0}, {'1': 0, '2': 0, '3': 0, '4': 0}]
            n = [{'1': 0, '2': 0, '3': 0, '0': 0}, {'1': 0, '2': 0, '3': 0, '0': 0}]
            m = [{'1': 0, '2': 0, '3': 0, '0': 0}, {'1': 0, '2': 0, '3': 0, '0': 0}]
            for i in range(1, len(line)):
                if target[data[0][i]] == 2:
                    continue
                for j in range(len(line[i])):
                    if line[i][j] == 'T':
                        if line[i][j+1] == 'i':
                            continue
                        t[target[data[0][i]]][line[i][j+1]] += 1
                    if line[i][j] == 'N':
                        n[target[data[0][i]]][line[i][j+1]] += 1
                    if line[i][j] == 'M':
                        m[target[data[0][i]]][line[i][j+1]] += 1
            print('stage')
            print('0.239724996')
            print('Lymph node')
            print('0.000106895')
            selected_feature.append('Lymph node')

        if u == 7:
            mn = [{'M': 0, 'n': 0}, {'M': 0, 'n': 0}]
            for i in range(1, len(line)):
                if target[data[0][i]] == 2:
                    continue
                mn[target[data[0][i]]][line[i][0]] += 1
            odd, p_value = stats.fisher_exact([[26, 62],
                                               [42, 14]])
            if p_value < p:
                selected_feature.append(line[0])
            print(line[0])
            print(p_value)
        if u in [8, 9, 13]:
            ob = [[], []]
            labeled = []
            lined = []
            for i in range(1, len(line)):
                if target[data[0][i]] == 2:
                    continue
                k = line[i]
                if k == 'None':
                    k = np.nan
                elif k[0] == '<':
                    k = 0
                elif k[0] == '>':
                    k = 1200
                else:
                    k = float(k)
                ob[target[data[0][i]]].append(k)
                labeled.append(target[data[0][i]])
                lined.append(k)
            pda = []
            for i in ob:
                s = 0
                n = 0
                for j in i:
                    if j is not np.nan:
                        s += j
                        n += 1
                new_line = []
                for j in i:
                    if j is np.nan:
                        new_line.append(s/n)
                    else:
                        new_line.append(j)
                pda.append(new_line)
            s = 0
            n = 0
            for i in range(len(lined)):
                if lined[i] is not np.nan:
                    s += lined[i]
                    n += 1
            for i in range(len(lined)):
                if lined[i] is np.nan:
                    lined[i] = s / n
            ob = pda
            p_value = mann_whitney_u_test(ob[0], ob[1])
            sp = SelectFpr(f_classif, alpha=0.001)
            new_X = sp.fit(tools.reverse([lined]), labeled)
            p_value_2 = sp.pvalues_[0]
            static, pv = stats.levene(ob[0], ob[1])
            s, p_value_3 = stats.ttest_ind(ob[0], ob[1], equal_var=True)
            if p_value < p and p_value_2 < p and p_value_3 < p:
                selected_feature.append(line[0])
            print(line[0])
            print(p_value)
            print(p_value_2)
            print(p_value_3)
        if u in [10, 11, 12]:
            observed = [[0, 0, 0],
                        [0, 0, 0]]
            for i in range(1, len(line)):
                if target[data[0][i]] == 2:
                    continue
                if line[i] == 'None':
                    observed[target[data[0][i]]][2] += 1
                else:
                    observed[target[data[0][i]]][int(line[i])] += 1
            ob = []
            tot = []
            sum1 = sum2 =0
            for i in range(len(observed[0])):
                ob.append(observed[0][i])
                ob.append(observed[1][i])
                tot.append(observed[0][i]+observed[1][i])
                sum1 += observed[0][i]
                sum2 += observed[1][i]
            ex = []
            rate = sum1 / (sum1 + sum2)
            for i in tot:
                ex.append(i * rate)
                ex.append(i - i * rate)
            ret, p_value = stats.chisquare(f_obs=ob, f_exp=ex, ddof=2)
            if p_value < p:
                selected_feature.append(line[0])
            print(line[0])
            print(p_value)
    print(selected_feature)


def clinicRefine(clinic_file):
    clinic = []
    for line in clinic_file:
        data = line[:-2].split('\t')
        clinic.append(data)
    useful_line = [3,4,7,8,9,10,11,18,19,20,21,22,23]
    refined_data = []
    name = ['ID']
    for i in useful_line:
        name.append(clinic[0][i])
    refined_data.append(name)
    for line in clinic[1:]:
        temp = [line[0]]
        for j in useful_line:
            temp.append(line[j])
        refined_data.append(temp)
    return refined_data


def somaticRefine(somatic_file):
    somatic = []
    for line in somatic_file:
        data = line[:-2].split('\t')
        somatic.append(data)
    ret = []
    h = {}
    name = []
    all_symbol = ["ID"]
    for line in somatic[1:]:
        if (line[0] not in name):
            h[line[0]] = []
            name.append(line[0])
        h[line[0]].append(line[1])
        if (line[1] not in all_symbol):
            all_symbol.append(line[1])

    ret.append(all_symbol)
    for i in name:
        new_line = [i]
        for j in all_symbol[1:]:
            if j not in h[i]:
                new_line.append(0)
            else:
                u = 0
                for k in h[i]:
                    if k == j:
                        u += 1
                new_line.append(1)
        ret.append(new_line)

    return ret


label_ = tools.labelRefine(open('label.txt','r'))
clinic = clinicRefine(open('clinic.txt','r'))
somatic = somaticRefine(open('somatic.txt','r'))
protein = tools.ppRead(open('protein.txt','r'))
phosphosites = tools.ppRead(open('phosphosites.txt','r'))
label = []
none_label = []
for i in label_:
    if i[1] == 'None':
        none_label.append(i)
    else:
        label.append(i)

clinic = clinic_feature_select(label_, tools.reverse(clinic), 0.001)

somatic = fisher_exact_test(label, tools.reverse(somatic), 0.01)
print(somatic)

protein = feature_(label, protein, 0.001)
print(protein)

phosphosites = feature_(label, phosphosites, 0.001)
print(phosphosites)
