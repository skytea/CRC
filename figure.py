import matplotlib.pyplot as plt
import math
import tools
#import dataselect
from sklearn import preprocessing
import numpy as np


def model_comparison():
    name = ['SVM', 'Ensemble', 'LR', 'Ensemble', 'NB', 'Ensemble']
    mean = [0.9599, 0.9802, 0.9643, 0.9802, 0.9488, 0.9802]
    std = [0.0222, 0.0155, 0.0216, 0.0155, 0.0261, 0.0155]
    color = [(255/255, 225/255, 150/255), (255/255, 225/255, 150/255), (100/255, 180/255, 246/255), (100/255, 180/255, 246/255), (76/255, 175/255, 79/255), (76/255, 175/255, 79/255)]

    x = list(1.5*i+0.75 for i in range(len(mean)))
    print(x)
    total_width, n = 0.5, 1
    width = total_width / n
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.ylim(0.7, 1)
    plt.xlim(0, 9)
    plt.ylabel("Area under curve", fontsize=20)
    plt.tick_params(labelsize=15)
    plt.bar(x, [0,0,0,0,0,0], tick_label=name)
    for i in range(len(name)):
        if i == 0:
            plt.bar(x[i], mean[i], width=width, fc=color[i])
        else:
            plt.bar(x[i], mean[i], width=width, fc=color[i])
    plt.errorbar(x, mean, yerr=std, fmt=" ", ecolor='black', elinewidth=1, capsize=4)
    plt.savefig('model_comparison.png')
    plt.show()


def plot_pca():
    p = 0.001
    label_ = tools.labelRefine(open('label.txt','r'))
    protein = tools.ppRead(open('protein.txt', 'r'))
    phosphosites = tools.ppRead(open('phosphosites.txt', 'r'))
    useful_data = []
    coin_name_ph = dataselect.feature_(label_, phosphosites, p)
    coin_name_pr = dataselect.feature_(label_, protein, p)
    for i in coin_name_ph:
        useful_data.append(i)
    for i in coin_name_pr:
        useful_data.append(i)
    protein = tools.ppRefine(protein, label_, useful_data)
    phosphosites = tools.ppRefine(phosphosites, label_, useful_data)
    s = preprocessing.StandardScaler()
    protein[1:] = s.fit_transform(tools.reverse(tools.extract_data(protein)))
    protein[1:], pro_pca = tools.pca(protein[1:], 2)
    protein[1:] = tools.reverse(protein[1:])
    phosphosites[1:] = s.fit_transform(tools.reverse(tools.extract_data(phosphosites)))
    phosphosites[1:], pro_pca = tools.pca(phosphosites[1:], 2)
    phosphosites[1:] = tools.reverse(phosphosites[1:])
    for i in range(1, 3):
        protein[i][1:] = protein[i]
        phosphosites[i][1:] = phosphosites[i]
    color = ['r', 'b']
    label = ['类标为0', '类标为1']
    target = {}
    for i in label_:
        target[i[2]] = (0 if i[1][0] == 'n' or i[1][0] == 'N' else 1)
        if i[1] == 'None':
            target[i[2]] = 2
    for i in range(1, len(protein[0])):
        if target[protein[0][i]] == 2:
            continue
        plt.scatter(protein[1][i], protein[2][i], c=color[target[protein[0][i]]])
    #plt.legend(loc='upper right', fontsize=8)
    plt.xlabel('First Principal Component', fontsize=20)
    plt.ylabel('Second Principal Component', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.savefig('PCA1.png')
    plt.show()
    for i in range(1, len(phosphosites[0])):
        if target[phosphosites[0][i]] == 2:
            continue
        plt.scatter(phosphosites[1][i], phosphosites[2][i], c=color[target[phosphosites[0][i]]])
    #plt.legend(loc='upper right', fontsize=8)
    plt.xlabel('First Principal Component', fontsize=20)
    plt.ylabel('Second Principal Component', fontsize=20)
    plt.tick_params(labelsize=15)
    plt.savefig('PCA2.png')
    plt.show()


def APSS():
    name = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity']
    label = ['SVM', 'LR', 'NB', 'Ensemble']
    mean = []
    std = []
    for i in label:
        file = open(i + '.txt', 'r')
        content = file.readlines()
        mean.append([float(i) for i in content[0].split(' ')[:4]])
        std.append([float(i) for i in content[1].split(' ')[:4]])
    print(std)

    color = [(255/255, 225/255, 150/255), (100/255, 180/255, 246/255), (76/255, 175/255, 79/255), (229/255, 115/255, 115/255)]

    x = list(range(len(mean[0])))
    total_width, n = 0.8, 4
    width = total_width / n
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.ylim(0.6, 1.15)
    plt.xlim(-0.3, 4)
    plt.tick_params(labelsize=15)
    for i in range(0, len(mean)):
        if i == 1:
            plt.bar(x, mean[i], width=width, label=label[i], fc=color[i], tick_label=name)
        else:
            plt.bar(x, mean[i], width=width, label=label[i], fc=color[i])
        plt.errorbar(x, mean[i], yerr=std[i], fmt=" ",ecolor='black', elinewidth=1, capsize=4)
        for j in range(len(x)):
            x[j] = x[j] + width
    plt.legend(loc='upper center', fontsize=14, ncol=4)
    plt.savefig('result-model.png')
    plt.show()


model_comparison()