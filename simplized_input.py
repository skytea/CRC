from openpyxl import load_workbook
import numpy as np


def read_table(data, table_name):
    ret = []
    for row in data.get_sheet_by_name(table_name).rows:
        line = []
        for cell in row:
            line.append(cell.value)
        ret.append(line)
    return ret


def fill_NA(data):
    ret = []
    for i in data:
        ret_line = []
        for j in i:
            if j is None:
                ret_line.append(np.nan)
            else:
                ret_line.append(j)
        ret.append(ret_line)
    return ret


def get_label(samples):
    j = -1
    label = []
    for i in samples:
        j += 1
        if j == 0:
            continue
        label.append((j, i[29], i[0]))
    return label

data = load_workbook('new_table.xlsx')

samples = read_table(data, '1. Clinicopathoglogic features')
somatic = read_table(data, '3. Somatic mutations')
protein = fill_NA(read_table(data, '4. Protein expression'))
phosphosites = fill_NA(read_table(data, '5. Phosphosites expression'))
label = get_label(samples)

label_file = open('label.txt','w')
clinic_file = open('clinic.txt','w')
somatic_file = open('somatic.txt','w')
protein_file = open('protein.txt','w')
phosphosites_file = open('phosphosites.txt','w')

for i in label:
    for j in i:
        if(str(j)=='done'):
            continue
        label_file.write(str(j)+'\t')
    label_file.write('\n')
label_file.close()

for i in samples:
    for j in i:
        clinic_file.write(str(j)+'\t')
    clinic_file.write('\n')
clinic_file.close()

for i in somatic:
    for j in i:
        somatic_file.write(str(j)+'\t')
    somatic_file.write('\n')
somatic_file.close()

for i in protein:
    for j in i:
        protein_file.write(str(j)+'\t')
    protein_file.write('\n')
protein_file.close()

for i in phosphosites:
    for j in i:
        phosphosites_file.write(str(j)+'\t')
    phosphosites_file.write('\n')
phosphosites_file.close()

