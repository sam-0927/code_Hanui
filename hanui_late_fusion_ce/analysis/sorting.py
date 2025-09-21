import glob, os, pdb
from collections import defaultdict
newf = open('sort.txt', 'w')
top_k = 10

dict_ = defaultdict(int)
with open('Validation.txt') as f:
    lines = f.readlines()
    for line in lines:
        idx, eer = line.strip().split(' : ')
        dict_[idx] = float(eer)

list_ = sorted(dict_.items(), key=lambda item: item[1])
newf.write('Validation\n')
for i in range(top_k):
    newf.write(str(list_[i][0])+'\n')


dict_ = defaultdict(int)
with open('T01.txt') as f:
    lines = f.readlines()
    for line in lines:
        idx, eer = line.strip().split(' : ')
        dict_[idx] = float(eer)

list_ = sorted(dict_.items(), key=lambda item: item[1])
newf.write('\nT01\n')
for i in range(top_k):
    newf.write(str(list_[i][0])+'\n')


dict_ = defaultdict(int)
with open('T02.txt') as f:
    lines = f.readlines()
    for line in lines:
        idx, eer = line.strip().split(' : ')
        dict_[idx] = float(eer)

list_ = sorted(dict_.items(), key=lambda item: item[1])
newf.write('\nT02\n')
for i in range(top_k):
    newf.write(str(list_[i][0])+'\n')


dict_ = defaultdict(int)
with open('T03.txt') as f:
    lines = f.readlines()
    for line in lines:
        idx, eer = line.strip().split(' : ')
        dict_[idx] = float(eer)

list_ = sorted(dict_.items(), key=lambda item: item[1])
newf.write('\nT03\n')
for i in range(top_k):
    newf.write(str(list_[i][0])+'\n')


dict_ = defaultdict(int)
with open('T04.txt') as f:
    lines = f.readlines()
    for line in lines:
        idx, eer = line.strip().split(' : ')
        dict_[idx] = float(eer)

list_ = sorted(dict_.items(), key=lambda item: item[1])
newf.write('\nT04\n')
for i in range(top_k):
    newf.write(str(list_[i][0])+'\n')


    
