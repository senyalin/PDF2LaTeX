import os
import shutil
import random


# evaluation
tgt = open('tgt-test.txt').readlines()
pred = open('pred.txt').readlines()
cnt = 0
for i in range(len(tgt)):
    if tgt[i][:-2] == pred[i][:-1]:
        cnt += 1
print(cnt)
print(cnt/len(tgt))


# lines = open('tgt.txt').readlines()
# # random.shuffle(lines)
# for i in range(len(lines)):
#     line = lines[i]
#     # new_line = [c + ' ' for c in line]
#     # lines[i] = ''.join(new_line)
#     lines[i] = line[1:]
# open('tgt.txt', 'w').writelines(lines)

# files = os.listdir('synthetic_data')

# f = open('gt.txt', 'w')
# for file in files:
#     f.write(file + '\n')
# f.close()

# # shuffle files
# lines = open('gt.txt').readlines()
# random.shuffle(lines)
# open('gt.txt', 'w').writelines(lines)