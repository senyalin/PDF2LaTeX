
# create labels, 1 for ME, 0 for text
lines = open('src-train.txt').readlines()
labels = ['0\n'] * len(lines)
for i, line in enumerate(lines):
    # if len(line) == 15:
    if '_' == line[0]:
        labels[i] = '1\n'
open('tgt-train.txt', 'w').writelines(labels)



# lines = open('src-test_perfect_me.txt').readlines()
# new_lines = []
# for i, line in enumerate(lines):
#     if len(line) != 15:
#         new_lines.append(line)
# open('src-test_perfect_me.txt', 'w').writelines(new_lines)

# # shuffle files
# import random
# lines = open('src.txt').readlines()
# random.shuffle(lines)
# open('src.txt', 'w').writelines(lines)


# # move files
# import shutil
# src = '/home/meca/OpenNMT-py/data_harvard/im2text/images/'
# dst = '/home/meca/BinaryCNN/data/images/'
# lines = open('src-val.txt').readlines()
# for line in lines:
#     shutil.copyfile((src+line)[:-1], (dst+line)[:-1])
# lines = open('src-test.txt').readlines()
# for line in lines:
#     shutil.copyfile((src+line)[:-1], (dst+line)[:-1])
# lines = open('src-train.txt').readlines()
# for line in lines:
#     shutil.copyfile((src+line)[:-1], (dst+line)[:-1])