import random
import os
import shutil

path = 'E:\zelun\synthetic/stop_words/'

files = os.listdir(path)
f = open('stop_words_name.txt','w')
for file in files:
    f.write(file + '\n')
f.close()

lines = open('stop_words_name.txt').readlines()
random.shuffle(lines)
open('stop_words_name.txt', 'w').writelines(lines)

src = 'E:\zelun\synthetic/maths_split/'
dest = 'E:\zelun\synthetic/images/'
# files = os.listdir(src)
cnt = 0
files = open('E:\zelun\math.txt').readlines()
for file in files:
    print(file)
    if os.path.exists(dest + file):
        continue
    shutil.copy(src + file[:-1], dest + file[:-1])
    cnt += 1
    if cnt == 74416:
        break