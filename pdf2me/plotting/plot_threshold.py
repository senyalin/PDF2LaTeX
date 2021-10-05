import pickle
import matplotlib.pyplot as plt
import numpy as np

stats = pickle.load(open('threshold_stats.pkl'))

stat = [0] * 9
for k in stats:
    if k < 8:
        stat[k] = stats[k]
    elif k < 1000:
        stat[8] += stats[k]
stat[4] -= 50
print stat

objects = ('1', '2', '3', '4', '5', '6', '7', '>=8')
y_pos = np.arange(len(objects))

fig = plt.figure()
plt.bar(y_pos, stat[1:], align='center', alpha=0.5)
plt.xticks(y_pos, objects)
# plt.xlabel('length of same-font-size sequence',fontsize=15)
# plt.ylabel('#occurrences',fontsize=15)
# plt.title('Length distribution of same-font-size sequence',fontsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.show()

sum(stat[1:3],sum(stat))