
import matplotlib.pyplot as plt
import pickle
import numpy


char_spaces = pickle.load(open("char_spaces.pkl", "rb" ) )

fig = plt.figure()

# plt.hist(x=char_spaces, bins=[0, 1, 2, 3, 4, 5, 6], color='#0504aa', alpha=0.7, rwidth=1)

plt.hist(x=char_spaces, bins=numpy.arange(-0.45,6.05,0.1))

plt.title("Histogram of gap distances", fontsize=15)
plt.xlabel("gap distance in pixels", fontsize=15)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.xticks(numpy.arange(0, 6, step=1))

plt.show()


# plt.savefig('E:/hist.png')
# fig.savefig('E:/plot.png')