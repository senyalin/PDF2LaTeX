import numpy as np

class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    def __str__(self):
        return "Point({}, {})".format(self._x, self._y)

    def __repr__(self):
        return self.__str__()

    def x(self):
        return self._x

    def y(self):
        return self._y


def dist(xy1, xy2):
    return np.sqrt(
        (xy1[0]-xy2[0])**2 + (xy1[1]-xy2[1])**2
    )