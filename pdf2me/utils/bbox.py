# might be moved to a general package
import numpy as np
from pdf_parser.point import Point
import copy
delta = 1e-10

class BBox(object):
    """
    TODO, migrate all tuple base bbox to this data structure.
    Make this even into some more common packages.

    """
    def __init__(self, quadruple):
        """

        :param quadruple: left, bottom, right, top
        """
        if isinstance(quadruple, list) or isinstance(quadruple, tuple):
            self.quadruple = list(quadruple)
        elif isinstance(quadruple, BBox):
            self.quadruple = list(quadruple.quadruple)
        elif isinstance(quadruple, dict):
            b_list = 'lrbt'
            for b_name in b_list:
                assert b_name in quadruple
            self.quadruple = [
                    quadruple['l'],
                    quadruple['b'],
                    quadruple['r'],
                    quadruple['t']
                ]
        elif isinstance(quadruple, str):
            import re
            for val in re.findall(r'[\d\.]+', quadruple):
                print val
            val_list = [float(val) for val in re.findall(r'[\d\.]+', quadruple)]
            self.quadruple = val_list
        else:
            raise Exception("unknown input")
        if self.quadruple[0] == self.quadruple[2]:
            self.quadruple[2] += delta
        if self.quadruple[1] == self.quadruple[3]:
            self.quadruple[3] += delta

    def __str__(self):
        return "BBox[{}, {}, {}, {}]".format(
            self.left(), self.bottom(), self.right(), self.top()
        )

    def __repr__(self):
        return self.__str__()

    def isvalid(self):
        return self.right() > self.left() and self.top() > self.bottom()

    def to_list(self):
        return copy.copy(self.quadruple)

    def area(self):
        return self.height()*self.width()

    def set_right(self, val):
        self.quadruple[2] = val

    def right(self):
        return self.quadruple[2]

    def set_left(self, val):
        self.quadruple[0] = val

    def left(self):
        return self.quadruple[0]

    def bottom(self):
        return self.quadruple[1]

    def set_bottom(self, val):
        self.quadruple[1] = val

    def top(self):
        return self.quadruple[3]

    def set_top(self, val):
        self.quadruple[3] = val

    def height(self):
        return self.top()-self.bottom()

    def width(self):
        return self.right()-self.left()

    def center(self):
        return self.h_center(), self.v_center()

    def h_center(self):
        return (self.left()+self.right())/2.0

    def v_center(self):
        return (self.top()+self.bottom())/2.0

    def contains(self, center_pt):
        """
        whether center_pt is in the bbox

        :param center_pt:
        :return:
        """
        if center_pt[0] < self.left() or center_pt[0] > self.right():
            return False
        if center_pt[1] < self.bottom() or center_pt[1] > self.top():
            return False
        return True

    def contain_bbox(self, other_bbox):
        other_bbox = BBox(other_bbox)
        if other_bbox.left() < self.left():
            return False
        if other_bbox.right() > self.right():
            return False
        if other_bbox.top() > self.top():
            return False
        if other_bbox.bottom() < self.bottom():
            return False
        return True

    def contained_by_bbox(self, other_bbox):
        other_bbox = BBox(other_bbox)
        return other_bbox.contain_bbox(self)

    def h_overlap(self, other_bbox):
        if isinstance(other_bbox, tuple) or isinstance(other_bbox, list):
            other_bbox = BBox(other_bbox)
        if self.right() < other_bbox.left() or self.left() > other_bbox.right():
            return False
        return True

    def h_partial_overlap(self, other_bbox):
        """
        horizontally partially overlap
        :param other_bbox:
        :return:
        """
        if not self.h_overlap(other_bbox):
            return False
        if self.left() > other_bbox.left() and self.right() < other_bbox.right():
            return False
        if self.left() < other_bbox.left() and self.right() > other_bbox.right():
            return False
        return True

    def v_overlap(self, other_bbox, thres=None):
        if thres is None:
            if isinstance(other_bbox, tuple) or isinstance(other_bbox, list):
                other_bbox = BBox(other_bbox)
            if self.top() < other_bbox.bottom() or self.bottom() > other_bbox.top():
                return False
            return True
        else:
            if self.top() < other_bbox.bottom() or self.bottom() > other_bbox.top():
                return False
            min_top = min(self.top(), other_bbox.top())
            max_bottom = max(self.bottom(), other_bbox.bottom())
            overlap_height = (min_top-max_bottom)
            if overlap_height > 0.5 * self.height() and overlap_height > 0.5*other_bbox.height():
                return True
            else:
                return False

    def overlap(self, other_bbox):
        """
        test whether overlap or not
        :param other_bbox:
        :return:
        """
        #if not self.h_overlap(other_bbox) and not self.v_overlap(other_bbox):
        if self.h_overlap(other_bbox) and self.v_overlap(other_bbox):
            # overlap only if h-overlap and v-overlap
            return True
        return False

    def v_split(self, split_pos, delta=0.01):
        if split_pos - self.bottom() < 1e-8:
            return self, BBox([0, 0, 0, 0])
        if split_pos - self.top() < 1e-8:
            return BBox([0, 0, 0, 0]), self
        if split_pos > self.top():
            return BBox([0, 0, 0, 0]), self
        if split_pos < self.bottom():
            return self, BBox([0, 0, 0, 0])

        return BBox([self.left(), self.bottom(), self.right(), split_pos-delta]), \
            BBox([self.left(), split_pos+delta, self.right(), self.top()])

    def intersect(self, other_bbox):
        """
        only valid if they overlap
        :param other_bbox:
        :return:
        """
        if isinstance(other_bbox, tuple) or isinstance(other_bbox, list):
            other_bbox = BBox(other_bbox)

        if not self.overlap(other_bbox):
            raise Exception("not overlapping, could not intersect")
        return BBox([
            max(self.left(), other_bbox.left()),
            max(self.bottom(), other_bbox.bottom()),
            min(self.right(), other_bbox.right()),
            min(self.top(), other_bbox.top())
        ])

    def exclude_largest(self, other_bbox):
        """
        the original bbox might be split into multiple parts

        :param other_bbox:
        :return:
        """
        if self.contain_bbox(other_bbox):
            raise Exception("contain other bbox")
        if self.contained_by_bbox(other_bbox):
            raise Exception("contained by other bbox")
        bbox1 = None
        if self.left() < other_bbox.left():
            bbox1 = BBox([self.left(), self.bottom(), other_bbox.left(), self.top()])
        bbox2 = None
        if self.right() > other_bbox.right():
            bbox2 = BBox([other_bbox.right(), self.bottom(), self.right(), self.top()])
        if bbox1 and bbox2:
            return bbox1 if bbox1.area() > bbox2.area() else bbox2
        elif bbox1:
            return bbox1
        elif bbox2:
            return bbox2
        else:
            raise Exception("exclue bbox error")

    def dist(self, pt):
        """
        minimal distance to a point,
        the current implementation is based on the shortest distance

        :param pt:
        :return:
        """
        from pdf_parser.point import dist
        min_dist = dist((self.left(), self.bottom()), pt)
        min_dist = min(min_dist, dist((self.left(), self.top()), pt))
        min_dist = min(min_dist, dist((self.right(), self.top()), pt))
        min_dist = min(min_dist, dist((self.right(), self.bottom()), pt))
        return min_dist


def merge_bbox_list(bbox_list):
    """

    :param bbox_list:
    :type bbox_list: list[BBox]
    :return:
    """
    if len(bbox_list) == 0:
        ttt = 1
    left = np.min([bbox.left() for bbox in bbox_list])
    right = np.max([bbox.right() for bbox in bbox_list])
    bottom = np.min([bbox.bottom() for bbox in bbox_list])
    top = np.max([bbox.top() for bbox in bbox_list])
    return BBox([left, bottom, right, top])


def merge_bbox(bbox1, bbox2):
    if isinstance(bbox1, BBox):
        left = min(bbox1.left(), bbox2.left())
        right = max(bbox1.right(), bbox2.right())
        top = max(bbox1.top(), bbox2.top())
        bottom = min(bbox1.bottom(), bbox2.bottom())
        return BBox([left, bottom, right, top])
    else:
        left = min(bbox1[0], bbox2[0])
        right = max(bbox1[2], bbox2[2])
        top = max(bbox1[3], bbox2[3])
        bottom = min(bbox1[1], bbox2[1])
        return [left, bottom, right, top]


def bbox_h_overlapping(bbox1, bbox2):
    """

    :param bbox1:
    :param bbox2:
    :return:
    :rtype: bool
    """
    if isinstance(bbox1, BBox):
        if bbox1.left() > bbox2.right():
            return False
        if bbox1.right() < bbox2.left():
            return False
        return True
    else:
        if bbox1[0] > bbox2[2]:
            return False
        if bbox1[2] < bbox2[0]:
            return False
        return True


def bbox_v_overlapping(bbox1, bbox2):
    if isinstance(bbox1, BBox):
        if bbox1.top() < bbox2.bottom():
            return False
        if bbox1.bottom() > bbox2.top():
            return False
        return True
    else:
        if bbox1[3] < bbox2[1]:
            return False
        if bbox1[1] > bbox2[3]:
            return False
        return True


def point_in_bbox(pt, bbox):
    """

    :param pt:
    :type pt: Point
    :param bbox:
    :return:
    """
    print pt, bbox
    cond1 = bbox.left() <= pt.x() <= bbox.right()
    cond2 = bbox.bottom() <= pt.y() <= bbox.top()
    return (cond1 and cond2)

