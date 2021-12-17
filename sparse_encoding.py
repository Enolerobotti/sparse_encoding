import scipy.sparse
import random
from typing import Optional


def sparse_one_hot_encoding(classes: int, randomize: Optional[bool] = True) -> scipy.sparse.coo_matrix:
    samples = list(range(classes))
    row = []
    data = []
    for i in range(1, classes):
        row += samples
        data.extend([1]*i + [0] * (classes - i))
    col = [0]*len(data)
    if randomize:
        row, data = shuffle(row, data)
    return scipy.sparse.coo_matrix((data, (row, col)), shape=(classes, 1))


def shuffle(*args):
    lst = list(zip(*args))
    random.shuffle(lst)
    return zip(*lst)


if __name__ == '__main__':
    classes_range = 10
    m = sparse_one_hot_encoding(classes_range)
    print(m.A)
    for j in range(classes_range):
        print(m.data[m.row == j])
