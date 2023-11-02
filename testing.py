import numpy as np

from r0758170 import Candidate


def create_random_array(size: int):
    array = np.array(list(range(size)))
    np.random.shuffle(array)
    return array


# a = Candidate(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
# b = Candidate(np.array([9, 3, 7, 8, 2, 6, 5, 1, 4]))

class Foo:
    def __init__(self):
        self.func = Candidate.mutate_swap

    def bar(self):
        a = Candidate(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        # self.func(a)
        # Candidate.mutate_swap(a)
        self.func(a)
        print(a)


foo = Foo()
foo.bar()
