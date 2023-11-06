import copy
from unittest import TestCase
from r0758170 import *
import random as rd
import numpy as np


def seed(the_seed):
    rd.seed(the_seed)
    np.random.seed(the_seed)


def create_random_candidate(size: int) -> Candidate:
    array = np.array(list(range(size)), dtype=int)
    np.random.shuffle(array)
    return Candidate(array)


class Test(TestCase):

    def test_mutate_inversion(self):
        a = Candidate(np.array(list(range(1, 10)), dtype=int))
        a.mutate_func = mutate_inversion
        a.mutate_prob = 1.0
        target = copy.deepcopy(a)
        target.array[:] = np.array([1, 5, 4, 3, 2, 6, 7, 8, 9], dtype=int)
        a.mutate(1, 4)
        self.assertEqual(target, a)

    def test_recombine_pmx(self):
        a = Candidate(np.array(list(range(1, 10)), dtype=int))
        b = Candidate(np.array([9, 3, 7, 8, 2, 6, 5, 1, 4], dtype=int))
        a.recombine_func = recombine_PMX
        a.recombine_prob = 1.0
        c = copy.deepcopy(a)
        d = copy.deepcopy(a)
        a.recombine(b, c, d, 3, 6)
        target_c = Candidate(np.array([9, 3, 2, 4, 5, 6, 7, 1, 8], dtype=int))
        target_d = Candidate(np.array([1, 7, 3, 8, 2, 6, 5, 4, 9], dtype=int))
        self.assertEqual(target_c, c)
        self.assertEqual(target_d, d)

    def test_recombine_order_crossover(self):
        a = Candidate(np.array(list(range(1, 10)), dtype=int))
        b = Candidate(np.array([9, 3, 7, 8, 2, 6, 5, 1, 4], dtype=int))
        a.recombine_func = recombine_order_crossover
        a.recombine_prob = 1.0
        c = copy.deepcopy(a)
        d = copy.deepcopy(a)
        a.recombine(b, c, d, 3, 6)
        target_c = Candidate(np.array([3, 8, 2, 4, 5, 6, 7, 1, 9], dtype=int))
        target_d = Candidate(np.array([3, 4, 7, 8, 2, 6, 5, 9, 1], dtype=int))
        self.assertEqual(target_c, c)
        self.assertEqual(target_d, d)
