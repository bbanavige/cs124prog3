import numpy as np
import random

file = "test.txt"

A = np.loadtxt(file)

print(A)

def random_kk(A, n, iter = 25000):

	res_best = np.infty

	for it in range(iter):
		S = [random.choice([-1, 1]) for i in range(n)]

		res = abs(sum([s * a for s,a in zip(S, A)]))

		if res < res_best:
			S_best = S
			res_best = res


	print("best residue:", res_best)

random_kk(A, 100)


def hill_kk(A):
	# TODO
	return


def anneal_kk(A):
	# TODO
	return