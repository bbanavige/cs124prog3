import numpy as np
import random

file = "test.txt"

n_int = 100

A = np.loadtxt(file)

def KK(A, n):
	# TODO
	return

def random_s(A, n, iter = 25000):

	res_best = np.infty

	for it in range(iter):
		S = [random.choice([-1, 1]) for i in range(n)]

		res = abs(sum([s * a for s,a in zip(S, A)]))

		if res < res_best:
			S_best = S
			res_best = res

	print("rand best residue:\t", res_best)
	return res_best


def hill_s(A, n, iter = 25000):
	S_best = [random.choice([-1, 1]) for i in range(n)]
	res_best = np.infty

	for it in range(iter):
		r_index1 = random.randint(0, n - 1)
		r_index2 = random.randint(0, n - 1)
		while  r_index1 == r_index2:
			r_index2 = random.randint(0, n - 1)

		switch = random.random() < 0.5

		S = S_best
		S[r_index1] = S_best[r_index1] * -1

		if switch:
			S[r_index2] = S_best[r_index2] * -1
			

		res = abs(sum([s * a for s,a in zip(S, A)]))

		if res < res_best:
			S_best = S
			res_best = res

	print("hill best residue:\t", res_best)
	return res_best

# returns x ^ y using the fast(er) method
def fast_expo(x,y):
  # place to store powers of x (mod z)
  powers = []
  n = int(np.ceil(np.log2(float(y + 1))))
  powers.append(x)
  for i in range(1, n + 1):
    powers.append(powers[i - 1] ** 2)
  y_bin = [int(a) for a in bin(y)[2:]][::-1]
  ans = 1
  for i in range(0, n):
    if y_bin[i]:
      ans = ans * powers[i]
  return(ans)

def T(iter):
	return 10 ** 10 * fast_expo(0.8, int(np.floor(iter / 300)))


def anneal_s(A, n, iter = 25000):
	S_best = S_cur = [random.choice([-1, 1]) for i in range(n)]
	res_best = res_cur = np.infty

	for it in range(iter):
		r_index1 = random.randint(0, n - 1)
		r_index2 = random.randint(0, n - 1)
		while  r_index1 == r_index2:
			r_index2 = random.randint(0, n - 1)

		switch = random.random() < 0.5

		S = S_cur
		S[r_index1] = S_cur[r_index1] * -1

		if switch:
			S[r_index2] = S_cur[r_index2] * -1
			

		res = abs(sum([s * a for s,a in zip(S, A)]))

		if res < res_cur:
			S_cur = S
			res_cur = res
		elif random.random() < T(it):
			S_cur = S
			res_cur = res

		if res < res_best:
			S_best = S
			res_best = res


	print("anneal best residue:\t", res_best)
	return res_best

def random_p(A, n, iter = 2500):
	res_best = np.inf
	for it in range(iter):
		values = {}
		
		for i in range(n):
			p = random.randint(1, n)
			values.setdefault(p, 0)
			values[p] += A[i]
		A_prime = list(values.values())
		res = KK(A_prime, n_int)
		if res < res_best:
			res_best = res

	print("random best residue:\t", res_best)
	return res_best

def hill_p(A, n, iter = 25000):
	res_best = np.inf
	p_best = [random.randint(1, n) for i in range(n)]
	for it in range(iter):
		values = {}
		
		ind = random.randint(1, n)

		p = p_best

		p_cur[ind] = random.randint(1, n)

		for i in range(n):
			values.setdefault(p[i], 0)
			values[p[i]] += A[i]

		A_prime = list(values.values())
		res = KK(A_prime, n_int)
		if res < res_best:
			p_best = p
			res_best = res

	print("hill best residue:\t", res_best)
	return res_best


def anneal_s(A, n, iter = 25000):
	p_best = p_cur = [random.randint(1, n) for i in range(n)]
	res_best = res_cur = np.infty

	for it in range(iter):
		values = {}
		
		ind = random.randint(1, n)

		p = p_cur

		p[ind] = random.randint(1, n)

		for i in range(n):
			values.setdefault(p[i], 0)
			values[p[i]] += A[i]

		A_prime = list(values.values())
		res = KK(A_prime, n_int)

		if res < res_cur:
			p_cur = p
			res_cur = res
		elif random.random() < T(it):
			res_cur = res

		if res < res_best:
			res_best = res
			p_best = p


	print("anneal best residue:\t", res_best)
	return res_best


# random_kk(A, n_int)
# hill_kk(A, n_int)
# anneal_kk(A, n_int)
random_p(A, n_int)




