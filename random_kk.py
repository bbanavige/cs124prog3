## CS124 Programming Assignment 3
## HUIDs: 70940507 and 60940371
## random_kk.py

import numpy as np
import random
from time import time
import matplotlib.pyplot as plt
from copy import copy

file = "test.txt"

n_iter = 25000

A = np.loadtxt(file).tolist()

n_int = len(A)

n_max = 10 ** 12

n_test = 10

def DP(A, N, print = False):
	tic = time()
	b = int(np.sum(A))
	T = [[-1 for k in range(b + 1)] for n in range(N + 1)]
	for k in range(b + 1):
		T[0][k] = False
	for n in range(N + 1):
		T[n][0] = True
	for k in range(1, b + 1):
		for n in range(1, N + 1):
			if A[n - 1] <= k:
				T[n][k] = T[n-1][k] or T[n - 1][k - int(A[n - 1])]
			else:
				T[n][k] = T[n - 1][k]
	min = b
	for k in range(b + 1):
		if T[n][k]:
			diff = np.abs(2 * k - b)
			if diff < min:	
				min = diff
	if print:
		print("best residue:\t", min)
		print("time:", time() - tic)
	return min


from bisect import insort_left

def KK(A_p, time_ = False): # O(n*n), so not what we want
	tic = time()
	A = copy(A_p)
	A.sort() # O(n log n)
	n = len(A)
	while n > 1: # O(n) times to insort * O(n) insort
		delta = abs(A.pop() - A.pop()) 
		insort_left(A, delta)
		n -= 1
	if time_:
		return [A[0], time() - tic]
	return A[0]


def random_s(A, n, iter = n_iter, print = False, time_ = False):

	res_best = np.infty

	tic = time()

	for it in range(iter):
		S = [random.choice([-1, 1]) for i in range(n)]

		res = abs(sum([s * a for s,a in zip(S, A)]))

		if res < res_best:
			S_best = S
			res_best = res

		if res_best == 0:
			if print: print("iterations taken:\t", it)
			break

	if print:
		print("best residue:\t", res_best)
		print("time:\t", time() - tic)
	if time_:
		return [res_best, time() - tic]
	return res_best


def hill_s(A, n, iter = n_iter, print = False, time_ = False):
	S_best = [random.choice([-1, 1]) for i in range(n)]
	res_best = np.infty

	tic = time()

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

		if res_best == 0:
			if print: print("iterations taken:\t", it)
			break

	if print:
		print("best residue:\t", res_best)
		print("time:\t", time() - tic)
	if time_:
		return [res_best, time() - tic]
	return res_best

# returns x ^ y using the fast(er) method
def fast_expo(x,y):
	# place to store powers of x
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


def anneal_s(A, n, iter = n_iter, print = False, time_ = False):
	S_best = S_cur = [random.choice([-1, 1]) for i in range(n)]
	res_best = res_cur = np.infty

	tic = time()

	T_list = [10 ** 10 * fast_expo(0.8, i) for i in range(int(np.floor(iter / 300)) + 1)]

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
		elif random.random() < T_list[int(np.floor(iter / 300))]:
			S_cur = S
			res_cur = res

		if res < res_best:
			S_best = S
			res_best = res

		if res_best == 0:
			if print: print("iterations taken:\t", it)
			break

	if print:
		print("best residue:\t", res_best)
		print("time:\t", time() - tic)
	if time_:
		return [res_best, time() - tic]
	return res_best

def random_p(A, n, iter = n_iter, print = False, time_ = False):
	res_best = np.inf
	tic = time()

	for it in range(iter):
		values = {}
		for i in range(n):
			p = random.randint(1, n)
			values.setdefault(p, 0)
			values[p] += A[i]
		A_prime = list(values.values())
		res = KK(A_prime)
		if res < res_best:
			res_best = res
		if res_best == 0:
			if print: print("iterations taken:\t", it)
			it_taken = it
			break
	if print:
		print("best residue:\t", res_best)
		print("time:\t", time() - tic)
	if time_:
		return [res_best, time() - tic]
	return res_best

def hill_p(A, n, iter = n_iter, print = False, time_ = False):
	res_best = res_cur = np.inf
	p_best = p_cur = [random.randint(1, n) for i in range(n)]
	tic = time()

	for it in range(iter):
		values = {}
		
		ind = random.randint(0, n-1)

		p = p_best

		p_cur[ind] = random.randint(1, n)

		for i in range(n):
			values.setdefault(p[i], 0)
			values[p[i]] += A[i]

		A_prime = list(values.values())
		res = KK(A_prime)
		if res < res_best:
			p_best = p
			res_best = res

		if res_best == 0:
			if print: print("iterations taken:\t", it)
			break

	if print:
		print("best residue:\t", res_best)
		print("time:\t", time() - tic)
	if time_:
		return [res_best, time() - tic]
	return res_best


def anneal_p(A, n, iter = n_iter, print = False, time_ = False):
	p_best = p_cur = [random.randint(1, n) for i in range(n)]
	res_best = res_cur = np.infty

	tic = time()
	for it in range(iter):
		values = {}
		
		ind = random.randint(0, n - 1)

		p = p_cur

		p[ind] = random.randint(0, n - 1)

		for i in range(n):
			values.setdefault(p[i], 0)
			values[p[i]] += A[i]
		A_prime = list(values.values())
		res = KK(A_prime)
		if res < res_cur:
			p_cur = p
			res_cur = res
		elif random.random() < T(it):
			res_cur = res
		if res < res_best:
			res_best = res
			p_best = p
		if res_best == 0:
			if print: print("iterations taken:\t", it)
			break
	if print: 
		print("best residue:\t", res_best)
		print("time:\t", time() - tic)
	if time_:
		return [res_best, time() - tic]
	return res_best

## analysis

kk_100 = np.zeros([100])
random_s_100 = np.zeros([100])
hill_s_100 = np.zeros([100])
anneal_s_100 = np.zeros([100])
random_p_100 = np.zeros([100])
hill_p_100 = np.zeros([100])
anneal_p_100 = np.zeros([100])
kk_time = np.zeros([100])
random_s_time = np.zeros([100])
hill_s_time = np.zeros([100])
anneal_s_time = np.zeros([100])
random_p_time = np.zeros([100])
hill_p_time = np.zeros([100])
anneal_p_time = np.zeros([100])

for i in range(100):
	print(i)
	A = [random.randint(1, n_max) for j in range(n_int)]
	[kk_100[i], kk_time[i]] = KK(A, time_ = True)
	[random_s_100[i], random_s_time[i]] = random_s(A, n_int, time_ = True)
	[hill_s_100[i], hill_s_time[i]] = hill_s(A, n_int, time_ = True)
	[anneal_s_100[i], anneal_s_time[i]] = anneal_s(A, n_int, time_ = True)
	[random_p_100[i], random_p_time[i]] = random_p(A, n_int, time_ = True)
	[hill_p_100[i], hill_p_time[i]] = hill_p(A, n_int, time_ = True)
	[anneal_p_100[i], anneal_p_time[i]] = anneal_p(A, n_int, time_ = True)

print("heuristic\tresidue\ttime")
print("random s\t", np.mean(random_s_100),"\t", np.mean(random_s_time))
print("hill s\t", np.mean(hill_s_100),"\t", np.mean(hill_s_time))
print("anneal s\t", np.mean(anneal_s_100),"\t", np.mean(anneal_s_time))
print("random p\t", np.mean(random_p_100),"\t", np.mean(random_p_time))
print("hill p\t", np.mean(hill_p_100),"\t", np.mean(random_s_time))
print("anneal p\t", np.mean(anneal_p_100),"\t", np.mean(anneal_p_time))

## second analysis

kk_result = np.zeros([n_test])
random_s_result = np.zeros([n_test, len(iter_vals), sims])
hill_s_result = np.zeros([n_test, len(iter_vals), sims])
anneal_s_result = np.zeros([n_test, len(iter_vals), sims])
random_p_result = np.zeros([n_test, len(iter_vals), sims])
hill_p_result = np.zeros([n_test, len(iter_vals), sims])
anneal_p_result = np.zeros([n_test, len(iter_vals), sims])

for i in range(n_test):
	A = [random.randint(1, n_max) for j in range(n_int)]
	# DP_result[i] = DP(A, n_int)
	kk_result[i] = KK(A)
	for j in range(len(iter_vals)):
		# if j % 5 == 0:
		print(iter_vals[j])
		for k in range(sims):

			random_s_result[i][j][k] = random_s(A,n_int, iter = iter_vals[j])
			hill_s_result[i][j][k] = hill_s(A,n_int, iter = iter_vals[j])
			anneal_s_result[i][j][k] = anneal_s(A,n_int, iter = iter_vals[j])
			random_p_result[i][j][k] = random_p(A,n_int, iter = iter_vals[j])
			hill_p_result[i][j][k] = hill_p(A,n_int, iter = iter_vals[j])
			anneal_p_result[i][j][k] = anneal_p(A,n_int, iter = iter_vals[j])

from pylab import *
ion()

plt.figure(figsize = (8,8))
plt.suptitle("Testing Randomized Convergence: P representation")
for i in range(n_test):
	# plt.subplot(2,2,i+1)
	plt.semilogy(iter_vals, [np.mean(random_p_result[i][j]) for j in range(len(iter_vals))], label = "random")
	plt.semilogy(iter_vals, [np.mean(hill_p_result[i][j]) for j in range(len(iter_vals))], label = "hill")
	plt.semilogy(iter_vals, [np.mean(anneal_p_result[i][j]) for j in range(len(iter_vals))], label = "anneal")
	# plt.title("Test {}".format(i + 1))
	plt.legend()


# # plot2
plt.figure(figsize = (8,8))
plt.suptitle("Testing Randomized Convergence: P representation")
for i in range(n_test):
	# plt.subplot(2,2,i+1)
	plt.semilogy(iter_vals, [np.mean(random_s_result[i][j]) for j in range(len(iter_vals))], label = "random")
	plt.semilogy(iter_vals, [np.mean(hill_s_result[i][j]) for j in range(len(iter_vals))], label = "hill")
	plt.semilogy(iter_vals, [np.mean(anneal_s_result[i][j]) for j in range(len(iter_vals))], label = "anneal")
	# plt.title("Test {}".format(i + 1))
	plt.legend()







