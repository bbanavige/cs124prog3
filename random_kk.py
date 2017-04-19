import numpy as np
import random
from time import time

file = "test.txt"

n_int = 100

n_iter = 25000

A = np.loadtxt(file).tolist()

def DP(A, N):
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
	print(b)
	for k in range(b + 1):
		if T[n][k]:
			diff = np.abs(2 * k - b)
			if diff < min:	
				min = diff
	print("best residue:\t", min)
	print("time:", time() - tic)
	return min


from bisect import insort_left

def KK(A): # O(n*n), so not what we want
	A.sort() # O(n log n)
	n = len(A)
	while n > 1: # O(n) times to insort * O(n) insort
		delta = abs(A.pop() - A.pop()) 
		insort_left(A, delta)
		n -= 1
	return A[0]


# 2nd 
# import heapq 
# import numpy as np

# # def kk_alg(A):
# # 	# heapq is a min heap, min of negative is max 
# # 	A = list(-1 * np.array(A))
# # 	heapq.heapify(A)
# # 	for _ in range(len(A) - 1):
# # 		delta = -1 * abs(heapq.heappop(A) - heapq.heappop(A))
# # 		heapq.heappush(A, delta)
# # 	return A[0]


def random_s(A, n, iter = n_iter):

	res_best = np.infty

	tic = time()

	for it in range(iter):
		S = [random.choice([-1, 1]) for i in range(n)]

		res = abs(sum([s * a for s,a in zip(S, A)]))

		if res < res_best:
			S_best = S
			res_best = res

		if res_best == 0:
			print("iterations taken:\t", it)
			break

	print("best residue:\t", res_best)
	print("time:\t", time() - tic)
	return res_best


def hill_s(A, n, iter = n_iter):
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
			print("iterations taken:\t", it)
			break

	print("best residue:\t", res_best)
	print("time:\t", time() - tic)
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


def anneal_s(A, n, iter = n_iter):
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
			print("iterations taken:\t", it)
			break


	print("best residue:\t", res_best)
	print("time:\t", time() - tic)
	return res_best

def random_p(A, n, iter = n_iter):
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
			print("iterations taken:\t", it)
			break
	print("best residue:\t", res_best)
	print("time:\t", time() - tic)
	return res_best

def hill_p(A, n, iter = n_iter):
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
			print("iterations taken:\t", it)
			break

	print("best residue:\t", res_best)
	print("time:\t", time() - tic)
	return res_best


def anneal_p(A, n, iter = n_iter):
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
			print("iterations taken:\t", it)
			break


	print("best residue:\t", res_best)
	print("time:\t", time() - tic)
	return res_best

# print("running DP solution")
# DP(A, n_int)

print("\nrunning processes\n\nFIRST REPRESENTATION\n\nrandom method")
random_s(A, n_int)
print("\nhill method")
hill_s(A, n_int)
print("\nanneal method")
anneal_s(A, n_int)

print("\nSECOND REPRESENTATION\n\nrandom method")
random_p(A, n_int)
print("\nhill method")
hill_p(A, n_int)
print("\nanneal method")
anneal_p(A, n_int)
print("\ndone")





