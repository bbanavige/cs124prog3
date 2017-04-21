## CS124 Programming Assignment 3
## HUIDs: 70940507 and 60940371
## random_kk.py

from bisect import insort_left
import numpy as np
from copy import copy

import sys

if len(sys.argv) != 2:
	print("usage: python kk.py infile")
	quit()

infile = sys.argv[1]

A = np.loadtxt(infile).tolist()

def KK(A_p): # O(n*n), so not what we want
	A = copy(A_p)
	A.sort() # O(n log n)
	n = len(A)
	while n > 1: # O(n) times to insort * O(n) insort
		delta = abs(A.pop() - A.pop()) 
		insort_left(A, delta)
		n -= 1
	return A[0]

print(KK(A))