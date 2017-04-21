kk: kk.o 
	gcc -std=c99 -o kk -O3 kk.c
simulate: simulate.o
	gcc -std=c99 -o simulate -O3 simulate.c -lm