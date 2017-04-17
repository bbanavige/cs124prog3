kk: kk.o 
	gcc -std=c99 -o kk -O3 kk.c
dynamic: dynamic.o
	gcc -std=c99 -o dynamic -O3 dynamic.c -lm
simulate: simulate.o
	gcc -std=c99 -o simulate -O3 simulate.c -lm
test: test.o
	gcc -std=c99 -o test -O3 test.c -lm