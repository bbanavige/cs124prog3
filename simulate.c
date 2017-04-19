/*

70940507 and 30978137

CS124 Coding Assignment 2
Program for generating random input files

Usage: ./simulate flag dimension outfile

*/
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int n_default = 100;
int max = 1000000000001;
// int max = 100001;

int main(int argc, char* argv[]) {
	if (argc != 2 && argc != 3){
        printf("usage: ./simulate outfile <n>");
        return -1;
    }

    int n;

    if (argc == 2){
        n = n_default;
    }
    else {
        n = atoi(argv[2]);
    }

    char* outfile = argv[1];

    FILE* out = fopen(outfile, "w");
    if (!out) {
        printf("Error opening file.\n");
        return -2;
    }
    
    for (int i = 0; i < n; i++) {
        int num = (rand() % max);
        fprintf(out, "%d\n", num); 
    }
    
}
