/*

70940507 and 60940371

CS124 Coding Assignment 3

Usage: ./kk infile

*/

#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char* argv[]){
	if (argc != 2){
        printf("usage: ./kk infile\n");
        return -1;
    }

    char* infile = argv[1];

    char command[100];
	snprintf(command, sizeof(command), "python kk.py %s", infile);

    system(command);
}