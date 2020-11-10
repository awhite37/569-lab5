#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 800

int main(int argc, char *argv[] ) {
   int i, j, k;
   int *A, *B, *result;
   clock_t start, end;

   A = (int *)malloc(N*N*sizeof(int));
   B = (int *)malloc(N*N*sizeof(int));
   result = (int *)malloc(N*N*sizeof(int));

   /* Initialize Matrices */
   for(i=0; i<N; i++) {
      for(j=0; j<N; j++) {
         *(A + i*N + j) = i+j;
         *(B + i*N + j) = i+j;
      }
   }
   start = clock();
   for(i=0; i<N; i++) {
      for(j=0; j<N; j++) {
         *(result + i*N + j) = 0;
         for (k=0; k<N; k++) {
            *(result + i*N + j) += *(A + i*N + k)* *(B + k*N + j);
         }
      }
   }
   
   end = clock();
   printf("total time: %.2f seconds\n", ((double)end-start)/CLOCKS_PER_SEC);
   return 0;
}  