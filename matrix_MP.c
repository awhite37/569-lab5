#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define N 5000

void sequential_mult(int *A, int *B, int *result) {
   int i,j,k;
   for(i=0; i<N; i++) {
      for(j=0; j<N; j++) {
         *(result + i*N + j) = 0;
         for (k=0; k<N; k++) {
            *(result + i*N + j) += *(A + i*N + k)* *(B + k*N + j);
         }
      }
   }
}

void compare_sequential(int *res1, int *res2) {
   int i,j;
   for(i=0; i<N; i++) {
      for(j=0; j<N; j++) {
         if (*(res1 + i*N + j) != *(res2 + i*N + j)) {
            printf("outputs don't match!\n");
            return;
         }
      }
   }
   printf("output matches sequential\n");
}

int main(int argc, char *argv[] ) {
   int *A, *B, *result, *seq_result;
   double timeStart, timeFinish;

   A = (int *)malloc(N*N*sizeof(int));
   B = (int *)malloc(N*N*sizeof(int));
   result = (int *)malloc(N*N*sizeof(int));
   seq_result = (int *)malloc(N*N*sizeof(int));

   for(int i=0; i<N; i++) {
      for(int j=0; j<N; j++) {
         *(A + i*N + j) = i+j;
         *(B + i*N + j) = i+j;
      }
   }

   timeStart=omp_get_wtime();

   #pragma omp parallel for collapse(2)
      for(int i=0; i<N; i++) {
         for(int j=0; j<N; j++) {
            int temp = 0;
            for (int k=0; k<N; k++) {
               temp += *(A + i*N + k)* *(B + k*N + j);
            }
            *(result + i*N + j) = temp;
         }
      }

   timeFinish=omp_get_wtime();
   //sequential_mult(A, B, seq_result);
   //compare_sequential(seq_result, result);
   printf("total time: %.2f seconds\n", (timeFinish-timeStart));


}