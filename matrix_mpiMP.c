#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include "mpi.h"

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
   int numprocs, rank, chunk_size, i, j;
   int *A, *B, *local_matrix, *result, *global_result, *seq_result;
   double start, end;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   
   chunk_size = N/numprocs;
   A = (int *)malloc(N*N*sizeof(int));
   B = (int *)malloc(N*N*sizeof(int));
   result = (int *)malloc(N*N*sizeof(int));
   local_matrix = (int *)malloc(N*N*sizeof(int));
   
   if (rank == 0) {
      seq_result = (int *)malloc(N*N*sizeof(int));
      global_result = (int *)malloc(N*N*sizeof(int));
      /* Initialize Matrices */
      for(i=0; i<N; i++) {
         for(j=0; j<N; j++) {
            *(A + i*N + j) = i+j;
            *(B + i*N + j) = i+j;
         }
      }
      start = MPI_Wtime();
   }

   MPI_Bcast(B, N*N, MPI_INT, 0 , MPI_COMM_WORLD);
   MPI_Scatter(A, N*chunk_size, MPI_INT, local_matrix, N*chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

   #pragma omp parallel for collapse(2)
      for(int i=0; i<chunk_size; i++) {
         for(int j=0; j<N; j++) {
            int temp = 0;
            for (int k=0; k<N; k++) {
               temp += *(local_matrix + i*N + k)* *(B + k*N + j);
            }
            *(result + i*N + j) = temp;
         }
   }


   MPI_Gather(result, N*chunk_size, MPI_INT, global_result, N*chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      //sequential_mult(A, B, seq_result);
      //compare_sequential(seq_result, global_result);
   }

   MPI_Barrier(MPI_COMM_WORLD);
   end = MPI_Wtime() - start;
   MPI_Finalize();
   if (rank == 0) { 
    printf("total time: %.2f seconds\n", end);
   }
   return 0;
} 

   