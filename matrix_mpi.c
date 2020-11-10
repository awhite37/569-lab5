#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "mpi.h"

#define N 800

int main(int argc, char *argv[] ) {
   int numprocs, rank, chunk_size, i, j, k;
   int *A, *B, *local_matrix, *result, *global_result;
   clock_t start, end;

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   
   chunk_size = N/numprocs;
   A = (int *)malloc(N*N*sizeof(int));
   B = (int *)malloc(N*N*sizeof(int));
   result = (int *)malloc(N*N*sizeof(int));
   global_result = (int *)malloc(N*N*sizeof(int));
   local_matrix = (int *)malloc(N*N*sizeof(int));
   
   if (rank == 0) {
      start = clock();
      /* Initialize Matrices */
      for(i=0; i<N; i++) {
         for(j=0; j<N; j++) {
            *(A + i*N + j) = i+j;
            *(B + i*N + j) = i+j;
         }
      }
   }

   MPI_Bcast(B, N*N, MPI_INT, 0 , MPI_COMM_WORLD);
   MPI_Scatter(A, N*chunk_size, MPI_INT, local_matrix, N*chunk_size, MPI_INT, 0, MPI_COMM_WORLD);
   
   for(i=0; i<chunk_size; i++) {
      for(j=0; j<N; j++) {
         *(result + i*N + j) = 0;
         for (k=0; k<N; k++) {
            *(result + i*N + j) += *(local_matrix + i*N + k)* *(B + k*N + j);
         }
      }
   }
   if (rank == 0) {
      /*do leftover work if numprocs not divisible by N*/
      if (N%numprocs != 0) {
         for(i=N-(N%numprocs); i<N; i++) {
            for(j=0; j<N; j++) {
               *(result + i*N + j) = 0;
               for (k=0; k<N; k++) {
                  *(global_result + i*N + j) += *(A + i*N + k)* *(B + k*N + j);
               }
            }
         }
      }
   }

   MPI_Gather(result, N*chunk_size, MPI_INT, global_result, N*chunk_size, MPI_INT, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      end = clock();
      printf("total time: %.2f seconds\n", ((double)end-start)/CLOCKS_PER_SEC);
   }

   MPI_Finalize();
   return 0;
} 

   