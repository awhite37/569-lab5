#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "mpi.h"

//using numprocs = 9, N must be divisible by sqrt(numprocs) 
#define N 5001

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

int main(int argc, char* argv[]) {
   int numprocs, rank, next, prev, pivot;
   int *A, *B, *global_result, *seq_result;
   int *localA, *localB, *local_result, *tempA;
   double start, end;
   MPI_Status Status;
   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   int GridCoords[2]; 
   MPI_Comm GridComm; 
   MPI_Comm ColComm; 
   MPI_Comm RowComm; 

   int num_chunks = sqrt(numprocs);
   int chunk_size = N/num_chunks;

   //initialize 2D grid of processes
   int dimension_size[2]; 
   int periodic[2]; 
   int subdims[2]; 
   dimension_size[0] = num_chunks; 
   dimension_size[1] = num_chunks;
   periodic[0] = 0;
   periodic[1] = 0;
   MPI_Cart_create(MPI_COMM_WORLD, 2, dimension_size, periodic, 1, &GridComm);
   MPI_Cart_coords(GridComm, rank, 2, GridCoords);
   subdims[0] = 0;
   subdims[1] = 1;
   MPI_Cart_sub(GridComm, subdims, &RowComm);
   subdims[0] = 1;
   subdims[1] = 0;
   MPI_Cart_sub(GridComm, subdims, &ColComm);

   localA = (int *)malloc(chunk_size*chunk_size*sizeof(int));
   tempA = (int *)malloc(chunk_size*chunk_size*sizeof(int));
   localB = (int *)malloc(chunk_size*chunk_size*sizeof(int));
   local_result = (int *)malloc(chunk_size*chunk_size*sizeof(int));
   for (int i=0; i<chunk_size*chunk_size; i++) {
      local_result[i] = 0;
   }

   if (rank == 0) {
      seq_result = (int *)malloc(N*N*sizeof(int));
      global_result = (int *)malloc(N*N*sizeof(int));
      A = (int *)malloc(N*N*sizeof(int));
      B = (int *)malloc(N*N*sizeof(int));
      /* Initialize Matrices */
      for(int i=0; i<N; i++) {
         for(int j=0; j<N; j++) {
            *(A + i*N + j) = i+j;
            *(B + i*N + j) = i+j;
         }
      }
      start = MPI_Wtime();
   }

   //divide matrix A blocks
   int *row = (int *)malloc(N*chunk_size*sizeof(int));
   if (GridCoords[1] == 0) {
      MPI_Scatter(A, N*chunk_size, MPI_INT, row, N*chunk_size, MPI_INT, 0, ColComm);
   }
   for (int i=0; i<chunk_size; i++) {
      MPI_Scatter((row + i*N), chunk_size, MPI_INT,(tempA + i*chunk_size), chunk_size, MPI_INT, 0, RowComm);
   }
   //divide matrix B blocks
   if (GridCoords[1] == 0) {
      MPI_Scatter(B, N*chunk_size, MPI_INT, row, N*chunk_size, MPI_INT, 0, ColComm);
   }
   for (int i=0; i<chunk_size; i++) {
      MPI_Scatter((row + i*N), chunk_size, MPI_INT,(localB + i*chunk_size), chunk_size, MPI_INT, 0, RowComm);
   }

   // do multiplying
   for (int l = 0; l < num_chunks; l++) {
      //broadcast A block row-wise
      pivot = (GridCoords[0] + l) % num_chunks;
      if (GridCoords[1] == pivot) {
         for (int j=0; j<chunk_size*chunk_size; j++) {
            localA[j] = tempA[j];
         }
      }
      MPI_Bcast(localA, chunk_size*chunk_size, MPI_INT, pivot, RowComm);

      //local block multiplication
      #pragma omp parallel for collapse(2)
      for(int i=0; i<chunk_size; i++) {
         for(int j=0; j<chunk_size; j++) {
            for (int k=0; k<chunk_size; k++) {
               *(local_result + i*chunk_size + j) += *(localA + i*chunk_size + k)* *(localB + k*chunk_size + j);
            }
         }
      }
      //rotate B blocks column-wise
      next = GridCoords[0] + 1;
      if (GridCoords[0] == num_chunks-1) {
         next = 0;
      }
      prev = GridCoords[0] - 1;
      if (GridCoords[0] == 0) {
         prev = num_chunks-1;
      }
      MPI_Sendrecv_replace(localB, chunk_size*chunk_size, MPI_INT, prev, 0, next, 0, ColComm, &Status); 
   }

   //gather result blocks in global_result matrix
   for (int i=0; i<chunk_size; i++) {
      MPI_Gather((local_result + i*chunk_size), chunk_size, MPI_INT, (row + i*N), chunk_size, MPI_INT, 0, RowComm);
   }
   if (GridCoords[1] == 0) {
      MPI_Gather(row, N*chunk_size, MPI_INT, global_result, N*chunk_size, MPI_INT, 0, ColComm);
   }

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
