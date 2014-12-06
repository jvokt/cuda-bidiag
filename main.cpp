// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/time.h>

/* Using updated (v2) interfaces to cublas */
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include <cublas_v2.h>

#include "mmio.h"

/* genTridiag: generate a random tridiagonal symmetric matrix */
void genTridiag(int *I, int *J, double *val, int N, int nz)
{
    I[0] = 0, J[0] = 0, J[1] = 1;
    val[0] = (double)rand()/RAND_MAX + 10.0f;
    val[1] = (double)rand()/RAND_MAX;
    int start;

    for (int i = 1; i < N; i++)
    {
        if (i > 1)
        {
            I[i] = I[i-1]+3;
        }
        else
        {
            I[1] = 2;
        }

        start = (i-1)*3 + 2;
        J[start] = i - 1;
        J[start+1] = i;

        if (i < N-1)
        {
            J[start+2] = i + 1;
        }

        val[start] = val[start-1];
        val[start+1] = (double)rand()/RAND_MAX + 10.0f;

        if (i < N-1)
        {
            val[start+2] = (double)rand()/RAND_MAX;
        }
    }

    I[N] = nz;
}

int main(int argc, char **argv)
{
    int M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
    double *val = NULL;
    const double tol = 1e-5f;
    const int max_iter = 10000;
    double *r, *p, *U, *V;
    double *alpha, *beta;
    int *d_col, *d_row, *d_rowcoo;
    double *d_val;
    double *d_r, *d_p, *d_U, *d_V;
    double *d_alpha, *d_beta;
    double *d_errormat;

    /* Get handle to the CUBLAS context */
    cublasHandle_t cublasHandle = 0;
    cublasStatus_t cublasStatus;
    cublasStatus = cublasCreate(&cublasHandle);

    /* Get handle to the CUSPARSE context */
    cusparseHandle_t cusparseHandle = 0;
    cusparseStatus_t cusparseStatus;
    cusparseStatus = cusparseCreate(&cusparseHandle);

    cusparseMatDescr_t descr = 0;
    cusparseStatus = cusparseCreateMatDescr(&descr);
    
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);
    
    /* Generate a random tridiagonal symmetric matrix in CSR format */
    /*
    M = N = 1048576;
    nz = (N-2)*3 + 4;
    I = (int *)malloc(sizeof(int)*(N+1));
    J = (int *)malloc(sizeof(int)*nz);
    val = (double *)malloc(sizeof(double)*nz);
    genTridiag(I, J, val, N, nz);
    */
    
    /* Load the MovieLens dataset with 1 million total movie ratings */
    char const * file = "movielens_1M.mtx";
    mm_read_unsymmetric_sparse(file, &M, &N, &nz, &val, &I, &J);
    printf("M: %d, N: %d, nz: %d\n",M,N,nz);
    
    struct timeval t1, t2;
    gettimeofday(&t1, 0);
    
    // Set number of required singular values K
    int K = 500;
    // Allocate host memory for alpha (size K)
    alpha = (double *)malloc(sizeof(double)*K);
    // Allocate host memory for beta (size K+1)
    beta = (double *)malloc(sizeof(double)*(K+1));
    // Allocate memory for U (size M by K)
    U = (double *)malloc(sizeof(double)*M*K);
    // Allocate memory for V (size N by K)
    V = (double *)malloc(sizeof(double)*N*K);
    // Initialize p to a random vector, normalize p (size M)
    p = (double *)malloc(sizeof(double)*M);
    double norm2 = 0;
    for (int i = 0; i < M; i++)
    {
      p[i] = (double)rand()/RAND_MAX + 10.0f;
      norm2 += p[i] * p[i];
    }
    double norm = sqrt(norm2);
    for (int i = 0; i < M; i++)
    {
      p[i] /= norm;
    }

    // Allocate device memory
    cudaMalloc((void **)&d_U, M*K*sizeof(double));
    cudaMalloc((void **)&d_V, N*K*sizeof(double));
    cudaMalloc((void **)&d_p, M*sizeof(double));
    cudaMalloc((void **)&d_r, N*sizeof(double));
    cudaMalloc((void **)&d_col, nz*sizeof(int));
    cudaMalloc((void **)&d_row, (N+1)*sizeof(int));
    cudaMalloc((void **)&d_rowcoo, nz*sizeof(int));
    cudaMalloc((void **)&d_val, nz*sizeof(double));

    // Copy to device memory
    cudaMemcpy(d_col, J, nz*sizeof(int), cudaMemcpyHostToDevice);
    //    cudaMemcpy(d_row, I, (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rowcoo, I, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, val, nz*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_p, p, M*sizeof(double), cudaMemcpyHostToDevice);

    // Convert matrix from COO to CSR
    cusparseXcoo2csr(cusparseHandle, d_rowcoo, nz, M, d_row, CUSPARSE_INDEX_BASE_ZERO);

    // Helper variables for BLAS calls
    double zero = 0;
    double one = 1;
    double minus = -1;
    double work = 0;

    // Initialize beta
    beta[0] = 1;
    int k = 0;
    // While k <= K and beta(k) > 0
    while (k < K && beta[k] > 0)
    {
      if (k > 0)
      {
	// U(:,k) = p/beta(k);
	work = 1/beta[k];
	cublasStatus = cublasDscal(cublasHandle, N, &work, d_p, 1);
	cublasStatus = cublasDcopy(cublasHandle, M, d_p, 1, &d_U[k*M], 1);
	// r = A'*U(:,k) - beta(k)*V(:,k-1);	
	cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, M, N, nz, 
		       &one, descr, d_val, d_row, d_col, &d_U[k*M], &zero, d_r);
	work = -beta[k];
	cublasStatus = cublasDaxpy(cublasHandle, N, &work, &d_V[(k-1)*N], 1, d_r, 1);
      } else {
	// U(:,k) = p;
	cublasStatus = cublasDcopy(cublasHandle, M, d_p, 1, &d_U[k*M], 1);
	// r = A'*U(:,k);	
	cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_TRANSPOSE, M, N, nz, 
		       &one, descr, d_val, d_row, d_col, &d_U[k*M], &zero, d_r);
      }
      // alpha(k) = norm(r);
      cublasDnrm2(cublasHandle, N, d_r, 1, &alpha[k]);
      // V(:,k) = r/alpha(k);
      work = 1/alpha[k];
      cublasStatus = cublasDscal(cublasHandle, N, &work, d_r, 1);
      cublasStatus = cublasDcopy(cublasHandle, N, d_r, 1, &d_V[k*M], 1);
      // p = A*V(:,k) - alpha(k)*U(:,k);
      cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, N, nz, 
		     &one, descr, d_val, d_row, d_col, &d_V[k*M], &zero, d_p);
      work = -alpha[k];
      cublasStatus = cublasDaxpy(cublasHandle, N, &work, &d_U[k*N], 1, d_p, 1);
      if (k < N)
      {
	// beta(k+1) = norm(p);
	cublasDnrm2(cublasHandle, N, d_p, 1, &beta[k+1]);
      }
      cudaThreadSynchronize();
      printf("iteration = %d, beta = %e\n", k, beta[k+1]);
      k++;
    }
    // Copy back results on the device (alpha and beta are already in host memory)
    cudaMemcpy(p, d_p, M*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(U, d_U, M*K*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(V, d_V, N*K*sizeof(double), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();

    gettimeofday(&t2, 0);
    double time = (1000000.0*(t2.tv_sec-t1.tv_sec) + t2.tv_usec-t1.tv_usec)/1000000.0;
    printf("Time:  %3.1f microseconds (1e-6 s) \n", time);

    /*
    cudaMalloc((void **)&d_errormat, M*K*sizeof(double));
    cudaMemset(d_errormat, 0, M*K*sizeof(double));
    // errormat = A*V(:,1:k); (sparse time dense)
    cusparseDcsrmm(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, M, K, N, nz, &one, descr, d_val, d_row, d_col, &d_V[0], M, &zero, d_errormat, M);
    // U(:,1:k)'*(A*V(:,1:k))
    double* d_errormat2;
    cudaMalloc((void **)&d_errormat2, K*K*sizeof(double));
    cudaMemset(d_errormat2, 0, K*K*sizeof(double));
    cublasDgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, K, K, M, &one, &d_U[0], M, d_errormat, M, &zero, d_errormat2, K);
    double* errormat = (double *)malloc(M*K*sizeof(double));
    cudaMemcpy(errormat, d_errormat2, K*K*sizeof(double), cudaMemcpyDeviceToHost);

    // Check error assuming that U(:,1:k)'*(A*V(:,1:k)) = spdiags([beta, alpha], [-1,0])
    double err = 0;
    for (int i = 0; i < K; i++) {
      double trueval = alpha[i];
      double approx = errormat[i+i*K];
      err += (trueval - approx) * (trueval - approx);
      if (i < K-1) {
	trueval = beta[i+1];
	approx = errormat[(i+1)+i*K];
	err += (trueval - approx) * (trueval - approx);
      }
    }
    printf("Test Summary:  Error amount = %f\n", err);
    free(errormat);
    cudaFree(d_errormat);
    cudaFree(d_errormat2);
    */
    
    cusparseDestroy(cusparseHandle);
    cublasDestroy(cublasHandle);
    
    free(I);
    free(J);
    free(val);
    cudaFree(d_col);
    cudaFree(d_row);
    cudaFree(d_val);
    cudaFree(d_p);
    cudaFree(d_r);
    cudaFree(d_U);
    cudaFree(d_V);

    cudaDeviceReset();

    exit(0);
}
