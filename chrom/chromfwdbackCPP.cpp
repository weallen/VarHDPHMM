#include <Eigen/Dense>

#include "mex.h"
#include <math.h>

#include "chromhmm.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{

  int T, K, L, N, tmp, t, i, j, k, l;
  double *Pi, *B, *A, *data;
  double *Xi, *GammaK, *GammaInit;

  const mwSize* emit_dims;
  double *outputToolPtr;	
  int wb_ndim = 3;
  int *wb_dims = (int*) mxMalloc(wb_ndim * sizeof(int));

  A = mxGetPr(prhs[0]);
  B = mxGetPr(prhs[1]);	
  Pi = mxGetPr(prhs[2]);
  data = mxGetPr(prhs[3]);

  emit_dims = mxGetDimensions(prhs[1]);
  K = emit_dims[0];
  L = emit_dims[1];
  T = mxGetM(prhs[3]);

  tmp = mxGetN(prhs[0]);
  if (tmp != K) 
    mexErrMsgTxt("The transition matrix must be of size KxK");
  tmp = mxGetM(prhs[0]);
  if (tmp != K)
    mexErrMsgTxt("The transition matrix must be of size KxK");	
  tmp = mxGetN(prhs[2]);
  if (tmp != K) 
    mexErrMsgTxt("The initial state distribution must be of size 1xK");

  Xi = (double*) mxMalloc(K * K * sizeof(double));
  GammaK = (double*) mxMalloc(K * L * 2 * sizeof(double));
  GammaInit = (double*) mxMalloc(1 * K * sizeof(double));

  // Actual algorithm

  double loglik;
  MatrixType alpha;
  MatrixType beta;
  MatrixType gamma;
  VectorType init = VectorType::Zero(K);
  MatrixType trans = MatrixType::Zero(K, K);
  VectorType start_counts = VectorType::Zero(K);
  MatrixType trans_counts = MatrixType::Zero(K, K);
  VectorType weights = VectorType::Zero(K);
  MatrixType softev = MatrixType::Zero(K, T);
  MatrixType bi;
  MatrixType xi_summed;


  // Copy data into trans and init
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      trans(i,j) = A[i + j*K];
    }
  }

  for (int i = 0; i < K; ++i) {
    init(i) = Pi[i];
  }


  // Update soft evidence
  for (t = 0; t < T; ++t) {
    for (k = 0; k < K; ++k) {
      double prod = 1.0;
      for (l = 0; l < L; ++l) {
        if (data[t + l*T] == 1) {
	  prod *= B[k + l*K + 0*K*L];// + 1E-9;	
        } else {
          prod *= B[k + l*K + 1*K*L];// + 1E-9;
        }
      }
      softev(k,t) = prod;
    }
  }

  // Run forward-backward and two-slice sum algorithm
  loglik = FwdBack(trans, init, softev, alpha, beta, gamma);
  trans_counts = TwoSliceSum(trans, softev, alpha, beta);
  init = gamma.col(1);

  // Fill arrays for output
  // Fill GammaInit
  for (int i = 0; i < K; ++i) {
    GammaInit[i] = init(i);
  }

  // Fill Xi
  for (int i = 0; i < K; ++i) {
    for (int j = 0; j < K; ++j) {
      Xi[i + j*K] = trans_counts(j,i);
    }
  }

  for (int k = 0; k < K; ++k) {
    for (int l = 0; l < L; ++l) {
      GammaK[k + l*K + 0*K*L] = 0.0;
      GammaK[k + l*K + 1*K*L] = 0.0;
    }
  }

  // Fill GammaK
  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < K; ++k) {
      for (int l = 0; l < L; ++l) {      
        if (data[t + l*T] == 1) {
          GammaK[k + l*K + 0*K*L] += gamma(k,t);
        } else {
          GammaK[k + l*K + 1*K*L] += gamma(k,t);
        }
      }
    }
  }

  // Copy output to MATLAB
  plhs[3] = mxCreateDoubleMatrix(1, 1, mxREAL);
  outputToolPtr = mxGetPr(plhs[3]);
  outputToolPtr[0] = loglik;	

  plhs[0] = mxCreateDoubleMatrix(K, K, mxREAL);
  outputToolPtr = mxGetPr(plhs[0]);
  memcpy(outputToolPtr, Xi, K*K*sizeof(double));
  
  plhs[2] = mxCreateDoubleMatrix(1, K, mxREAL);
  outputToolPtr = mxGetPr(plhs[2]);
  memcpy(outputToolPtr, GammaInit, 1*K*sizeof(double));

  wb_dims[0] = K;
  wb_dims[1] = L;
  wb_dims[2] = 2;
  plhs[1] = mxCreateNumericArray(wb_ndim, wb_dims, mxDOUBLE_CLASS, mxREAL);
  outputToolPtr = mxGetPr(plhs[1]);
  memcpy(outputToolPtr, GammaK, K*L*2*sizeof(double));

  mxFree(wb_dims);
  mxFree(Xi);
  mxFree(GammaInit);
  mxFree(GammaK);
}
