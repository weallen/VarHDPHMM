#include <Eigen/Dense>
#include "mex.h"
#include <math.h>

#include "chromhmm.h"

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[])
{
  int T, K, L, N, tmp, t, i, j, k, l;
  double *Pi, *B, *A, *data;
	double *outputToolPtr;
	const mwSize* emit_dims;

	if (nrhs != 4) {
		mexErrMsgTxt("Requires 4 arguments: tans, emit, init, data");
		return;
	}

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

  VectorType init = VectorType::Zero(K);
  MatrixType trans = MatrixType::Zero(K, K);
  MatrixType softev = MatrixType::Zero(K, T);

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
	
	double loglik;	
	loglik = LogProb(trans, init, softev);

	plhs[0] = mxCreateDoubleMatrix(1,1,mxREAL);
	outputToolPtr = mxGetPr(plhs[0]);
	outputToolPtr[0] = loglik;	
}
