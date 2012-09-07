#include "chromhmm.h"

void 
ViterbiDecode(const MatrixType& trans, const VectorType& init, const MatrixType& softev, StateVectorType& path)
{
	int K = (int) trans.cols();
	int T = (int) softev.cols();
  MatrixType delta = MatrixType::Zero(K, T);

    // psi stores the indices of the maximum value for each
  StateMatrixType psi = StateMatrixType::Zero(K, T);

    // Initialize last element as soft evidence weighted
    // By prior probabilities
  delta.col(0) = init * softev.col(0);
  delta.col(0) /= delta.col(0).sum();
  VectorType v;
  MatrixType::Index idx;
  for (int t=1; t < T; ++t) {
       for (int j=0; j < K; ++j) {
          // find the unnormalized values
          v = delta.col(t-1) * trans.col(j);
          delta(j,t) = v.maxCoeff(&idx) * softev(j,t);
          psi(j,t) = (int) idx;
       }
       // Normalize
       delta.col(t) /= delta.col(t).sum();
   }

    // Traceback

  delta.col(T - 1).maxCoeff(&idx);
  path(T-1) = (int)idx;
  for (int t = T - 2; t >= 0; --t) {
    path(t) = psi(path(t+1), t+1);
  }
}

double
LogProb(const MatrixType& trans, const VectorType& init, const MatrixType& softev)
{
	int K = (int) trans.cols();
	int T = (int) softev.cols();
	MatrixType alpha = MatrixType::Zero(K,T);
	VectorType scale = VectorType::Zero(T);
	double loglik;
	FilterFwd(trans, softev, init, loglik, alpha, scale);
	return loglik;
}


MatrixType 
TwoSliceSum(const MatrixType& transmat, const MatrixType& softev,
                 const MatrixType& alpha, const MatrixType& beta)
{
  int K = (int) transmat.cols();
  MatrixType xi_summed = MatrixType::Zero(K, K);
  int T = (int) softev.cols();
  VectorType b;
  MatrixType xit;
  
  for (int t = T-2; t >= 0; --t) {
    // multiply by soft evidence
    b = beta.col(t+1) * softev.col(t+1);
    
    xi_summed += transmat * (alpha.col(t).matrix() * b.matrix().transpose()).array();    
    //xi_summed += xit / xit.sum();
  }
  return xi_summed;
}


double 
FwdBack(const MatrixType& transmat, const VectorType& init, 
             const MatrixType& softev, MatrixType& alpha, MatrixType& beta, 
             MatrixType& gamma)
{ 
  int K = (int) transmat.cols();
  int T = (int) softev.cols();
  if (gamma.cols() != T && gamma.rows() != K) {
    gamma.resize(K, T);
  }

	VectorType scale = VectorType::Zero(T);
  double loglik;
  
  FilterFwd(transmat, softev, init, loglik, alpha, scale);

  SmoothBack(transmat, softev, scale, beta);
  int i = 0; int j = 0;
  beta.minCoeff(&i, &j);
  gamma = alpha * beta;

  // Normalize
  for (int t = 0; t < T; ++t) {
    gamma.col(t) /= gamma.col(t).sum();
  }
  
  return loglik;
}

void 
FilterFwd(const MatrixType& transmat, const MatrixType& softev, 
               const VectorType& init, double& loglik, MatrixType& alpha,
							 VectorType& scale)
{
  int T = (int) softev.cols();
  int K = (int) softev.rows();

  if (alpha.cols() != T && alpha.rows() != K) {
    alpha.resize(K, T);
  }
  scale = VectorType::Zero(T);
  Eigen::MatrixXd at = transmat.matrix().transpose();
  
  alpha.col(0) = init * softev.col(0);
  scale(0) = alpha.col(0).sum();
  alpha.col(0) /= scale(0);  

  for (int t = 1; t < T; ++t) {
    alpha.col(t) = (at.matrix() * alpha.col(t-1).matrix()).array();
    alpha.col(t) *= softev.col(t);
    scale(t) = alpha.col(t).sum();
    alpha.col(t) /= scale(t);
  }
  loglik = scale.log().sum();
}

void 
SmoothBack(const MatrixType& transmat, const MatrixType& softev, const VectorType& scale,
                MatrixType& beta)
{
  int K = (int)softev.rows();
  int T = (int)softev.cols();
  if (beta.cols() != T && beta.rows() != K) 
    beta.resize(K, T);  
  for (int k = 0; k < K; ++k) 
    beta(k, T-1) = 1.0;
  for (int t = T-2; t >= 0; --t) {    
    // beta(:,t) = trans_ * (beta(:,t+1) .* soft_evidence_(:,t+1))
    beta.col(t) = transmat.matrix() * (beta.col(t+1) * softev.col(t+1)).matrix();    
    // normalize
//    beta.col(t) /= beta.col(t).sum();
			beta.col(t) /= scale(t);
  }
}


