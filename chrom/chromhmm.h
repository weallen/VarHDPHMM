#include <Eigen/Dense>

typedef Eigen::ArrayXXd MatrixType;
typedef Eigen::ArrayXi StateVectorType;
typedef Eigen::ArrayXd VectorType;
typedef Eigen::ArrayXXi StateMatrixType;


MatrixType 
TwoSliceSum(const MatrixType&, const MatrixType&,
	    const MatrixType&, const MatrixType&);
double 
FwdBack(const MatrixType&, const VectorType& , 
	const MatrixType&, MatrixType&, MatrixType&, 
	MatrixType&);

void 
FilterFwd(const MatrixType&, const MatrixType&, 
	  const VectorType&, double&, MatrixType&,
		VectorType&);

void 
SmoothBack(const MatrixType&, const MatrixType&, 
					const VectorType&, MatrixType&);

double 
LogProb(const MatrixType&, const VectorType&, const MatrixType&);

void
ViterbiDecode(const MatrixType& trans, const VectorType& init, const MatrixType& softev, StateVectorType& path);



