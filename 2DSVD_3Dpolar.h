/**
###############################################################################
This file implements 2D and 3D polar decompositions and SVDs.

T may be float or double.

2D SVD:
    Eigen::Matrix<T, 2, 2> A;
    A<<1,2,3,4;
    Eigen::Matrix<T, 2, 1> S;
    Eigen::Matrix<T, 2, 2> U;
    Eigen::Matrix<T, 2, 2> V;
    JIXIE::singularValueDecomposition(A,U,S,V);
    // A = U S V'
    // U and V will be rotations
    // S will be singular values sorted by decreasing magnitude. Only the last one may be negative.

################################################################################
*/

/**
*/
#ifndef JIXIE_IMPLICIT_POLAR_SVD_HW1_H
#define JIXIE_IMPLICIT_POLAR_SVD_HW1_H

#include "Tools.h"
#include <iostream>

namespace JIXIE {

/**
    Class for givens rotation.
    Row rotation G*A corresponds to something like
    c -s  0
    ( s  c  0 ) A
    0  0  1
    Column rotation A G' corresponds to something like
    c -s  0
    A ( s  c  0 )
    0  0  1

    c and s are always computed so that
    ( c -s ) ( a )  =  ( * )
    s  c     b       ( 0 )

    Assume rowi<rowk.
    */
template <class T>
class GivensRotation {
public:
    int rowi;
    int rowk;
    T c;
    T s;

    inline GivensRotation(int rowi_in, int rowk_in)
        : rowi(rowi_in)
        , rowk(rowk_in)
        , c(1)
        , s(0)
    {
    }

    inline GivensRotation(T a, T b, int rowi_in, int rowk_in)
        : rowi(rowi_in)
        , rowk(rowk_in)
    {
        compute(a, b);
    }

    ~GivensRotation() {}

    inline void transposeInPlace()
    {
        s = -s;
    }

    /**
        Compute c and s from a and b so that
        ( c -s ) ( a )  =  ( * )
        s  c     b       ( 0 )
        */
    inline void compute(const T a, const T b)
    {
        using std::sqrt;

        T d = a * a + b * b;
        c = 1;
        s = 0;
        if (d != 0) {
            // T t = 1 / sqrt(d);
            T t = JIXIE::MATH_TOOLS::rsqrt(d);
            c = a * t;
            s = -b * t;
        }
    }

    /**
        This function computes c and s so that
        ( c -s ) ( a )  =  ( 0 )
        s  c     b       ( * )
        */
    inline void computeUnconventional(const T a, const T b)
    {
        using std::sqrt;

        T d = a * a + b * b;
        c = 0;
        s = 1;
        if (d != 0) {
            // T t = 1 / sqrt(d);
            T t = JIXIE::MATH_TOOLS::rsqrt(d);
            s = a * t;
            c = b * t;
        }
    }
    /**
      Fill the R with the entries of this rotation
        */
    template <class MatrixType>
    inline void fill(const MatrixType& R) const
    {
        MatrixType& A = const_cast<MatrixType&>(R);
        A = MatrixType::Identity();
        A(rowi, rowi) = c;
        A(rowk, rowi) = -s;
        A(rowi, rowk) = s;
        A(rowk, rowk) = c;
    }

    /**
        This function does something like
        c -s  0
        ( s  c  0 ) A -> A
        0  0  1
        It only affects row i and row k of A.
        */
    template <class MatrixType>
    inline void rowRotation(MatrixType& A) const
    {
        for (int j = 0; j < MatrixType::ColsAtCompileTime; j++) {
            T tau1 = A(rowi, j);
            T tau2 = A(rowk, j);
            A(rowi, j) = c * tau1 - s * tau2;
            A(rowk, j) = s * tau1 + c * tau2;
        }
    }

    /**
        This function does something like
        c  s  0
        A ( -s  c  0 )  -> A
        0  0  1
        It only affects column i and column k of A.
        */
    template <class MatrixType>
    inline void columnRotation(MatrixType& A) const
    {
        for (int j = 0; j < MatrixType::RowsAtCompileTime; j++) {
            T tau1 = A(j, rowi);
            T tau2 = A(j, rowk);
            A(j, rowi) = c * tau1 - s * tau2;
            A(j, rowk) = s * tau1 + c * tau2;
        }
    }

    /**
      Multiply givens must be for same row and column
      **/
    inline void operator*=(const GivensRotation<T>& A)
    {
        T new_c = c * A.c - s * A.s;
        T new_s = s * A.c + c * A.s;
        c = new_c;
        s = new_s;
    }

    /**
      Multiply givens must be for same row and column
      **/
    inline GivensRotation<T> operator*(const GivensRotation<T>& A) const
    {
        GivensRotation<T> r(*this);
        r *= A;
        return r;
    }
};

/**
   \brief 2x2 Jacobi Rotaion C = VDV'
   \param[in] C Symmetric matrix
   \param[out] Sigma Vector containing all the eigenvalues
   \param[out] V Orthogonal matrix
*/

template <class T>
inline void
JacobiRotation(
   const Eigen::Matrix<T, 2, 2>& C,
   Eigen::Matrix<T, 2, 2>& V,
   Eigen::Matrix<T, 2, 1>& Sigma)
{

    T t, c, s, tau;	// tangent, cosine, sine
    T C11 = C(0, 0); 
    T C21 = C(1, 0);
    T C22 = C(1, 1);

    if (std::abs(C21) == 0) 
    {
		V = Eigen::Matrix<T, 2, 2>::Identity();
	    	Sigma(0) = C(0, 0);
		Sigma(1) = C(1, 1);
    }

    else
    {
		tau = (C22 - C11)/(2. * C21);
		if (tau > 0)
			t = - C21 / (sqrt(pow(C21, 2) + pow((C22-C11)/2., 2)) + (C22-C11)/2.);
		
		else
			t = - C21 / ((C22-C11)/2. - sqrt(pow(C21, 2) + pow((C22-C11)/2., 2)));
		c = JIXIE::MATH_TOOLS::rsqrt(1. + pow(t,2));
		s = t * c;
    		V << c, -s,
		     s, c;
     		// Eigen::Matrix<T, 2, 2> D = V.transpose() * C * V;
     		// Sigma(0) = std::max(T(D(0, 0)), T(0));
     		// Sigma(1) = std::max(T(D(1, 1)), T(0));
     		 Sigma(0) = std::max(T(pow(c, 2) * C11 + 2. * c * s * C21 + pow(s, 2) * C22), T(0));
     		 Sigma(1) = std::max(T(pow(s, 2) * C11 - 2. * c * s * C21 + pow(c, 2) * C22), T(0));
     } 
}

/**
   \brief 2x2 SVD (singular value decomposition) A=USV'
   \param[in] F Input matrix.
   \param[out] U Robustly a rotation matrix.
   \param[out] Sigma Vector of singular values sorted with decreasing magnitude. The second one can be negative.
   \param[out] V Robustly a rotation matrix.
*/
template <class T>
inline void 
singularValueDecomposition(
    const Eigen::Matrix<T, 2, 2>& F,
    Eigen::Matrix<T, 2, 2>& U,
    Eigen::Matrix<T, 2, 1>& Sigma,
    Eigen::Matrix<T, 2, 2>& V,
    ScalarType<T> tol = 64 * std::numeric_limits<ScalarType<T> >::epsilon())
{
    using std::sqrt;

    bool det_V_neg = false, det_U_neg = false;	// flag of the negativity of determinant of V, U.
    
    /**
     * ALGORITHM
     */

    // Step 1
    Eigen::Matrix<T, 2, 2> C;
    C.noalias() = F.transpose() * F;

    // Step 2, Jacobi Rotation, the sigma here is the square of singular values
    JacobiRotation(C, V, Sigma);

    // Step 3
    Sigma(0) = sqrt(Sigma(0));
    Sigma(1) = sqrt(Sigma(1));
    
    // Step 4, sorting the eigenvalues
    if (Sigma(0) < Sigma(1)){
	Sigma.row(0).swap(Sigma.row(1));
	V.col(0).swap(V.col(1));
	det_V_neg = true;	// change the flag
    }

    // Step 5, A = F * Vtilde
    Eigen::Matrix<T, 2, 2> A;
    A.noalias() = F * V; 
    
    // Step 6, QR Decomposition using Givens Rotation
    GivensRotation<T> r(A(0, 0), A(1, 0), 0, 1);
    r.fill(U);
    
    // Step 7, fill in U = [q1, \pm q2]
    r.rowRotation(A);
    if (A(1,1) < 0){
	    U.col(1) = - U.col(1);
	    det_U_neg = true;	// change the flag
    }

    // Step 8, sign rearrangement
    T det_F = F(0,0) * F(1,1) - F(1,0) * F(0,1);
    if (det_F < 0){
	    Sigma(1) = -Sigma(1);
	    if (det_U_neg)
	    	{U.col(1) = -U.col(1);}
	    else
	    	{V.col(1) = -V.col(1);}
    }
    else if (det_F > 0){

	    if (det_U_neg && det_V_neg){
		    U.col(1) = - U.col(1);
		    V.col(1) = - V.col(1);
	    }
    }
    else if (det_F == 0){
	    if (det_V_neg){
		    V.col(0) = -V.col(0);
		    if (!det_U_neg)
		    {
		    	    Sigma(0) = -Sigma(0);
		    }
	    }
	    if (det_U_neg){
		    U.col(0) = - U.col(0);
	   	    if (!det_V_neg)
		    {
			    Sigma(0) = -Sigma(0);
		    }
	    }
    }

    //if(std::isnan((V*V.transpose() - Eigen::Matrix<T,2,2>::Identity()).array().abs().maxCoeff())) std::cout << tau << "   " << t << "   " << C22-C11 << "   " << C21 << "   "<<  (sqrt(pow(1e-10,2) + pow(-5./2,2)) + -5./2) << " NAN\n";
    // std::cout << Sigma(0) << ", " << Sigma(1) << "\n";
		    
}


/**
   \brief 3x3 polar decomposition.
   \param[in] F Input matrix.
   \param[out] R Robustly a rotation matrix.
   \param[out] S Robustly a SPD matrix.
*/
template <class T>
inline void 
PolarDecomposition(
    const Eigen::Matrix<T, 3, 3>& F,
    Eigen::Matrix<T, 3, 3>& R,
    Eigen::Matrix<T, 3, 3>& S,
    ScalarType<T> tol = 64 * std::numeric_limits<ScalarType<T> >::epsilon())
{
	using std::fabs;
	using std::max;
        R = Eigen::Matrix<T, 3, 3>::Identity();
 	S = F;
	int it = 0;
	int max_it = 1000;
	while ((it < max_it) && max({(fabs(S(0, 1) - S(1, 0))), (fabs(S(0, 2) - S(2, 0))), (fabs(S(1, 2) - S(2, 1)))}) > tol){
		for (int j = 1; j < 3; ++j){
			for (int i = 0; i < j; ++i){
				GivensRotation<T> r(S(i, i) + S(j, j), S(j, i) - S(i, j), i, j);
				r.columnRotation(R);
				r.rowRotation(S);
			}
		}
		++it;
	}
	
}
}
#endif
