/**
Copyright (c) 2016 Theodore Gast, Chuyuan Fu, Chenfanfu Jiang, Joseph Teran

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

If the code is used in an article, the following paper shall be cited:
@techreport{qrsvd:2016,
  title={Implicit-shifted Symmetric QR Singular Value Decomposition of 3x3 Matrices},
  author={Gast, Theodore and Fu, Chuyuan and Jiang, Chenfanfu and Teran, Joseph},
  year={2016},
  institution={University of California Los Angeles}
}

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

################################################################################
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
SVD based on implicit QR with Wilkinson Shift
*/
#ifndef JIXIE_IMPLICIT_QR_SVD_H
#define JIXIE_IMPLICIT_QR_SVD_H

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
    \brief zero chasing the 3X3 matrix to bidiagonal form
    original form of H:   x x 0
    x x x
    0 0 x
    after zero chase:
    x x 0
    0 x x
    0 0 x
    */
template <class T>
inline void zeroChase(Eigen::Matrix<T, 3, 3>& H, Eigen::Matrix<T, 3, 3>& U, Eigen::Matrix<T, 3, 3>& V)
{

    /**
        Reduce H to of form
        x x +
        0 x x
        0 0 x
        */
    GivensRotation<T> r1(H(0, 0), H(1, 0), 0, 1);
    /**
        Reduce H to of form
        x x 0
        0 x x
        0 + x
        Can calculate r2 without multiplying by r1 since both entries are in first two
        rows thus no need to divide by sqrt(a^2+b^2)
        */
    GivensRotation<T> r2(1, 2);
    if (H(1, 0) != 0)
        r2.compute(H(0, 0) * H(0, 1) + H(1, 0) * H(1, 1), H(0, 0) * H(0, 2) + H(1, 0) * H(1, 2));
    else
        r2.compute(H(0, 1), H(0, 2));

    r1.rowRotation(H);

    /* GivensRotation<T> r2(H(0, 1), H(0, 2), 1, 2); */
    r2.columnRotation(H);
    r2.columnRotation(V);

    /**
        Reduce H to of form
        x x 0
        0 x x
        0 0 x
        */
    GivensRotation<T> r3(H(1, 1), H(2, 1), 1, 2);
    r3.rowRotation(H);

    // Save this till end for better cache coherency
    // r1.rowRotation(u_transpose);
    // r3.rowRotation(u_transpose);
    r1.columnRotation(U);
    r3.columnRotation(U);
}

/**
     \brief make a 3X3 matrix to upper bidiagonal form
     original form of H:   x x x
                           x x x
                           x x x
     after zero chase:
                           x x 0
                           0 x x
                           0 0 x
  */
template <class T>
inline void makeUpperBidiag(Eigen::Matrix<T, 3, 3>& H, Eigen::Matrix<T, 3, 3>& U, Eigen::Matrix<T, 3, 3>& V)
{
    U = Eigen::Matrix<T, 3, 3>::Identity();
    V = Eigen::Matrix<T, 3, 3>::Identity();

    /**
      Reduce H to of form
                          x x x
                          x x x
                          0 x x
    */

    GivensRotation<T> r(H(1, 0), H(2, 0), 1, 2);
    r.rowRotation(H);
    // r.rowRotation(u_transpose);
    r.columnRotation(U);
    // zeroChase(H, u_transpose, V);
    zeroChase(H, U, V);
}

/**
     \brief make a 3X3 matrix to lambda shape
     original form of H:   x x x
     *                     x x x
     *                     x x x
     after :
     *                     x 0 0
     *                     x x 0
     *                     x 0 x
  */
template <class T>
inline void makeLambdaShape(Eigen::Matrix<T, 3, 3>& H, Eigen::Matrix<T, 3, 3>& U, Eigen::Matrix<T, 3, 3>& V)
{
    U = Eigen::Matrix<T, 3, 3>::Identity();
    V = Eigen::Matrix<T, 3, 3>::Identity();

    /**
      Reduce H to of form
      *                    x x 0
      *                    x x x
      *                    x x x
      */

    GivensRotation<T> r1(H(0, 1), H(0, 2), 1, 2);
    r1.columnRotation(H);
    r1.columnRotation(V);

    /**
      Reduce H to of form
      *                    x x 0
      *                    x x 0
      *                    x x x
      */

    r1.computeUnconventional(H(1, 2), H(2, 2));
    r1.rowRotation(H);
    r1.columnRotation(U);

    /**
      Reduce H to of form
      *                    x x 0
      *                    x x 0
      *                    x 0 x
      */

    GivensRotation<T> r2(H(2, 0), H(2, 1), 0, 1);
    r2.columnRotation(H);
    r2.columnRotation(V);

    /**
      Reduce H to of form
      *                    x 0 0
      *                    x x 0
      *                    x 0 x
      */
    r2.computeUnconventional(H(0, 1), H(1, 1));
    r2.rowRotation(H);
    r2.columnRotation(U);
}

/**
   \brief 2x2 polar decomposition.
   \param[in] A matrix.
   \param[out] R Robustly a rotation matrix in givens form
   \param[out] S_Sym Symmetric. Whole matrix is stored

   Whole matrix S is stored since its faster to calculate due to simd vectorization
   Polar guarantees negative sign is on the small magnitude singular value.
   S is guaranteed to be the closest one to identity.
   R is guaranteed to be the closest rotation to A.
*/
template <class TA, class T, class TS>
inline std::enable_if_t<isSize<TA>(2, 2) && isSize<TS>(2, 2)>
polarDecomposition(const Eigen::MatrixBase<TA>& A,
    GivensRotation<T>& R,
    const Eigen::MatrixBase<TS>& S_Sym)
{
    Eigen::Matrix<T, 2, 1> x(A(0, 0) + A(1, 1), A(1, 0) - A(0, 1));
    T denominator = x.norm();
    R.c = (T)1;
    R.s = (T)0;
    if (denominator != 0) {
        /*
          No need to use a tolerance here because x(0) and x(1) always have
          smaller magnitude then denominator, therefore overflow never happens.
        */
        R.c = x(0) / denominator;
        R.s = -x(1) / denominator;
    }
    Eigen::MatrixBase<TS>& S = const_cast<Eigen::MatrixBase<TS>&>(S_Sym);
    S = A;
    R.rowRotation(S);
}

/**
   \brief 2x2 polar decomposition.
   \param[in] A matrix.
   \param[out] R Robustly a rotation matrix.
   \param[out] S_Sym Symmetric. Whole matrix is stored

   Whole matrix S is stored since its faster to calculate due to simd vectorization
   Polar guarantees negative sign is on the small magnitude singular value.
   S is guaranteed to be the closest one to identity.
   R is guaranteed to be the closest rotation to A.
*/
template <class TA, class TR, class TS>
inline std::enable_if_t<isSize<TA>(2, 2) && isSize<TR>(2, 2) && isSize<TS>(2, 2)>
polarDecomposition(const Eigen::MatrixBase<TA>& A,
    const Eigen::MatrixBase<TR>& R,
    const Eigen::MatrixBase<TS>& S_Sym)
{
    using T = ScalarType<TA>;
    GivensRotation<T> r(0, 1);
    polarDecomposition(A, r, S_Sym);
    r.fill(R);
}

/**
   \brief 2x2 Jacobi rotation S = VDV'
   \param[in] S Input symmetric matrix.
   \param[out] V Rotation matrix
   \param[out] D Vector of eigenvalues of S
 */
/**
template <class T, class TD>
inline std::enable_if_t<isSize<T> (2, 2) && isSize<TD>(2, 1)> 
JacobiRotation(const Eigen::MatrixBase<T>& S, 
		const Eigen::MatrixBase<T>& V, 
		const Eigen::MatrixBase<TD>& D)
{
	using std::sqrt;

	T t, c, s, tau;	// tangent, cosine, sine
	if (S(1,0) != 0)
	{
		tau = (S(1,1) - S(0,0))/(2 * S(1,0));
		if (tau > 0)
			t = 1 / (tau + sqrt(1 + pow(tau,2)));
		
		else
			t = 1 / (tau - sqrt(1 + pow(tau,2)));
		c = 1 / sqrt(1 + pow(t,2));
		s = t * c;
	}
	else
	{
		c = 0;
		s = 1;
	}
	V(0,0) = c;
	V(1,0) = s;
	V(0,1) = -s;
	V(1,1) = c;
	D(0) = (c * S(0,0) - s * S(1,0)) * c - (c * S(1,0) - s * S(1,1)) * s;
	D(1) = (s * S(1,1) + c * S(1,0)) * s + (s * S(1,0) + c * S(1,1)) * c;
}	
*/


/**
   \brief 2x2 SVD (singular value decomposition) A=USV'
   \param[in] A Input matrix.
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

    Eigen::Matrix<T, 2, 2> C, Vhat, Utilde, Vtilde;
    Eigen::Matrix<T, 2, 1> Sigma_hat, Sigma_tilde;
    bool det_V_neg = false, det_U_neg = false;	// flag of the negativity of determinant of V, U.
    
    /**
     * ALGORITHM
     */

    // Step 1
    C = F * F.transpose();

    // Step 2, Jacobi Rotation
    Eigen::Matrix<T, 2, 1> Sigma_s;	// square of eigenvalues
    // JacobiRotation(C, Vhat, Sigma_s);	// do Jacobi rotation
    T t, c, s, tau;	// tangent, cosine, sine
    T C11 = C(0, 0); 
    T C21 = C(1, 0);
    T C22 = C(1, 1);

    if (C11 != 0)
    {
		tau = (C22 - C11)/(2 * C21);
		if (tau > 0)
			t = 1 / (tau + sqrt(1 + pow(tau,2)));
		
		else
			t = 1 / (tau - sqrt(1 + pow(tau,2)));
		c = 1 / sqrt(1 + pow(t,2));
		s = t * c;
     } 
    else
    {
		c = 0;
		s = 1;
    }
    Vhat(0,0) = c;
    Vhat(1,0) = s;
    Vhat(0,1) = -s;
    Vhat(1,1) = c;
    Sigma_s(0) = (c * C11- s * C21) * c - (c * C21 - s * C22) * s;
    Sigma_s(1) = (s * C22 + c * C21) * s + (s * C21 + c * C22) * c;

    // Step 3
    Sigma_hat(0) = sqrt(Sigma_s(0)), Sigma_hat(1) = sqrt(Sigma_s(1));
    
    // Step 4, sorting the eigenvalues
    if (Sigma_hat(0) < Sigma_hat(1)){
	Sigma_tilde(0) = Sigma_hat(1);
	Sigma_tilde(1) = Sigma_hat(0);
	Vtilde(0,0) = Vhat(1,0);
	Vtilde(0,1) = Vhat(1,1);
	Vtilde(1,0) = Vhat(0,0);
	Vtilde(1,1) = Vhat(0,1);
	det_V_neg = true;	// change the flag
    }

    else{
	Sigma_tilde(0) = Sigma_hat(0);
	Sigma_tilde(1) = Sigma_hat(1);
	Vtilde(0,0) = Vhat(1,0);
	Vtilde(0,1) = Vhat(1,1);
	Vtilde(1,0) = Vhat(0,0);
	Vtilde(1,1) = Vhat(0,1);
    } 

    // Step 5, A = FV
    Eigen::Matrix<T, 2, 2> A;
    A = F * V; 
    
    // Step 6, QR Decomposition using Givens Rotation
    GivensRotation<T> r(A(0,0), A(1,0), 0, 1);
    Eigen::Matrix<T, 2, 2> Q_tran;
    r.fill(Q_tran);
    Utilde = Q_tran.transpose();
    
    // Step 7,
    T r22 = Q_tran(1,0) * A(0,1) + Q_tran(1,1) * A(1,1);
    if (r22 < 0){
	    Utilde(1,0) = -Utilde(1,0);
	    Utilde(1,1) = -Utilde(1,1);
	    det_U_neg = true;	// change the flag
    }

    // Step 8, sign rearrangement
    U = Utilde;
    V = Vtilde;
    T det_F = F(0,0) * F(1,1) - F(1,0) * F(0,1);
    if (det_F < 0){
	    Sigma(0) = Sigma_tilde(0);
	    Sigma(1) = - Sigma_tilde(1);
	    if (det_U_neg)
		    U.col(1) = -U.col(1);
	    else
		    V.col(1) = -V.col(1);
    }
    else if (det_F > 0){
	    Sigma(0) = Sigma_tilde(0);
	    Sigma(1) = Sigma_tilde(1);

	    if (det_U_neg && det_V_neg){
		    U.col(1) = - U.col(1);
		    V.col(1) = - V.col(1);
	    }
    }
    else if (det_F == 0){
	    Sigma(0) = Sigma_tilde(0);
	    Sigma(1) = Sigma_tilde(1);
	    if (det_U_neg && det_V_neg){
		    U.col(0) = - U.col(0);
		    V.col(0) = - V.col(0);
	    
	    }
    }
    
		    
}

}
#endif
