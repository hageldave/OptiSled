package hageldave.optisled.generic.numerics;

import java.util.Arrays;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

/**
 * Interface for operations on a generic matrix type {@code M}.
 * This allows for the implementation of numeric algorithms
 * (e.g. gradient descent) that are agnostic to the choice 
 * of a linear algebra library, and makes it possible to 
 * integrate this API into environments where the a matrix
 * type is already present.
 * 
 * @author hageldave
 * @param <M> the matrix type
 */
public interface MatCalc<M> {
	
	/**
	 * @param v values
	 * @return column vector of given values
	 */
	public M vecOf(double ... v);
	
	/**
	 * @param nRows number of rows
	 * @param values (in row major order)
	 * @return matrix with {@code nRows} and {@code nCols=values.length/nRows} filled with specified values
	 */
	public M matOf(int nRows, double... values);
	
	/**
	 * @param values  of the matrix
	 * @return matrix from the specified 2d array {@code (rowValues[] = values[i])}
	 */
	public M matOf(double[][] values);
	
	/**
	 * @param n number of dimensions
	 * @return zero column vector of dimension {@code n}
	 */
	public M zeros(int n);
	
	/** 
	 * @param rows number of rows
	 * @param columns number of columns
	 * @return matrix of zeros of specified size
	 */
	public M zeros(int rows, int columns);
	
	/**
	 * @param n number of rows (=number of columns)
	 * @return identity matrix of size n x n
	 */
	public default M eye(int n) {
		return eye(n, 1.0);
	}
	
	/**
	 * @param n number of rows (=number of columns)
	 * @param s scaling factor
	 * @return s*identity matrix of size n x n 
	 */
	public M eye(int n, double s);
	
	/**
	 * @param rows number of rows
	 * @param cols number of columns
	 * @param rnd random number generator
	 * @return matrix of random values in range 0.0 .. 1.0
	 */
	public default M rand(int rows, int cols, Random rnd) {
		M m = zeros(rows, cols);
		for(int r=0; r<rows; r++)
			for(int c=0; c<cols; c++)
				set_inp(m, r, c, rnd.nextDouble());
		return m;
	}
	
	/** 
	 * @param rows number of rows
	 * @param cols number of columns
	 * @return matrix of random values in range 0.0 .. 1.0
	 */
	public default M rand(int rows, int cols) {
		return rand(rows, cols, new Random());
	}
	
	/**
	 * @param n
	 * @return column vector of random values in range 0.0 .. 1.0
	 */
	public default M rand(int n) {
		return rand(n, 1);
	}
	
	/**
	 * @param rows number of rows
	 * @param cols number of columns
	 * @param rnd random number generator
	 * @return matrix with normally distributed random values (standard normal, mean=0, var=1)
	 */
	public default M randN(int rows, int cols, Random rnd) {
		M m = zeros(rows, cols);
		for(int r=0; r<rows; r++)
			for(int c=0; c<cols; c++)
				set_inp(m, r, c, rnd.nextGaussian());
		return m;
	}
	
	/**
	 * @param rows number of rows
	 * @param cols number of columns
	 * @return random matrix with normally distributed values (standard normal, mean=0, var=1)
	 */
	public default M randN(int rows, int cols) {
		return randN(rows, cols, new Random());
	}
	
	/**
	 * @param m matrix
	 * @return number of rows
	 */
	public int numRows(M m);
	
	/**
	 * @param m matrix 
	 * @return number of columns */
	public int numCols(M m);
	
	/**
	 * @param m matrix
	 * @return number of elements
	 */
	public default int numElem(M m) {
		return numCols(m)*numRows(m);
	}
	
	/**
	 * @param a vector 
	 * @param b vector
	 * @return inner or dot product for given vectors ‹a,b›
	 */
	public double inner(M a, M b);
	
	/**
	 * just an alias for {@link #inner(M, M)}
	 * @param a vector 
	 * @param b vector
	 * @return inner or dot product for given vectors ‹a,b›
	 */
	public default double dot(M a, M b) {return inner(a,b);}
	
	/**
	 * @param m matrix/vector
	 * @param s scaling factor
	 * @return a scaled version of given matrix/vector
	 */
	public M scale(M m, double s);
	
	/**
	 * @param m matrix/vector to be scaled in-place
	 * @param s scaling
	 * @return the same matrix/vector argument which was scaled in-place 
	 */
	public M scale_inp(M m, double s);
	
	/**
	 * @param a matrix/vector
	 * @param b matrix/vector
	 * @return result of matrix multiplication {@code a * b}
	 */
	public M matmul(M a, M b);
	
	/**
	 * @param a matrix/vector
	 * @param b matrix/vector
	 * @return element-wise multiplication of a and b (need to be of same size)
	 */
	public M elmmul(M a, M b);
	
	/**
	 * @param m matrix
	 * @param colV column vector
	 * @return copy of m where each row i was scaled by entry i of column vector
	 */
	public M mulRowsByColVec(M m, M colV);
	
	/**
	 * @param m matrix
	 * @param rowV row vector
	 * @return copy of m where each column j was scaled by entry j of row vector
	 */
	public M mulColsByRowVec(M m, M rowV);
	
	/**
	 * @param m matrix
	 * @return column vector of row sums of m
	 */
	public M rowSums(M m);
	
	/**
	 * @param m matrix
	 * @return column vector of row means of m
	 */
	public default M rowMeans(M m) {
		return scale_inp(rowSums(m), 1.0/numCols(m));
	}
	
	/**
	 * @param m matrix
	 * @return row vector of column means of m
	 */
	public default M colMeans(M m) {
		return scale_inp(colSums(m), 1.0/numRows(m));
	}
	
	/**
	 * @param m matrix
	 * @return column vector of row minimums
	 */
	public M rowMins(M m);
	
	/**
	 * @param m matrix
	 * @return column vector of row maximums
	 */
	public M rowMaxs(M m);
	
	/**
	 * @param m matrix
	 * @return row vector of column minimums
	 */
	public M colMins(M m);
	
	/**
	 * @param m matrix
	 * @return row vector of column maximums
	 */
	public M colMaxs(M m);
	
	/**
	 * @param m matrix
	 * @return row vector of column sums of m
	 */
	public M colSums(M m);
	
	/**
	 * @param m matrix
	 * @param rowV row vector
	 * @return result of subtracting specified row vector from each row of m
	 */
	public M subRowVec(M m, M rowV);
	
	/**
	 * @param m matrix
	 * @param colV column vector
	 * @return result of subtracting specified column vector from each column of m
	 */
	public M subColVec(M m, M colV);
	
	/**
	 * @param m matrix
	 * @param rowV row vector
	 * @return result of adding specified row vector to each row of m
	 */
	public M addRowVec(M m, M rowV);
	
	/**
	 * @param m matrix
	 * @param colV column vector
	 * @return result of adding specified column vector to each column of m
	 */
	public M addColVec(M m, M colV);
	
	/**
	 * @param m matrix
	 * @return transpose of m
	 */
	public M trp(M m);
	
	/**
	 * @param a matrix/vector
	 * @param b scalar
	 * @return result of adding b to each entry of a
	 */
	public M add(M a, double b);
	
	/**
	 * @param a matrix/vector
	 * @param b matrix/vector
	 * @return a+b (need to be same size)
	 */
	public M add(M a, M b);
	
	/**
	 * @param a matrix/vector
	 * @param b matrix/vector
	 * @return a-b (in-place subtraction from a, need to be same size)
	 */
	public M sub(M a, M b);
	
	/**
	 * @param a matrix/vector
	 * @param b matrix/vector
	 * @return a=a+b (in-place addition to a, need to be same size)
	 */
	public M add_inp(M a, M b);
	
	/**
	 * @param a matrix/vector
	 * @param b matrix/vector
	 * @return a=a-b (in-place subtraction from a, need to be same size)
	 */
	public M sub_inp(M a, M b);
	
	/**
	 * @param a matrix/vector
	 * @param b scalar
	 * @return result of subtracting b from each entry of a
	 */
	public default M sub(M a, double b) {
		return add(a, -b);
	}
	
	/** @return the same matrix/vector argument where the entry at given index was set (row major indexing) */
	public M set_inp(M m, int idx, double v);
	
	/** @return the same matrix/vector argument where the entry at specified position was set */
	public default M set_inp(M m, int row, int col, double v) {
		return set_inp(m, row*numCols(m)+col, v);
	}
	
	/** @return entry of matrix/vector at given index (row major indexing) */
	public double get(M m, int idx);
	
	/** @return entry of matrix/vector at specified position */
	public default double get(M m, int row, int col) {
		return get(m, row*numCols(m)+col);
	}
	
	/** @return a copy of the argument */
	public M copy(M m);
	
	/** @return values of the matrix/vector in row major order */
	public double[] toArray(M m);
	
	/** @return values of the matrix/vector {@code ( values[row][col] )} */
	public double[][] toArray2D(M m);
	
	/** @return same as argument which was normalized in place to unit length (if thresh smaller than norm) */
	public default M normalize_inp(M m, double thresh) {
		double norm = norm(m);
		return norm < thresh ? m : scale_inp(m, 1/norm);
	}
	
	/** @return normalized to unit length version of vector (if thresh smaller than norm) */
	public default M normalize(M m, double thresh) { return normalize_inp(copy(m), thresh); }
	
	/** @return same as argument which was normalized in place to unit length (if 1e-7 smaller than norm) */
	public default M normalize_inp(M m) { return normalize_inp(m, 1e-7); }
	
	/** @return normalized to unit length version of vector (if 1e-7 smaller than norm) */
	public default M normalize(M m) { return normalize_inp(copy(m)); }
	
	/** @return squared vector norm ||v||^2 = ‹v,v› */
	public default double norm2(M v) { return inner(v,v); }
	
	/** @return vector norm ||v|| = sqrt(‹v,v›) */
	public default double norm(M v) { return Math.sqrt(norm2(v)); }
	
	public default double frob2(M m) {
		return sum(elmmul(m, m));
	}
	
	public default double frob(M m) {
		return Math.sqrt(frob2(m));
	}
	
	/** @return singular value decomposition (full or sparse/economic) 
	 * with matrices U,S,V where m=U*S*trp(V)
	 */
	public M[] svd(M m, boolean full);
	
	/** @returns Cholesky decomposition of m, the upper triangular part U, so that m = U' U */
	public M cholesky(M m);
	
	public M[] symEvd(M m);
	
	/** @return determinant of m */
	public double det(M m);
	
	/** @return element wise exp */
	public default M exp_inp(M m) {
		return elemwise_inp(m, Math::exp);
	}
	
	/** @return element wise sqrt */
	public default M sqrt_inp(M m) {
		return elemwise_inp(m, Math::sqrt);
	}
	
	/** @return element wise f(m_ij) */
	public default M elemwise_inp(M m, DoubleUnaryOperator f) {
		for(int i=0; i<numElem(m); i++)
			set_inp(m, i, f.applyAsDouble(get(m, i)));
		return m;
	}
	
	/** @return side by side concatenation (equal number of rows) */
	public M concatHorz(M a, M b);
	
	/** @return stacked concatenation (equal number of columns) */
	public M concatVert(M a, M b);
	
	/** @return rows ra to rb (exclusive) and columns ca to cb (exclusive) */
	public M getRange(M m, int ra, int rb, int ca, int cb);
	
	
	public default M getRow(M m, int r) {
		return getRange(m, r, r+1, 0, numCols(m));
	}
	
	public default M getCol(M m, int c) {
		return getRange(m, 0, numRows(m), c, c+1);
	}
	
	/** @return the diagonal of the matrix as column vector */
	public M diagV(M m);
	
	/** @return the diagonal matrix from the vector */
	public M diagM(M v);
	
	public default void copyValues(M src, M target) {
		copyValues(src, 0, target, 0, Math.min(numElem(src), numElem(target)));
	}
	
	public default void copyValues(M src, int startSrc, M target, int startTarget, int len) {
		for(int i=0; i<len; i++)
			set_inp(target, i+startTarget, get(src, i+startSrc));
	}
	
	public M[] matArray(int n);
	
	public M[][] matArray(int m, int n);
	
	
//	public default M[] matArray(M ... ms) {
//		return ms;
//	}
	
	public default M mult_ab(M a, M b) {
		return matmul(a, b);
	}
	
	public M mult_aTb(M a, M b);
	
	public M mult_abT(M a, M b);
	
	public default M mult_aTbc(M a, M b, M c) {
		return matmul(mult_aTb(a, b), c);
	}
	
	public default M mult_abcT(M a, M b, M c) {
		return matmul(a, mult_abT(b, c));
	}
	
	public double sum(M m);
	
	public default double dist2(M v1, M v2) {
		double sum=0;
		for(int i=0; i<numElem(v1); i++) {
			double diff=get(v1, i)-get(v2, i);
			sum += diff*diff;
		}
		return sum;
	}
	
	public default M pairwiseDistances2(M a, M b) {
		M dists = zeros(numRows(b), numRows(a));
		for(int i=0; i<numRows(b); i++) {
			M bi = getRow(b, i);
			for(int j=0; j<numRows(a); j++) {
				M aj = getRow(a, j);
				double distSquared = dist2(bi, aj);
				set_inp(dists, i, j, distSquared);
			}
		}
		return dists;
	}
	
	default M[] sortEVD_inp(M[] evd) {
		double[] negvals = toArray(scale(diagV(evd[1]), -1));
		int[] order = argsort(negvals);
		M vectors = copy(evd[0]);
		for(int i=0; i<order.length; i++) {
			int j = order[i];
			M ev = getCol(vectors, j);
			for(int r=0; r<numElem(ev); r++) {
				set_inp(evd[0], r, i, get(ev, r));
			}
			set_inp(evd[1], i, i, -negvals[j]);
		}
		return evd;
	}
	
	
	public static int[] argsort(final double[] toSort) {
		Integer[] indices = IntStream.range(0, toSort.length).mapToObj(Integer::valueOf).toArray(Integer[]::new);
		Arrays.sort(indices, (i,j)->Double.compare(toSort[i], toSort[j]));
		return Arrays.stream(indices).mapToInt(Integer::intValue).toArray();
	}
	
}










