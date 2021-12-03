package hageldave.optisled.generic.numerics;

import java.util.Random;

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
	
	/** @return column vector of given values */
	public M vecOf(double ... v);
	
	/** @return matrix with {@code nRows} and {@code nCols=values.length/nRows} filled with specified values (row major order) */
	public M matOf(int nRows, double... values);
	
	/** @return matrix from the specified 2d array {@code (rowValues[] = values[i])} */
	public M matOf(double[][] values);
	
	/** @return zero column vector of dimension {@code size} */
	public M zeros(int size);
	
	/** @return matrix of zeros of specified size */
	public M zeros(int rows, int columns);
	
	/** @return random matrix of values in range 0.0 .. 1.0 */
	public default M rand(int rows, int cols, Random rnd) {
		M m = zeros(rows, cols);
		for(int r=0; r<rows; r++)
			for(int c=0; c<cols; c++)
				set_inp(m, r, c, rnd.nextDouble());
		return m;
	}
	
	/** @return random matrix of values in range 0.0 .. 1.0 */
	public default M rand(int rows, int cols) {
		return rand(rows, cols, new Random());
	}
	
	/** @return number of rows */
	public int numRows(M m);
	
	/** @return number of columns */
	public int numCols(M m);
	
	/** @return inner or dot product for given vectors ‹a,b› */
	public double inner(M a, M b);
	
	/** @return a scaled version of given matrix/vector */
	public M scale(M m, double s);
	
	/** @return the same matrix/vector argument which was scaled in place */
	public M scale_inp(M m, double s);
	
	/** @return result of matrix multiplication {@code a * b} */
	public M matmul(M a, M b);
	
	/** @return matrix transpose */
	public M trp(M m);
	
	/** @return result of matrix/vector addition {@code a + b} */
	public M add(M a, M b);
	
	/** @return result of matrix/vector subtraction {@code a - b} */
	public M sub(M a, M b);
	
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
	
}
