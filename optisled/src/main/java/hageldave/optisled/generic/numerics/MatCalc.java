package hageldave.optisled.generic.numerics;

public interface MatCalc<M> {
	
	public M vecOf(double ... v);
	
	public M zeros(int size);
	
	public int numRows(M m);
	
	public int numCols(M m);
	
	public double inner(M a, M b);
	
	public M scale(M m, double s);
	
	public M scale_inp(M m, double s);
	
	public M matmul(M a, M b);
	
	public M add(M a, M b);
	
	public M sub(M a, M b);
	
	public M set_inp(M m, int idx, double v);
	
	public double get(M m, int idx);
	
	public M copy(M m);
	
	public double[] toArray(M m);
	
	public double[][] toArray2D(M m);
	
	public default M normalize_inp(M m, double thresh) {
		double norm = norm(m);
		return norm < thresh ? m : scale_inp(m, 1/norm);
	}
	
	public default M normalize(M m, double thresh) { return normalize_inp(copy(m), thresh); }
	
	public default M normalize_inp(M m) { return normalize_inp(m, 1e-7); }
	
	public default M normalize(M m) { return normalize_inp(copy(m)); }
	
	public default double norm2(M m) { return inner(m,m); }
	
	public default double norm(M m) { return Math.sqrt(norm2(m)); }
	
}
