package hageldave.optisled.problem;

public interface OptimizationProblem {
	
	public int dimensionality();
	
	/** objective function f(x) */
	public ScalarFN f();
	
	/** inequality constraints gi(x) LEQ 0 */
	public ScalarFN[] g();
	
	/** derivative of objective */
	public VectorFN df();
	
	/** derivatives of constraints */
	public VectorFN[] dg();
	
	public default int numConstraints() {
		return g().length;
	}
	
}
