package hageldave.optisled.generic.problem;

public interface OptimizationProblem<M> {
	
	public int dimensionality();
	
	/** objective function f(x) */
	public ScalarFN<M> f();
	
	/** inequality constraints gi(x) LEQ 0 */
	public ScalarFN<M>[] g();
	
	/** derivative of objective */
	public VectorFN<M> df();
	
	/** derivatives of constraints */
	public VectorFN<M>[] dg();
	
	public default int numConstraints() {
		return g().length;
	}
	
}
