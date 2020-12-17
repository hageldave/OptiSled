package hageldave.optisled.problem;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

import hageldave.optisled.problem.ScalarFN.ScalarFNWithGradient;
import hageldave.optisled.util.NumericGradient;

public class OptimizationProblemBuilder {

	public static class OptimizationProblemImpl implements OptimizationProblem {
		
		int dim;
		
		ScalarFN objective;
		VectorFN objective_deriv;
		
		ScalarFN[] constraints;
		VectorFN[] constraint_derivs;
		
		@Override
		public int dimensionality() {
			return dim;
		}
		
		@Override
		public ScalarFN f() {
			return objective;
		}
		
		@Override
		public VectorFN df() {
			return objective_deriv;
		}
		
		@Override
		public ScalarFN[] g() {
			return constraints;
		}
		
		@Override
		public VectorFN[] dg() {
			return constraint_derivs;
		}
		
	}
	
	int dim;
	
	private ScalarFN objective;
	private VectorFN objective_deriv;
	
	private List<ScalarFN> constraints = new LinkedList<>();
	private List<VectorFN> constraint_derivs= new LinkedList<>();
	
	private OptimizationProblemBuilder() {}
	
	public static OptimizationProblemBuilder instance(int dim) {
		OptimizationProblemBuilder b = new OptimizationProblemBuilder();
		b.dim = dim;
		return b;
	}
	
	public OptimizationProblem build() {
		OptimizationProblemImpl impl = new OptimizationProblemImpl();
		impl.dim = dim;
		impl.objective = objective;
		impl.objective_deriv = objective_deriv;
		impl.constraints = constraints.toArray(new ScalarFN[0]);
		impl.constraint_derivs = constraint_derivs.toArray(new VectorFN[0]);
		return impl;
	}
	
	/**
	 * sets objective function to minimize
	 * @param f
	 * @param df optional gradient (can be null)
	 * @return this for chaining
	 */
	public OptimizationProblemBuilder setObjective(ScalarFN f, VectorFN df){
		this.objective = Objects.requireNonNull(f);
		if(Objects.isNull(df)){
			if(f instanceof ScalarFNWithGradient){
				df = ((ScalarFNWithGradient) f).gradient();
			} else {
				df = new NumericGradient(f);
			}
		}
		this.objective_deriv = Objects.requireNonNull(df);
		return this;
	}
	
	/**
	 * adds constraint g(x) LEQ 0
	 * @param g
	 * @param dg optional gradient (can be null)
	 * @return this for chaining
	 */
	public OptimizationProblemBuilder addIneqConstraint(ScalarFN g, VectorFN dg){
		this.constraints.add(Objects.requireNonNull(g));
		if(Objects.isNull(dg)){
			if(g instanceof ScalarFNWithGradient){
				dg = ((ScalarFNWithGradient) g).gradient();
			} else {
				dg = new NumericGradient(g);
			}
		}
		this.constraint_derivs.add(Objects.requireNonNull(dg));
		return this;
	}
	
	
}
