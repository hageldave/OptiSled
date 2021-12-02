package hageldave.optisled.generic.problem;

import java.util.LinkedList;
import java.util.List;
import java.util.Objects;

import hageldave.optisled.generic.numerics.MatCalc;
import hageldave.optisled.generic.numerics.NumericGradient;
import hageldave.optisled.generic.problem.ScalarFN.ScalarFNWithGradient;

public class OptimizationProblemBuilder<M> {

	public static class OptimizationProblemImpl<M> implements OptimizationProblem<M> {
		
		int dim;
		
		ScalarFN<M> objective;
		VectorFN<M> objective_deriv;
		
		ScalarFN<M>[] constraints;
		VectorFN<M>[] constraint_derivs;
		
		@Override
		public int dimensionality() {
			return dim;
		}
		
		@Override
		public ScalarFN<M> f() {
			return objective;
		}
		
		@Override
		public VectorFN<M> df() {
			return objective_deriv;
		}
		
		@Override
		public ScalarFN<M>[] g() {
			return constraints;
		}
		
		@Override
		public VectorFN<M>[] dg() {
			return constraint_derivs;
		}
		
	}
	
	int dim;
	MatCalc<M> mc;
	
	private ScalarFN<M> objective;
	private VectorFN<M> objective_deriv;
	
	private List<ScalarFN<M>> constraints = new LinkedList<>();
	private List<VectorFN<M>> constraint_derivs= new LinkedList<>();
	
	private OptimizationProblemBuilder(MatCalc<M> mc) {
		this.mc = mc;
	}
	
	public static <M> OptimizationProblemBuilder<M> instance(int dim, MatCalc<M> mc) {
		OptimizationProblemBuilder<M> b = new OptimizationProblemBuilder<>(mc);
		b.dim = dim;
		return b;
	}
	
	@SuppressWarnings("unchecked")
	public OptimizationProblem<M> build() {
		OptimizationProblemImpl<M> impl = new OptimizationProblemImpl<>();
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
	public OptimizationProblemBuilder<M> setObjective(ScalarFN<M> f, VectorFN<M> df){
		this.objective = Objects.requireNonNull(f);
		if(Objects.isNull(df)){
			if(f instanceof ScalarFNWithGradient){
				df = ((ScalarFNWithGradient<M>) f).gradient();
			} else {
				df = new NumericGradient<M>(f, mc);
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
	public OptimizationProblemBuilder<M> addIneqConstraint(ScalarFN<M> g, VectorFN<M> dg){
		this.constraints.add(Objects.requireNonNull(g));
		if(Objects.isNull(dg)){
			if(g instanceof ScalarFNWithGradient){
				dg = ((ScalarFNWithGradient<M>) g).gradient();
			} else {
				dg = new NumericGradient<M>(g,mc);
			}
		}
		this.constraint_derivs.add(Objects.requireNonNull(dg));
		return this;
	}
	
	
}
