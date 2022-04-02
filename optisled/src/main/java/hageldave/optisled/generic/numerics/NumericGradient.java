package hageldave.optisled.generic.numerics;

import hageldave.optisled.generic.problem.ScalarFN;
import hageldave.optisled.generic.problem.VectorFN;

public class NumericGradient<M> implements VectorFN<M> {

	final ScalarFN<M> f;
	final MatCalc<M> mc;
	public double h = 0.0001;
	public final NumericCentralGradient central = new NumericCentralGradient();
	
	public NumericGradient(ScalarFN<M> f, MatCalc<M> mc) {
		this.f = f;
		this.mc = mc;
	}

	@Override
	public M evaluate(M vec) {
		// calculating forward differences
		final int dim = mc.numRows(vec);
		final double eps = h;
		final double divByEps = 1.0/eps;
		final double fx = f.evaluate(vec);
		M d = mc.zeros(dim);
		for(int i=0; i<dim; i++){
			double vec_i = mc.get(vec, i);
			mc.set_inp(vec, i, vec_i+eps);
			double diff = (f.evaluate(vec)-fx)*divByEps;
			mc.set_inp(vec, i, vec_i);
			mc.set_inp(d, i, diff);
		}
		return d;
	}
	
	
	public class NumericCentralGradient implements VectorFN<M> {

		@Override
		public M evaluate(M vec) {
			// calculating central differences
			final int dim = mc.numRows(vec);
			final double eps = h;
			final double divBy2Eps = 0.5/eps;
			M d = mc.zeros(dim);
			for(int i=0; i<dim; i++){
				double vec_i = mc.get(vec, i);
				mc.set_inp(vec, i, vec_i+eps);
				double evalplus = f.evaluate(vec);
				mc.set_inp(vec, i, vec_i-eps);
				double evalminus = f.evaluate(vec);
				mc.set_inp(vec, i, vec_i);
				double diff = (evalplus-evalminus)*divBy2Eps;
				mc.set_inp(d, i, diff);
			}
			return d;
		}
	}
	
}
