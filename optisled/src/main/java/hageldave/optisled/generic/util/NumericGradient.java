package hageldave.optisled.generic.util;

import hageldave.optisled.generic.problem.ScalarFN;
import hageldave.optisled.generic.problem.VectorFN;

public class NumericGradient<M> implements VectorFN<M> {

	final ScalarFN<M> f;
	final MatCalc<M> mc;
	
	public NumericGradient(ScalarFN<M> f, MatCalc<M> mc) {
		this.f = f;
		this.mc = mc;
	}

	@Override
	public M evaluate(M vec) {
		// calculating forward differences
		final int dim = mc.numRows(vec);
		final double eps = 0.0001;
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
	
}
