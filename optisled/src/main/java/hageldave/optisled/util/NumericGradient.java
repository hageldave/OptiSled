package hageldave.optisled.util;

import org.ejml.simple.SimpleMatrix;

import hageldave.optisled.problem.ScalarFN;
import hageldave.optisled.problem.VectorFN;

public class NumericGradient implements VectorFN {

	final ScalarFN f;
	
	public NumericGradient(ScalarFN f) {
		this.f = f;
	}

	@Override
	public SimpleMatrix evaluate(SimpleMatrix vec) {
		// calculating forward differences
		final int dim = vec.numRows();
		final double eps = 0.0001;
		final double divByEps = 1.0/eps;
		final double fx = f.evaluate(vec);
		SimpleMatrix d = MatUtil.vector(dim);
		for(int i=0; i<dim; i++){
			double vec_i = vec.get(i);
			vec.set(i, vec_i+eps);
			double diff = (f.evaluate(vec)-fx)*divByEps;
			vec.set(i, vec_i);
			d.set(i, diff);
		}
		return d;
	}
	
}
