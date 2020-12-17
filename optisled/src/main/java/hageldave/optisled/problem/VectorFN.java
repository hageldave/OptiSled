package hageldave.optisled.problem;

import org.ejml.simple.SimpleMatrix;

import hageldave.optisled.util.MatUtil;

public interface VectorFN {
	public SimpleMatrix evaluate(SimpleMatrix vec);
	
	public default SimpleMatrix evaluate(double ...vecValues){
		return evaluate(MatUtil.vectorOf(vecValues));
	}
	
	public static VectorFN constant(final SimpleMatrix c){
		return x->c;
	}
	
	public static VectorFN linear(final SimpleMatrix transform, final SimpleMatrix c){
		return x->transform.mult(x).plus(c);
	}
}