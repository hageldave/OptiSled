package hageldave.optisled.problem;

import org.ejml.simple.SimpleMatrix;

import hageldave.optisled.util.MatUtil;

public interface ScalarFN {
	
	public double evaluate(SimpleMatrix vec);
	
	public default double evaluate(double ...vecValues){
		return evaluate(MatUtil.vectorOf(vecValues));
	}
	
	public interface ScalarFNWithGradient extends ScalarFN {
		public VectorFN gradient();
	}
	
	public static ScalarFNWithGradient constant(final double c) {
		return new ScalarFNWithGradient() {
			@Override
			public double evaluate(SimpleMatrix x) {
				return c;
			}
			@Override
			public VectorFN gradient() {
				return x->x.scale(0);
			}
		};
	}
	
	public static ScalarFNWithGradient linear(final SimpleMatrix coefficients, final double c) {
		return new ScalarFNWithGradient() {
			@Override
			public double evaluate(SimpleMatrix x) {
				return coefficients.dot(x) + c;
			}
			@Override
			public VectorFN gradient() {
				return x->coefficients;
			}
		};
	}
	
}