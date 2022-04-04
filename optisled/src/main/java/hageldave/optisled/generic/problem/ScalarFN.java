package hageldave.optisled.generic.problem;

import hageldave.optisled.generic.numerics.MatCalc;

/**
 * Function taking vector input and giving scalar output.
 * @param <M> matrix (vector) type
 */
public interface ScalarFN<M> {
	
	public double evaluate(M vec);

	/**
	 * Special {@link ScalarFN} with gradient information
	 * @param <M> matrix type
	 */
	public interface ScalarFNWithGradient<M> extends ScalarFN<M> {
		public VectorFN<M> gradient();
	}
	
	public static <M> ScalarFNWithGradient<M> constant(final double c, MatCalc<M> mc) {
		return new ScalarFNWithGradient<M>() {
			@Override
			public double evaluate(M x) {
				return c;
			}
			@Override
			public VectorFN<M> gradient() {
				return x->mc.scale(x, 0.0);
			}
		};
	}
	
	public static <M> ScalarFNWithGradient<M> linear(final M coefficients, final double c, MatCalc<M> mc) {
		return new ScalarFNWithGradient<M>() {
			@Override
			public double evaluate(M x) {
				return mc.inner(coefficients,x) + c;
			}
			@Override
			public VectorFN<M> gradient() {
				return x->coefficients;
			}
		};
	}
	
}