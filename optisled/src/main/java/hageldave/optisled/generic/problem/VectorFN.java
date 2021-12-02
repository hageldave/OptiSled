package hageldave.optisled.generic.problem;

import hageldave.optisled.generic.numerics.MatCalc;

public interface VectorFN<M> {
	public M evaluate(M vec);
	
	public static <M> VectorFN<M> constant(final M c){
		return x->c;
	}
	
	public static <M> VectorFN<M> linear(final M transform, final M c, MatCalc<M> mc){
		return x->mc.add(mc.matmul(transform,x),c);
	}
}