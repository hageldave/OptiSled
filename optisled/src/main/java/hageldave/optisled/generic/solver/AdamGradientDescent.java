package hageldave.optisled.generic.solver;

import java.util.Random;

import hageldave.optisled.generic.numerics.MatCalc;
import hageldave.optisled.generic.problem.ScalarFN;
import hageldave.optisled.generic.problem.VectorFN;
import hageldave.optisled.history.DescentLog;
import hageldave.utils.Ref;

/**
 * Stochastic Gradient descent implementation with Adam.
 * <p>
 * Stochasticity has to be realized through the {@link #randRef} object which holds a random number which is 
 * changing in each iteration (change events are fired and can be listened to).
 * When the random number changes, the objective function and gradient, that were passed to 
 * {@link #arg_min(ScalarFN, VectorFN, Object, DescentLog)}, need to change behavior to only correspond to a subset of
 * the dimensions (based on the random number).
 * <p>
 * Descent is run by calling {@link #arg_min(ScalarFN, VectorFN, Object, DescentLog)}.
 * @param <M> matrix type
 */
public class AdamGradientDescent<M> implements DescentAlgorithm<M> {
	
	public static class HyperparamsAdam extends Hyperparams {
		{
			set(PARAM_BETA1, 0.9);
			set(PARAM_BETA2, 0.999);
			set(PARAM_STEP_SCALING, 1.0);
			set(PARAM_TERMINATION_STEPSIZE, 1e-8);
			set(PARAM_MAX_ITERATIONS, 100);
		}
	}
	
	/**
	 * exponential decay of first moment.
	 */
	public static final String PARAM_BETA1 = "BETA1";
	/**
	 * exponential decay of second moment.
	 */
	public static final String PARAM_BETA2 = "BETA2";
	/**
	 * step size (alpha)
	 */
	public static final String PARAM_STEP_SCALING = "STEP_SCALING";
	/**
	 * when the algorithm's steps have decreased below this step size threshold
	 * it terminates, thinking it has reached the minimum
	 */
	public static final String PARAM_TERMINATION_STEPSIZE = "TERMINATION_STEPSIZE";
	/**
	 * maximum number of descent steps to take
	 * (preventing infinite loops in ill conditioned problems) 
	 */
	public static final String PARAM_MAX_ITERATIONS = "MAX_ITERATIONS";
	
	//
	
	/** the hyperparameters for Adam */
	public Hyperparams hyperparams = new HyperparamsAdam();

	/** the matrix calculation object for the matrix type M */
	public final MatCalc<M> mc;
	
	/** the step size of gd when argmin terminated */
	public double stepSizeOnTermination;

	/** the loss when arg_min terminates */
	public double lossOnTermination;

	/** RNG, used to generate a new number on each iteration  */
	public Random rand;
	/** 
	 * Reference to the random number of the current iteration.
	 * Should be used to alter which part of the loss and respective gradient 
	 * is returned by {@code f} and {@code df} in {@link #arg_min(ScalarFN, VectorFN, Object, DescentLog)}.
	 * This way stochastic gradient descent can be realized.
	 */
	public Ref<Integer> randRef = new Ref<>();

	/**
	 * Creates a new GD instance for matrices of type M using
	 * specified matrix calculator.
	 * @param mc matrix calculator to perform linear algebra calculations
	 */
	public AdamGradientDescent(MatCalc<M> mc) {
		this.mc=mc;
		this.rand = new Random();
	}
	
	@Override
	public Hyperparams getHyperparams() {
		return this.hyperparams;
	}
	
	@Override
	public void setHyperparams(Hyperparams hyperparams) {
		this.hyperparams = hyperparams;
	}

	@Override
	public M arg_min(ScalarFN<M> f, VectorFN<M> df, M initialGuess, DescentLog log){
		// get hyperparams
		double a = hyperparams.getOrDefault(PARAM_STEP_SCALING, 1.0);
		double beta1 = hyperparams.getOrDefault(PARAM_BETA1, 0.9);
		double beta2 = hyperparams.getOrDefault(PARAM_BETA2, 0.999);
		double terminationStepSize = hyperparams.getOrDefault(PARAM_TERMINATION_STEPSIZE, 1e-8);
		int maxIter = hyperparams.getOrDefault(PARAM_MAX_ITERATIONS, 100);
		
		
		M x = mc.copy(initialGuess);
		int numSteps = 0;
		//
		double fx;
		M dfx;
		M step;
		// adam things
		M m = mc.scale(x, 0.0);
		M v = mc.scale(x, 0.0);
		do {
			int r = rand.nextInt(Integer.MAX_VALUE);
			if(randRef != null) 
				randRef.set(r);
			fx = f.evaluate(x);
			dfx = df.evaluate(x);
			
			m = mc.add(mc.scale(m, beta1) , mc.scale(dfx, 1.0-beta1));
			v = mc.add(mc.scale(v, beta2) , mc.scale(mc.elemmul(dfx,dfx), 1.0-beta2));
			
			double alpha = a * (Math.sqrt(1-Math.pow(beta2, numSteps+1)) / (1-Math.pow(beta1, numSteps+1)));
			step = mc.elemwise_inp(v, AdamGradientDescent::divBySqrtSanitized);
			step = mc.scale(mc.elemmul(m, step), -alpha);
			
			if(log != null) {
				log.position(mc.toArray(x));
				log.loss(fx);
				log.direction(mc.toArray(dfx));
				log.stepSize(alpha);
			}
			
			// update location
			x = mc.add(x,step);
			stepSizeOnTermination = mc.norm(step);
		} while( ++numSteps < maxIter &&  stepSizeOnTermination > terminationStepSize );

		this.lossOnTermination = f.evaluate(x);
		if(log != null) {
			log.position(mc.toArray(x));
			log.loss(lossOnTermination);
		}
		
		return x;
	}
	
	@Override
	public double getLoss() {
		return this.lossOnTermination;
	}
	
	private static double divBySqrtSanitized(double val) {
		return 1.0/(Math.sqrt(val) + 1e-9);
	}
}
