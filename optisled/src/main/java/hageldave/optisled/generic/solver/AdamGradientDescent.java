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
 * {@link #arg_min(ScalarFN, VectorFN, Object)}, need to change behavior to only correspond to a subset of
 * the dimensions (based on the random number).
 * <p>
 * Descent is run by calling {@link #arg_min(ScalarFN, VectorFN, M, DescentLog)}.
 * @param <M> matrix type
 */
public class AdamGradientDescent<M> {

	public double beta1 = 0.9;
	
	public double beta2 = 0.999;
	/**
	 * initial step size the algorithm starts with
	 */
	public double initialStepSize = 1.0;
	/**
	 * when the algorithm's steps have decreased below this step size threshold
	 * it terminates, thinking it has reached the minimum
	 */
	public double terminationStepSize = 1e-8;
	/**
	 * maximum number of descent steps to take
	 * (preventing infinite loops in ill conditioned problems) 
	 */
	public int maxDescentSteps = 100;

	/** the step size of gd when argmin terminated */
	public double stepSizeOnTermination;

	/** the matrix calculation object for the matrix type M */
	public final MatCalc<M> mc;

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

	/**
	 * finds argmin by performing gradient descent
	 * @param f function to be minimized
	 * @param df gradient of the function
	 * @param initialGuess initialization (guess of minimum location)
	 * @param log optional log object for recording the optimization trajectory (can be null)
	 * @return location of minimum
	 */
	public M arg_min(ScalarFN<M> f, VectorFN<M> df, M initialGuess, DescentLog log){
		double a = initialStepSize;
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
			v = mc.add(mc.scale(v, beta2) , mc.scale(mc.elmmul(dfx,dfx), 1.0-beta2));
			
			double alpha = a * Math.sqrt(1-Math.pow(beta2, numSteps+1)) / (1-Math.pow(beta1, numSteps+1));
			step = mc.elemwise_inp(v, AdamGradientDescent::divBySqrtSanitized);
			step = mc.scale(mc.elmmul(m, step), -alpha);
			
			if(log != null) {
				log.position(mc.toArray(x));
				log.loss(fx);
				log.direction(mc.toArray(dfx));
				log.stepSize(alpha);
			}
			
			// update location
			x = mc.add(x,step);
			stepSizeOnTermination = mc.norm(step);
		} while( ++numSteps < maxDescentSteps &&  stepSizeOnTermination > terminationStepSize );

		this.lossOnTermination = f.evaluate(x);
		if(log != null) {
			log.position(mc.toArray(x));
			log.loss(lossOnTermination);
		}
		
		return x;
	}
	
	private static double divBySqrtSanitized(double val) {
		return 1.0/(Math.sqrt(val) + 1e-9);
	}
}
