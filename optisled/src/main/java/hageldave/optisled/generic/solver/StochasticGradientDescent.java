package hageldave.optisled.generic.solver;

import java.util.Random;

import hageldave.optisled.generic.numerics.MatCalc;
import hageldave.optisled.generic.problem.ScalarFN;
import hageldave.optisled.generic.problem.VectorFN;
import hageldave.optisled.history.DescentLog;
import hageldave.utils.Ref;

/**
 * Stochastic Gradient descent implementation with line search (satisfying 1st Wolfe condition in each step).
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
public class StochasticGradientDescent<M> extends GradientDescent<M> {
	
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
	public StochasticGradientDescent(MatCalc<M> mc) {
		super(mc);
		this.rand = new Random();
	}

	@Override
	public M arg_min(ScalarFN<M> f, VectorFN<M> df, M initialGuess, DescentLog log){
		// hyperparameters
		double a = hyperparams.getOrDefault(PARAM_INIT_STEPSIZE, 1.0);
		double stepDecr = hyperparams.getOrDefault(PARAM_STEP_DECR, 0.5);
		double stepIncr = hyperparams.getOrDefault(PARAM_STEP_INCR, 1.2);
		double terminationStepSize = hyperparams.getOrDefault(PARAM_TERMINATION_STEPSIZE, 1e-8);
		double lineSearchFactor = hyperparams.getOrDefault(PARAM_LINESEARCH_FACTOR, 0.01);
		int maxDescentSteps = hyperparams.getOrDefault(PARAM_MAX_ITERATIONS, 100);
		int maxLineSearchIter = hyperparams.getOrDefault(PARAM_MAX_LINESEARCH_ITER, 20);

		
		M x = mc.copy(initialGuess);
		int numSteps = 0;
		//
		double fx;
		M dfx;
		M d;
		M step;
		do {
			int r = rand.nextInt(Integer.MAX_VALUE);
			if(randRef != null) 
				randRef.set(r);
			fx = f.evaluate(x);
			dfx = df.evaluate(x);
			d = mc.normalize_inp(mc.scale(dfx, -1.0));
			if(log != null) {
				log.position(mc.toArray(x));
				log.loss(fx);
				log.direction(mc.toArray(d));
				log.stepSize(a);
			}
			// perform line search
			int numLinsrchIter = 0;
			// while( f(x+a*d) > f(x) + df(x)'a*d*l ) 1st wolfe condition
			while( 
					f.evaluate(mc.add(x, step=mc.scale(d,a))) > fx + mc.inner(dfx,step)*lineSearchFactor
					&& numLinsrchIter++ < maxLineSearchIter
			){
				a *= stepDecr;
				if(log != null)
					log.stepSize(a);
			}
			// update location
			x = mc.add(x,step);
			stepSizeOnTermination = a;
			a *= stepIncr;
		} while( ++numSteps < maxDescentSteps && mc.norm(step) > terminationStepSize );

		this.lossOnTermination = f.evaluate(x);
		if(log != null) {
			log.position(mc.toArray(x));
			log.loss(lossOnTermination);
		}
		
		return x;
	}
	
}
