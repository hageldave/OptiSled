package hageldave.optisled.generic.solver;

import hageldave.optisled.generic.numerics.MatCalc;
import hageldave.optisled.generic.problem.ScalarFN;
import hageldave.optisled.generic.problem.VectorFN;
import hageldave.optisled.history.DescentLog;

/**
 * Gradient descent implementation with line search (satisfying 1st Wolfe condition in each step).
 * Descent is run by calling {@link #arg_min(ScalarFN, VectorFN, Object, DescentLog)}.
 * @param <M> matrix type
 */
public class GradientDescent<M> implements DescentAlgorithm<M> {
	
	public static class HyperparamsGD extends Hyperparams {
		{
			set(PARAM_STEP_DECR, 0.5);
			set(PARAM_STEP_INCR, 1.2);
			set(PARAM_INIT_STEPSIZE, 1.0);
			set(PARAM_TERMINATION_STEPSIZE, 1e-8);
			set(PARAM_LINESEARCH_FACTOR, 0.01);
			set(PARAM_MAX_ITERATIONS, 100);
			set(PARAM_MAX_LINESEARCH_ITER, 20);
		}
	}
	
	/**
	 * step size (alpha)
	 */
	public static final String PARAM_INIT_STEPSIZE = "INIT_STEPSIZE";
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
	/**
	 * maximum number of line search iterations. 
	 * Determines max decrease of stepsize per iteration, = STEP_DECR ^ MAX_LINESEARCH_ITER
	 * (preventing infinite loops in ill conditioned problems) 
	 */
	public static final String PARAM_MAX_LINESEARCH_ITER = "MAX_LINESEARCH_ITER";
	/** 
	 * factor by which the step size is decreased during line search, 
	 * in ]0,1[ 
	 */
	public static final String PARAM_STEP_DECR = "STEP_DECR";
	/** 
	 * factor by which the step size is increased again after a step has 
	 * been taken, in [1,infty] 
	 */ 
	public static final String PARAM_STEP_INCR = "PARAM_STEP_INCR";
	/** 
	 * factor for determining 'sufficient decrease' during line search 
	 * (see 1st wolfe condition), in [0.01,0.1] typically 
	 */
	public static final String PARAM_LINESEARCH_FACTOR = "LINESEARCH_FACTOR";
	
	// 

	/** the hyperparameters for gradient descent with adaptive step size (through line search) */
	public Hyperparams hyperparams = new HyperparamsGD();
	
	/** the matrix calculation object for the matrix type M */
	public final MatCalc<M> mc;
	
	/** the step size of gd when argmin terminated */
	public double stepSizeOnTermination;

	/** the loss when arg_min terminates */
	public double lossOnTermination;

	/**
	 * Creates a new GD instance for matrices of type M using
	 * specified matrix calculator.
	 * @param mc matrix calculator to perform linear algebra calculations
	 */
	public GradientDescent(MatCalc<M> mc) {
		this.mc = mc;
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
	
	@Override
	public double getLoss() {
		return this.lossOnTermination;
	}
	
}
