package hageldave.optisled.generic.solver;

import java.util.LinkedList;

import hageldave.optisled.generic.problem.ScalarFN;
import hageldave.optisled.generic.problem.VectorFN;
import hageldave.optisled.generic.util.MatCalc;

public class GradientDescent<M> {

	/** 
	 * factor by which the step size is decreased during line search, 
	 * in ]0,1[ 
	 */
	public double stepDecr = 0.5;
	/** 
	 * factor by which the step size is increased again after a step has 
	 * been taken, in [1,infty] 
	 */ 
	public double stepIncr = 1.2;
	/** 
	 * factor for determining sufficient decrease during line search 
	 * (see 1st wolfe condition), in [0.01,0.1] typically 
	 */
	public double lineSearchFactor = 0.01;
	/**
	 * initial step size the algorithm starts with
	 */
	public double initialStepSize = 1.0;
	/**
	 * when the algorithm's steps have decreased below this step size threshold
	 * it terminates, thinking it has reached the minimum
	 */
	public double terminationStepSize = 1e-5;
	/**
	 * maximum number of line search iterations 
	 * (preventing infinite loops in ill conditioned problems) 
	 */
	public int maxLineSearchIter = 100;
	/**
	 * maximum number of descent steps to take
	 * (preventing infinite loops in ill conditioned problems) 
	 */
	public int maxDescentSteps = 100;
	
	public double stepSizeOnTermination;
	
	public LinkedList<TrajectoryInfo> trajecInfo = new LinkedList<>();
	
	public final MatCalc<M> mc;
	
	public GradientDescent(MatCalc<M> mc) {
		this.mc = mc;
	}
	
	
	public M arg_min(ScalarFN<M> f, VectorFN<M> df, M initialGuess, TrajectoryInfo initialInfo){
		double a = initialStepSize;
		M x = mc.copy(initialGuess);
		int numSteps = 0;
		//
		double fx;
		M dfx;
		M d;
		do {
			fx = f.evaluate(x);
			if(initialInfo != null) {
				TrajectoryInfo info = initialInfo.copy();
				info.loss = fx;
				info.x = mc.toArray(x);
				info.isGradientDescent = true;
				trajecInfo.add(info);
			}
			dfx = df.evaluate(x);
			d = mc.normalize_inp(mc.scale(dfx, -1.0));
			// perform line search
			M step;
			int numLinsrchIter = 0;
			// while( f(x+a*d) > f(x) + df(x)'a*d*l ) 1st wolfe condition
			while( 
					f.evaluate(mc.add(x, step=mc.scale(d,a))) > fx + mc.inner(dfx,step)*lineSearchFactor
					&& numLinsrchIter++ < maxLineSearchIter
			){
				a *= stepDecr;
			}
			// update location
			x = mc.add(x,step);
			a *= stepIncr;
			stepSizeOnTermination = a;
		} while( ++numSteps < maxDescentSteps && a*mc.norm(d) > terminationStepSize );
		return x;
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
