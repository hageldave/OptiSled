package hageldave.optisled.solver;

import java.util.LinkedList;

import org.ejml.simple.SimpleMatrix;

import hageldave.optisled.ejml.MatUtil;
import hageldave.optisled.problem.ScalarFN;
import hageldave.optisled.problem.VectorFN;

public class GradientDescent {

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
	public double terminationStepSize = 0.001;
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
	
	
	public SimpleMatrix arg_min(ScalarFN f, VectorFN df, SimpleMatrix initialGuess, TrajectoryInfo initialInfo){
		double a = initialStepSize;
		SimpleMatrix x = initialGuess.copy();
		int numSteps = 0;
		//
		double fx;
		SimpleMatrix dfx;
		SimpleMatrix d;
		do {
			fx = f.evaluate(x);
			if(initialInfo != null) {
				TrajectoryInfo info = initialInfo.copy();
				info.loss = fx;
				info.x = x;
				info.isGradientDescent = true;
				trajecInfo.add(info);
			}
			dfx = df.evaluate(x);
			d = MatUtil.normalizeInPlace(dfx.negative());
			// perform line search
			SimpleMatrix step;
			int numLinsrchIter = 0;
			// while( f(x+a*d) > f(x) + df(x)'a*d*l ) 1st wolfe condition
			while( 
					f.evaluate(x.plus(step=d.scale(a))) > fx + (dfx.dot(step))*lineSearchFactor 
					&& numLinsrchIter++ < maxLineSearchIter
			){
				a *= stepDecr;
			}
			// update location
			x = x.plus(step);
			a *= stepIncr;
			stepSizeOnTermination = a;
		} while( ++numSteps < maxDescentSteps && a*d.normF() > terminationStepSize );
		return x;
	}
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
	
}
