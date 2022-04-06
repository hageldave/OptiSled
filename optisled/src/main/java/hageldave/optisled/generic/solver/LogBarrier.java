package hageldave.optisled.generic.solver;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import hageldave.optisled.generic.numerics.MatCalc;
import hageldave.optisled.generic.problem.OptimizationProblem;
import hageldave.optisled.generic.problem.ScalarFN.ScalarFNWithGradient;
import hageldave.optisled.generic.problem.VectorFN;

/**
 * Log-Barrier solver for nonlinear constrained optimization problems.
 * This solver needs feasible initialization. i.e. init has to satisfy all constraints.
 * There is a strategy to walk into a feasible region from outside if initialization is infeasible,
 * but it is not guaranteed to succeed.
 * @param <M> matrix type
 */
public class LogBarrier<M> {

	public double initialMu = 8.0;
	public double muDecr = 0.95;
	public int maxNumIterations = 300;
	
	public final MatCalc<M> mc;
	
	public LogBarrier(MatCalc<M> mc) {
		this.mc = mc;
	}

	public M arg_min(OptimizationProblem<M> p, M initialGuess) {
		return arg_min(p, initialGuess, null);
	}
	
	public M arg_min(OptimizationProblem<M> p, M initialGuess, List<TrajectoryInfo> trace) {
		double mu = initialMu;
		M x = mc.copy(initialGuess);
		if(Objects.nonNull(trace)){
			TrajectoryInfo info = new TrajectoryInfo();
			info.x = mc.toArray(x);
			info.fx = p.f().evaluate(x);
			M x_=x;
			info.gx = Arrays.stream(p.g()).mapToDouble(g->g.evaluate(x_)).toArray();
			double currmu = mu;
			info.lambda = Arrays.stream(info.gx).map(gx->-currmu/gx).toArray();
			info.loss = logBarrFN(p, mu, mc).evaluate(x);
			info.mu = mu;
			trace.add(info);
		}
		int numIterations = 0;
		do {
			ScalarFNWithGradient<M> f = logBarrFN(p, mu, mc);
			GradientDescent<M> gd = new GradientDescent<>(mc);
			gd.maxDescentSteps = 100;
			x = gd.arg_min(f, f.gradient(), x, null);
			if(Objects.nonNull(trace)){
				TrajectoryInfo info = new TrajectoryInfo();
				info.x = mc.toArray(x);
				info.fx = p.f().evaluate(x);
				M x_=x;
				info.gx = Arrays.stream(p.g()).mapToDouble(g->g.evaluate(x_)).toArray();
				double currmu = mu;
				info.lambda = Arrays.stream(info.gx).map(gx->-currmu/gx).toArray();
				info.loss = f.evaluate(x);
				info.mu = mu;
				trace.add(info);
			}
			mu *= muDecr;
		} while(++numIterations < maxNumIterations);
		return x;
	}
	
	public static <M> ScalarFNWithGradient<M> logBarrFN(OptimizationProblem<M> p, double mu, MatCalc<M> mc){
		return new ScalarFNWithGradient<M>() {
			
			@Override
			public double evaluate(M x) {
				double result = p.f().evaluate(x);
				for(int i=0; i<p.numConstraints(); i++){
					// result -= mu*log( max(0,-gi(x)) )
					result -= mu*Math.log( Math.max(0, -p.g()[i].evaluate(x)) );
				}
				return result;
			}
			
			VectorFN<M> grad = new VectorFN<M>() {
				@Override
				public M evaluate(M x) {
					M result = mc.normalize(p.df().evaluate(x));
					for(int i=0; i<p.numConstraints(); i++){
						double gx = p.g()[i].evaluate(x);
						M dgx = mc.normalize(p.dg()[i].evaluate(x));
						if(gx < 0){
							// result -= (mu/gx) * dgx
							result = mc.sub(result, mc.scale(dgx, mu/gx));
						} else {
							// handling of gradient for nondefined negative logarithms in infeasible regions
							result = mc.add(result,mc.scale(dgx,1+gx));
						}
					}
					return result;
				}
			};
			
			@Override
			public VectorFN<M> gradient() {
				return grad;
			}
		};
	}
	
	
}
