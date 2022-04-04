package hageldave.optisled.generic.solver;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import hageldave.optisled.generic.numerics.MatCalc;
import hageldave.optisled.generic.problem.OptimizationProblem;
import hageldave.optisled.generic.problem.ScalarFN.ScalarFNWithGradient;
import hageldave.optisled.generic.problem.VectorFN;
import hageldave.optisled.history.DescentLog;

/**
 * Augmented Lagrangian solver for nonlinear constrained optimization problems.
 * @param <M> matrix type
 */
public class AugmentedLagrangian<M> {

	public int maxNumIterations = 80;
	public double muIncr = 1.01;
	public double initialStepsize = 1.0;
	
	public final MatCalc<M> mc;
	
	public AugmentedLagrangian(MatCalc<M> mc) {
		this.mc=mc;
	}
	
	
	public M arg_min(OptimizationProblem<M> p, M initialGuess) {
		return arg_min(p, initialGuess, null);
	}
	
	public M arg_min(OptimizationProblem<M> p, M initialGuess, List<TrajectoryInfo> trace) {
		double[] lambda = new double[p.numConstraints()];
		double mu = 1;
		M x = mc.copy(initialGuess);
		int numIterations = 0;
		do {
			TrajectoryInfo info = new TrajectoryInfo();
			if(Objects.nonNull(trace)){
				info = new TrajectoryInfo();
				info.x = mc.toArray(x);
				info.fx = p.f().evaluate(x);
				M x_=x;
				info.gx = Arrays.stream(p.g()).mapToDouble(g->g.evaluate(x_)).toArray();
				info.lambda = lambda.clone();
				info.loss = augLagrangian(p, lambda, mu, mc).evaluate(x);
				info.mu = mu;
			}
			
			ScalarFNWithGradient<M> f = augLagrangian(p, lambda, mu, mc);
			GradientDescent<M> gd = new GradientDescent<>(mc);
			gd.initialStepSize = initialStepsize;
			DescentLog descentLog = null; // TODO: conditionally create a descent log
			x = gd.arg_min(f, f.gradient(), x, descentLog);
			
			if(Objects.nonNull(trace)){	
				info = new TrajectoryInfo();
				info.x = mc.toArray(x);
				info.fx = p.f().evaluate(x);
				TrajectoryInfo info_ = info;
				info.gx = Arrays.stream(p.g()).mapToDouble(g->g.evaluate(mc.vecOf(info_.x))).toArray();
				info.lambda = lambda.clone();
				info.loss = f.evaluate(x);
				info.mu = mu;
				trace.add(info);
			}
			
			
			for(int i=0; i<p.numConstraints(); i++){
				lambda[i] = Math.max(0, lambda[i] + p.g()[i].evaluate(x)*2*mu);
			}
			mu *= muIncr;
		} while(++numIterations < maxNumIterations);
		return x;
	}
	
	
	public static <M> ScalarFNWithGradient<M> augLagrangian(OptimizationProblem<M> p, double[] lambda, double mu, MatCalc<M> mc){
		return new ScalarFNWithGradient<M>() {
			
			@Override
			public double evaluate(M x) {
				double result = p.f().evaluate(x);
				for(int i=0; i<p.numConstraints(); i++){
					double gx = p.g()[i].evaluate(x);
					// += lambda_i * g(x) + [g(x) > 0] mu * g(x)^2
					result += lambda[i]*gx;
					result += Math.max(0, gx)*gx*mu;
				}
				return result;
			}
			
			VectorFN<M> grad = new VectorFN<M>() {
				
				@Override
				public M evaluate(M x) {
					M result = p.df().evaluate(x);
					for(int i=0; i<p.numConstraints(); i++){
						double gx = p.g()[i].evaluate(x);
						M dgx = p.dg()[i].evaluate(x);
						// += lambda_i * dg(x) + [g(x) > 0] * 2mu * dg(x)
						result = mc.add(result, mc.scale(dgx,lambda[i]));
						if(gx > 0){
							result = mc.add(result, mc.scale(dgx,mu*2));
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
