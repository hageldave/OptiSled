package hageldave.optisled.solver;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

import org.ejml.simple.SimpleMatrix;

import hageldave.optisled.ejml.MatUtil;
import hageldave.optisled.problem.OptimizationProblem;
import hageldave.optisled.problem.ScalarFN.ScalarFNWithGradient;
import hageldave.optisled.problem.VectorFN;

public class AugmentedLagrangian {

	public int maxNumIterations = 80;
	public double muIncr = 1.01;
	public double initialStepsize = 1.0;
	
	public SimpleMatrix arg_min(OptimizationProblem p, SimpleMatrix initialGuess) {
		return arg_min(p, initialGuess, null);
	}
	
	public SimpleMatrix arg_min(OptimizationProblem p, SimpleMatrix initialGuess, List<TrajectoryInfo> trace) {
		double[] lambda = new double[p.numConstraints()];
		double mu = 1;
		SimpleMatrix x = initialGuess.copy();
		int numIterations = 0;
		do {
			TrajectoryInfo info = new TrajectoryInfo();
			if(Objects.nonNull(trace)){
				info = new TrajectoryInfo();
				TrajectoryInfo info_ = info;
				info.x = x;
				info.fx = p.f().evaluate(x);
				info.gx = MatUtil.vectorOf(
					Arrays.stream(p.g()).mapToDouble(g->g.evaluate(info_.x)).toArray()
				);
				info.lambda = MatUtil.vectorOf(lambda.clone());
				info.loss = augLagrangian(p, lambda, mu).evaluate(x);
				info.mu = mu;
//				trace.add(info);
			}
			
			ScalarFNWithGradient f = augLagrangian(p, lambda, mu);
			GradientDescent gd = new GradientDescent();
			gd.initialStepSize = initialStepsize;
			x = gd.arg_min(f, f.gradient(), x, info);
			
			if(Objects.nonNull(trace)){
				// fill in missing info information for gradient descent
				for(TrajectoryInfo gdinfo : gd.trajecInfo) {
					TrajectoryInfo gdinfo_ = gdinfo;
					gdinfo.fx = p.f().evaluate(gdinfo.x);
					gdinfo.gx = MatUtil.vectorOf(
							Arrays.stream(p.g()).mapToDouble(g->g.evaluate(gdinfo_.x)).toArray()
					);
				}
				trace.addAll(gd.trajecInfo);
				
				info = new TrajectoryInfo();
				info.x = x;
				info.fx = p.f().evaluate(x);
				TrajectoryInfo info_ = info;
				info.gx = MatUtil.vectorOf(
					Arrays.stream(p.g()).mapToDouble(g->g.evaluate(info_.x)).toArray()
				);
				info.lambda = MatUtil.vectorOf(lambda.clone());
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
	
	
	public static ScalarFNWithGradient augLagrangian(OptimizationProblem p, double[] lambda, double mu){
		return new ScalarFNWithGradient() {
			
			@Override
			public double evaluate(SimpleMatrix x) {
				double result = p.f().evaluate(x);
				for(int i=0; i<p.numConstraints(); i++){
					double gx = p.g()[i].evaluate(x);
					// += lambda_i * g(x) + [g(x) > 0] mu * g(x)^2
					result += lambda[i]*gx;
					result += Math.max(0, gx)*gx*mu;
				}
				return result;
			}
			
			VectorFN grad = new VectorFN() {
				
				@Override
				public SimpleMatrix evaluate(SimpleMatrix x) {
					SimpleMatrix result = p.df().evaluate(x);
					for(int i=0; i<p.numConstraints(); i++){
						double gx = p.g()[i].evaluate(x);
						SimpleMatrix dgx = p.dg()[i].evaluate(x);
						// += lambda_i * dg(x) + [g(x) > 0] * 2mu * dg(x)
						result = result.plus(dgx.scale(lambda[i]));
						if(gx > 0){
							result = result.plus(dgx.scale(mu*2));
						}
					}
					return result;
				}
			};
			
			@Override
			public VectorFN gradient() {
				return grad;
			}
		};
	}
	
}
