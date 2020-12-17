package hageldave.optisled.solver;

import org.ejml.simple.SimpleMatrix;

import hageldave.optisled.problem.OptimizationProblem;
import hageldave.optisled.problem.ScalarFN.ScalarFNWithGradient;
import hageldave.optisled.problem.VectorFN;
import hageldave.optisled.util.MatUtil;

import static hageldave.optisled.util.MatUtil.*;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

public class LogBarrier {

	public double initialMu = 8.0;
	public double muDecr = 0.92;
	public int maxNumIterations = 135;
	
	public SimpleMatrix arg_min(OptimizationProblem p, SimpleMatrix initialGuess) {
		return arg_min(p, initialGuess, null);
	}
	
	public SimpleMatrix arg_min(OptimizationProblem p, SimpleMatrix initialGuess, List<TrajectoryInfo> trace) {
		double mu = initialMu;
		SimpleMatrix x = initialGuess.copy();
		if(Objects.nonNull(trace)){
			TrajectoryInfo info = new TrajectoryInfo();
			info.x = x;
			info.fx = p.f().evaluate(x);
			info.gx = MatUtil.vectorOf(
				Arrays.stream(p.g()).mapToDouble(g->g.evaluate(info.x)).toArray()
			);
			double currmu = mu;
			info.lambda = MatUtil.vectorOf(
				MatUtil.streamValues(info.gx).map(gx->-currmu/gx).toArray()
			);
			info.loss = logBarrFN(p, mu).evaluate(x);
			info.mu = mu;
			trace.add(info);
		}
		int numIterations = 0;
		do {
			ScalarFNWithGradient f = logBarrFN(p, mu);
			GradientDescent gd = new GradientDescent();
			x = gd.arg_min(f, f.gradient(), x, null);
			if(Objects.nonNull(trace)){
				TrajectoryInfo info = new TrajectoryInfo();
				info.x = x;
				info.fx = p.f().evaluate(x);
				info.gx = MatUtil.vectorOf(
					Arrays.stream(p.g()).mapToDouble(g->g.evaluate(info.x)).toArray()
				);
				double currmu = mu;
				info.lambda = MatUtil.vectorOf(
					MatUtil.streamValues(info.gx).map(gx->-currmu/gx).toArray()
				);
				info.loss = f.evaluate(x);
				info.mu = mu;
				trace.add(info);
			}
			mu *= muDecr;
		} while(++numIterations < maxNumIterations);
		return x;
	}
	
	public static ScalarFNWithGradient logBarrFN(OptimizationProblem p, double mu){
		return new ScalarFNWithGradient() {
			
			@Override
			public double evaluate(SimpleMatrix x) {
				double result = p.f().evaluate(x);
				for(int i=0; i<p.numConstraints(); i++){
					// result -= mu*log( max(0,-gi(x)) )
					result -= mu*Math.log( Math.max(0, -p.g()[i].evaluate(x)) );
				}
				return result;
			}
			
			VectorFN grad = new VectorFN() {
				@Override
				public SimpleMatrix evaluate(SimpleMatrix x) {
					SimpleMatrix result = normalize(p.df().evaluate(x));
					for(int i=0; i<p.numConstraints(); i++){
						double gx = p.g()[i].evaluate(x);
						SimpleMatrix dgx = normalize(p.dg()[i].evaluate(x));
						if(gx < 0){
							// result -= (mu/gx) * dgx
							result = result.minus(dgx.scale(mu/gx));
						} else {
							// handling of gradient for nondefined negative logarithms in infeasible regions
							result = result.plus(dgx.scale(1+gx));
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
