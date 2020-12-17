package hageldave.optisled.solver;

import org.ejml.simple.SimpleMatrix;

public class TrajectoryInfo implements Cloneable {
	public SimpleMatrix x;
	public double fx;
	public SimpleMatrix gx;
	public SimpleMatrix lambda;
	public double loss;
	public double mu;
	public boolean isGradientDescent;
	
	public TrajectoryInfo copy() {
		try {
			return (TrajectoryInfo)this.clone();
		} catch (CloneNotSupportedException e) {
			// cannot happen since we are cloneable
			return null;
		}
	}
}
