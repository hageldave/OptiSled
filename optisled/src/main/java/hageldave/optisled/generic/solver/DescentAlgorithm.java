package hageldave.optisled.generic.solver;

import hageldave.optisled.generic.problem.ScalarFN;
import hageldave.optisled.generic.problem.VectorFN;
import hageldave.optisled.history.DescentLog;

public interface DescentAlgorithm<M> {

	/**
	 * finds argmin of a function by performing gradient descent
	 * @param f function to be minimized
	 * @param df gradient of the function
	 * @param initialGuess initialization (guess of minimum location)
	 * @param log (optional, can be null) log object for recording the optimization trajectory
	 * @return location of minimum
	 */
	public M arg_min(ScalarFN<M> f, VectorFN<M> df, M initialGuess, DescentLog log);
	
	/**
	 * finds argmin by performing gradient descent
	 * @param f function to be minimized
	 * @param df gradient of the function
	 * @param initialGuess initialization (guess of minimum location)
	 * @return location of minimum
	 */
	public default M arg_min(ScalarFN<M> f, VectorFN<M> df, M initialGuess){
		return this.arg_min(f,df,initialGuess,null);
	}
	
	/**
	 * @return the hyperparameters used by this gradient descent implementation
	 */
	public Hyperparams getHyperparams();
	
	/**
	 * @param params the hyperparameters used by this gradient descent implementation
	 */
	public void setHyperparams(Hyperparams params);
	
	/**
	 * @return the loss (value of objective function) when descent terminated and arg_min returned.
	 */
	public double getLoss();
	
}
