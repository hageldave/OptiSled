package hageldave.optisled;

import static org.junit.jupiter.api.Assertions.assertEquals;

import hageldave.optisled.generic.problem.OptimizationProblem;
import hageldave.optisled.generic.problem.OptimizationProblemBuilder;
import hageldave.optisled.generic.solver.AugmentedLagrangian;
import hageldave.optisled.generic.solver.LogBarrier;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import hageldave.optisled.ejml.MatCalcEJML;
import hageldave.optisled.generic.numerics.MatCalc;
import hageldave.optisled.generic.numerics.NumericGradient;
import hageldave.optisled.generic.problem.ScalarFN;
import hageldave.optisled.generic.solver.GradientDescent;

import java.lang.reflect.InvocationTargetException;
import java.util.Arrays;

public class SanityTest {

	@ParameterizedTest
	@ValueSource(classes = {MatCalcEJML.class})
	public <M> void testQuadraticProblem(Class<MatCalc<M>> implementation)
			throws InstantiationException, IllegalAccessException, NoSuchMethodException, InvocationTargetException
	{
		MatCalc<M> mc = implementation.getDeclaredConstructor().newInstance();
		
		M transform = mc.matOf(2, 2.0,0.2,0.4,1.0);
		M translate = mc.vecOf(3.2, -5);
		
		ScalarFN<M> fx = new ScalarFN<M>() {
			/* f(x) = (x-a)^T B (x-a) */
			@Override
			public double evaluate(M x) {
				x = mc.sub(x,translate);
				return mc.inner(x, mc.matmul(transform, x));
			}
		};
		NumericGradient<M> dfx = new NumericGradient<>(mc, fx);
		
		GradientDescent<M> gd = new GradientDescent<>(mc);
		M init = mc.rand(2, 1);
		M argmin = gd.arg_min(fx, dfx, init, null);
		
		M diff = mc.sub(argmin, translate);
		assertEquals(0.0, mc.get(diff, 0), 1e-4);
		assertEquals(0.0, mc.get(diff, 1), 1e-4);
	}

	@ParameterizedTest
	@ValueSource(classes = {MatCalcEJML.class})
	public <M> void testQuadraticConstrainedProblemAug(Class<MatCalc<M>> implementation)
			throws InstantiationException, IllegalAccessException, NoSuchMethodException, InvocationTargetException
	{
		MatCalc<M> mc = implementation.getDeclaredConstructor().newInstance();

		M transform = mc.eye(2, 0.2);
		M translate = mc.vecOf(3.2, -5);

		ScalarFN<M> fx = new ScalarFN<M>() {
			/* f(x) = (x-a)^T B (x-a) */
			@Override
			public double evaluate(M x) {
				x = mc.sub(x,translate);
				return mc.inner(x, mc.matmul(transform, x));
			}
		};
		// boundary constraint allowing only x-2 < 0 == x < 2
		ScalarFN.ScalarFNWithGradient<M> boundary = ScalarFN.linear(mc, mc.vecOf(1.0, 0.0), -2.0);

		OptimizationProblem<M> problem = OptimizationProblemBuilder.instance(2, mc)
				.setObjective(fx, null)
				.addIneqConstraint(boundary, null)
				.build();

		AugmentedLagrangian<M> augl = new AugmentedLagrangian<>(mc);
		M argmin = augl.arg_min(problem, mc.vecOf(0, 0));

		assertEquals(0.0, boundary.evaluate(argmin), 1e-4, Arrays.toString(mc.toArray(argmin)));
		assertEquals(boundary.evaluate(translate), mc.dist(argmin,translate), 1e-4, Arrays.toString(mc.toArray(argmin)));
	}

	@ParameterizedTest
	@ValueSource(classes = {MatCalcEJML.class})
	public <M> void testQuadraticConstrainedProblemLB(Class<MatCalc<M>> implementation)
			throws InstantiationException, IllegalAccessException, NoSuchMethodException, InvocationTargetException
	{
		MatCalc<M> mc = implementation.getDeclaredConstructor().newInstance();

		M transform = mc.eye(2, 0.2);
		M translate = mc.vecOf(3.2, -5);

		ScalarFN<M> fx = new ScalarFN<M>() {
			/* f(x) = (x-a)^T B (x-a) */
			@Override
			public double evaluate(M x) {
				x = mc.sub(x,translate);
				return mc.inner(x, mc.matmul(transform, x));
			}
		};
		// boundary constraint allowing only x-2 < 0 == x < 2
		ScalarFN.ScalarFNWithGradient<M> boundary = ScalarFN.linear(mc, mc.vecOf(1.0, 0.0), -2.0);

		OptimizationProblem<M> problem = OptimizationProblemBuilder.instance(2, mc)
				.setObjective(fx, null)
				.addIneqConstraint(boundary, null)
				.build();

		LogBarrier<M> lb = new LogBarrier<>(mc);
		M argmin = lb.arg_min(problem, mc.vecOf(0, 0));

		assertEquals(0.0, boundary.evaluate(argmin), 1e-5, Arrays.toString(mc.toArray(argmin)));
		assertEquals(boundary.evaluate(translate), mc.dist(argmin,translate), 1e-5, Arrays.toString(mc.toArray(argmin)));
	}
	
}
