package hageldave.optisled;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import hageldave.optisled.ejml.MatCalcEJML;
import hageldave.optisled.generic.numerics.MatCalc;
import hageldave.optisled.generic.numerics.NumericGradient;
import hageldave.optisled.generic.problem.ScalarFN;
import hageldave.optisled.generic.solver.GradientDescent;

public class SanityTest {
	
	

	@ParameterizedTest
	@ValueSource(classes = {MatCalcEJML.class})
	public <M> void testQuadraticProblem(Class<MatCalc<M>> implementation) throws InstantiationException, IllegalAccessException {
		MatCalc<M> mc = implementation.newInstance();
		
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
		NumericGradient<M> dfx = new NumericGradient<>(fx, mc);
		
		GradientDescent<M> gd = new GradientDescent<>(mc);
		M init = mc.rand(2, 1);
		M argmin = gd.arg_min(fx, dfx, init, null);
		
		M diff = mc.sub(argmin, translate);
		assertEquals(0.0, mc.get(diff, 0), 1e-4);
		assertEquals(0.0, mc.get(diff, 1), 1e-4);
	}
	
}
