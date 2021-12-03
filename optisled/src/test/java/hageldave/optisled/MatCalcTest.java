package hageldave.optisled;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import hageldave.optisled.ejml.MatCalcEJML;
import hageldave.optisled.generic.numerics.MatCalc;

public class MatCalcTest {

	@ParameterizedTest
	@ValueSource(classes = {MatCalcEJML.class})
	public <M> void testMatCalc(Class<MatCalc<M>> implementation) throws InstantiationException, IllegalAccessException {
		MatCalc<M> mc = implementation.newInstance();
		// a = [0,0]
		M a = mc.zeros(2);
		assertEquals(1, mc.numCols(a));
		assertEquals(2, mc.numRows(a));
		assertEquals(0.0, mc.get(a, 0));
		assertEquals(0.0, mc.get(a, 1));
		// a = [1,0]
		assertTrue(a == mc.set_inp(a, 0, 1.0));
		assertEquals(1.0, mc.get(a, 0,0));
		assertEquals(0.0, mc.get(a, 1,0));
		// b = [1,0]
		M b = mc.copy(a);
		assertEquals(1.0, mc.inner(a, b));
		assertEquals(0.0, mc.inner(a, mc.vecOf(0.0, 1.0)));
		// b = [2,0]
		assertTrue(b == mc.scale_inp(b, 2.0));
		assertEquals(2.0, mc.inner(a, b));
		assertEquals(4.0, mc.norm2(b));
		assertEquals(2.0, mc.norm(b));
		// C = [1,0 ; 0,1]
		M C = mc.zeros(2, 2);
		mc.set_inp(C, 0,0, 1.0);mc.set_inp(C, 1,1, 1.0);
		assertEquals(1.0, mc.get(mc.matmul(C,a),0));
		assertEquals(0.0, mc.get(mc.matmul(C,a),1));
		// D = [.5,0 ; 0,.5]
		M D = mc.scale(C, 0.5);
		assertEquals(1.0, mc.get(mc.matmul(D,b),0));
		assertEquals(0.0, mc.get(mc.matmul(D,b),1));
		assertEquals(4.0, mc.norm(mc.add(b, b)));
		assertEquals(1.0, mc.norm2(mc.sub(a, b)));
		// e = a = [1,0]
		M e = mc.matOf(2, mc.toArray(a));
		// F = D = [.5,0 ; 0,.5]
		M F = mc.matOf(mc.toArray2D(D));
		assertEquals(0.5, mc.norm(mc.matmul(F, e)));
		// g = a^T
		M g = mc.trp(a);
		assertEquals(mc.numCols(a), mc.numRows(g));
		assertEquals(mc.numRows(a), mc.numCols(g));
	}
	
}
