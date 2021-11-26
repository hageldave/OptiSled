package hageldave.optisled.generic.util;

import java.util.function.BinaryOperator;
import java.util.function.UnaryOperator;

public class Util {

	public static <T> T[] applyElementWise(T[] array, UnaryOperator<T> elementFn){
		for(int i=0; i<array.length; i++){
			array[i] = elementFn.apply(array[i]);
		}
		return array;
	}
	
	public static <T> T[] applyElementWise(T[] op1, T[] op2, BinaryOperator<T> elementFn){
		for(int i=0; i<op1.length; i++){
			op1[i] = elementFn.apply(op1[i], op2[i]);
		}
		return op1;
	}
	
}
