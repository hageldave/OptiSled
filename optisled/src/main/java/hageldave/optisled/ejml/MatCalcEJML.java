package hageldave.optisled.ejml;

import org.ejml.data.DMatrixRMaj;
import org.ejml.data.Matrix;
import org.ejml.data.MatrixType;
import org.ejml.simple.SimpleMatrix;
import org.ejml.simple.SimpleOperations;
import org.ejml.simple.ops.SimpleOperations_CDRM;
import org.ejml.simple.ops.SimpleOperations_DDRM;
import org.ejml.simple.ops.SimpleOperations_DSCC;
import org.ejml.simple.ops.SimpleOperations_FDRM;
import org.ejml.simple.ops.SimpleOperations_FSCC;
import org.ejml.simple.ops.SimpleOperations_ZDRM;

import hageldave.optisled.generic.numerics.MatCalc;

public class MatCalcEJML implements MatCalc<SimpleMatrix> {
	
	protected static final SimpleOperations<? extends Matrix>[] ops;
	
	static {
		ops = new SimpleOperations<?>[MatrixType.values().length];
		for(MatrixType mt : MatrixType.values())
			ops[mt.ordinal()] = lookupOps(mt);
	}
	
	static SimpleOperations<? extends Matrix> lookupOps( MatrixType type ) {
        switch( type ) {
        case DDRM:
            return new SimpleOperations_DDRM();
        case FDRM:
            return new SimpleOperations_FDRM();
        case ZDRM:
            return new SimpleOperations_ZDRM();
        case CDRM:
            return new SimpleOperations_CDRM();
        case DSCC:
            return new SimpleOperations_DSCC();
        case FSCC:
            return new SimpleOperations_FSCC();
        default: 
        	return null;
        }
    }

	@Override
	public SimpleMatrix vecOf(double... v) {
		return MatUtil.vectorOf(v);
	}
	
	@Override
	public SimpleMatrix matOf(double[][] values) {
		return MatUtil.rowmajorMat(values);
	}
	
	@Override
	public SimpleMatrix matOf(int nRows, double... values) {
		return MatUtil.matrix(nRows, values.length/nRows, values);
	}

	@Override
	public SimpleMatrix zeros(int size) {
		return MatUtil.vector(size);
	}
	
	@Override
	public SimpleMatrix zeros(int rows, int columns) {
		return MatUtil.matrix(rows, columns);
	}

	@Override
	public int numRows(SimpleMatrix m) {
		return m.numRows();
	}

	@Override
	public int numCols(SimpleMatrix m) {
		return m.numCols();
	}

	@Override
	public double inner(SimpleMatrix a, SimpleMatrix b) {
		return a.dot(b);
	}

	@Override
	public SimpleMatrix scale(SimpleMatrix m, double s) {
		return m.scale(s);
	}

	@Override
	public SimpleMatrix scale_inp(SimpleMatrix m, double s) {
		ops[m.getType().ordinal()].scale(m.getMatrix(), s, m.getMatrix());
		return m;
	}

	@Override
	public SimpleMatrix matmul(SimpleMatrix a, SimpleMatrix b) {
		return a.mult(b);
	}
	
	@Override
	public SimpleMatrix trp(SimpleMatrix m) {
		return m.transpose();
	}

	@Override
	public SimpleMatrix add(SimpleMatrix a, SimpleMatrix b) {
		return a.plus(b);
	}

	@Override
	public SimpleMatrix sub(SimpleMatrix a, SimpleMatrix b) {
		return a.minus(b);
	}

	@Override
	public SimpleMatrix set_inp(SimpleMatrix m, int idx, double v) {
		m.set(idx, v);
		return m;
	}

	@Override
	public double get(SimpleMatrix m, int idx) {
		return m.get(idx);
	}

	@Override
	public SimpleMatrix copy(SimpleMatrix m) {
		return m.copy();
	}

	@Override
	public double[] toArray(SimpleMatrix m) {
		if(m.getMatrix() instanceof DMatrixRMaj)
			return m.getDDRM().data;
		double[] data = new double[m.getNumElements()];
		for(int i=0; i<m.getNumElements(); i++) {
			data[i] = m.get(i);
		}
		return data;
	}

	@Override
	public double[][] toArray2D(SimpleMatrix m) {
		double[][] data = new double[m.numRows()][m.numCols()];
		for(int r=0; r<m.numRows(); r++)
			for(int c=0; c<m.numCols(); c++)
				data[r][c] = m.get(r, c);
		return data;
	}

}
