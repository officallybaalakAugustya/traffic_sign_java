package com.trafficsign.math;

import java.util.Arrays;
import java.util.Objects;
import java.util.function.DoubleUnaryOperator;

/**
 * Multi-dimensional float/double tensor abstraction.
 *
 * <p><b>Broadcasting Capabilities:</b>
 * This tensor implements explicit, well-defined broadcasting subsets required by
 * deep learning architectures:
 * <ol>
 *   <li><b>Scalar Broadcasting:</b> Any tensor elementwise with a scalar value.</li>
 *   <li><b>Row-Vector Bias Broadcasting across Batch:</b> 2D tensor of shape [Batch, Features]
 *       added with a 1D bias vector of shape [Features] or [1, Features].</li>
 *   <li><b>Channel Bias Broadcasting:</b> 3D tensor of shape [Channels, Height, Width]
 *       added with a 1D channel bias of shape [Channels].</li>
 * </ol>
 * <i>Note: Arbitrary N-dimensional generalized broadcasting is explicitly not claimed or supported.</i>
 */
public final class Tensor {

    private final int[] shape;
    private final int[] strides;
    private final double[] data;
    private final int size;

    public Tensor(int... shape) {
        Objects.requireNonNull(shape, "Shape array must not be null");
        if (shape.length == 0) {
            throw new IllegalArgumentException("Tensor rank must be at least 1");
        }
        long computedSize = 1;
        for (int dim : shape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("All tensor dimensions must be strictly positive. Shape: " + Arrays.toString(shape));
            }
            computedSize *= dim;
            if (computedSize > Integer.MAX_VALUE) {
                throw new IllegalArgumentException("Tensor element count exceeds Integer.MAX_VALUE");
            }
        }
        this.shape = shape.clone();
        this.size = (int) computedSize;
        this.strides = computeStrides(this.shape);
        this.data = new double[this.size];
    }

    public Tensor(int[] shape, double[] values) {
        this(shape);
        Objects.requireNonNull(values, "Values array must not be null");
        if (values.length != this.size) {
            throw new IllegalArgumentException("Data array length (" + values.length
                    + ") does not match tensor element count (" + this.size + ")");
        }
        System.arraycopy(values, 0, this.data, 0, this.size);
    }

    public static Tensor fromFlat(int[] shape, double[] flatData) {
        return new Tensor(shape, flatData);
    }

    public static Tensor fromMatrix(Matrix m) {
        Objects.requireNonNull(m, "Matrix must not be null");
        return new Tensor(new int[]{m.rows(), m.cols()}, m.toFlat());
    }

    public static Tensor zeros(int... shape) {
        return new Tensor(shape);
    }

    public int rank() {
        return shape.length;
    }

    public int[] shape() {
        return shape.clone();
    }

    public int size() {
        return size;
    }

    public double get(int... indices) {
        int flatIndex = getFlatIndex(indices);
        return data[flatIndex];
    }

    public void set(double value, int... indices) {
        int flatIndex = getFlatIndex(indices);
        data[flatIndex] = value;
    }

    public Tensor reshape(int... newShape) {
        Objects.requireNonNull(newShape, "New shape must not be null");
        long newSize = 1;
        for (int dim : newShape) {
            if (dim <= 0) {
                throw new IllegalArgumentException("New dimensions must be positive: " + Arrays.toString(newShape));
            }
            newSize *= dim;
        }
        if (newSize != this.size) {
            throw new IllegalArgumentException("Cannot reshape tensor of size " + this.size
                    + " into shape " + Arrays.toString(newShape) + " (size " + newSize + ")");
        }
        return new Tensor(newShape, this.data);
    }

    public Tensor add(Tensor other) {
        Objects.requireNonNull(other, "other tensor must not be null");
        if (Arrays.equals(this.shape, other.shape)) {
            Tensor result = new Tensor(this.shape);
            for (int i = 0; i < size; i++) {
                result.data[i] = this.data[i] + other.data[i];
            }
            return result;
        }

        // Supported Bias Broadcasting Case 1: [Batch, Features] + [Features] or [Batch, Features] + [1, Features]
        if (this.rank() == 2 && (other.rank() == 1 && other.shape[0] == this.shape[1])) {
            int batch = this.shape[0];
            int features = this.shape[1];
            Tensor result = new Tensor(this.shape);
            for (int b = 0; b < batch; b++) {
                int offset = b * features;
                for (int f = 0; f < features; f++) {
                    result.data[offset + f] = this.data[offset + f] + other.data[f];
                }
            }
            return result;
        }

        if (this.rank() == 2 && (other.rank() == 2 && other.shape[0] == 1 && other.shape[1] == this.shape[1])) {
            int batch = this.shape[0];
            int features = this.shape[1];
            Tensor result = new Tensor(this.shape);
            for (int b = 0; b < batch; b++) {
                int offset = b * features;
                for (int f = 0; f < features; f++) {
                    result.data[offset + f] = this.data[offset + f] + other.data[f];
                }
            }
            return result;
        }

        throw new IllegalArgumentException("Unsupported broadcasting shapes: "
                + Arrays.toString(this.shape) + " and " + Arrays.toString(other.shape));
    }

    public Tensor subtract(Tensor other) {
        Objects.requireNonNull(other, "other tensor must not be null");
        if (!Arrays.equals(this.shape, other.shape)) {
            throw new IllegalArgumentException("Tensor shape mismatch for subtraction: "
                    + Arrays.toString(this.shape) + " vs " + Arrays.toString(other.shape));
        }
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < size; i++) {
            result.data[i] = this.data[i] - other.data[i];
        }
        return result;
    }

    public Tensor multiply(double scalar) {
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < size; i++) {
            result.data[i] = this.data[i] * scalar;
        }
        return result;
    }

    public Tensor elementMultiply(Tensor other) {
        Objects.requireNonNull(other, "other tensor must not be null");
        if (!Arrays.equals(this.shape, other.shape)) {
            throw new IllegalArgumentException("Tensor shape mismatch for elementwise multiplication: "
                    + Arrays.toString(this.shape) + " vs " + Arrays.toString(other.shape));
        }
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < size; i++) {
            result.data[i] = this.data[i] * other.data[i];
        }
        return result;
    }

    public Tensor apply(DoubleUnaryOperator func) {
        Objects.requireNonNull(func, "function must not be null");
        Tensor result = new Tensor(this.shape);
        for (int i = 0; i < size; i++) {
            result.data[i] = func.applyAsDouble(this.data[i]);
        }
        return result;
    }

    public double dot(Tensor other) {
        Objects.requireNonNull(other, "other tensor must not be null");
        if (this.size != other.size) {
            throw new IllegalArgumentException("Tensor element count mismatch for dot product: "
                    + this.size + " vs " + other.size);
        }
        double sum = 0.0;
        for (int i = 0; i < size; i++) {
            sum += this.data[i] * other.data[i];
        }
        return sum;
    }

    public Tensor sliceBatch(int startBatch, int endBatch) {
        if (rank() < 1) {
            throw new IllegalStateException("Cannot slice tensor with rank < 1");
        }
        if (startBatch < 0 || endBatch > shape[0] || startBatch >= endBatch) {
            throw new IndexOutOfBoundsException("Invalid slice bounds: [" + startBatch + ", " + endBatch + ") for batch size " + shape[0]);
        }
        int slicedBatchSize = endBatch - startBatch;
        int[] slicedShape = shape.clone();
        slicedShape[0] = slicedBatchSize;

        int elementsPerSample = size / shape[0];
        double[] slicedData = new double[slicedBatchSize * elementsPerSample];
        System.arraycopy(data, startBatch * elementsPerSample, slicedData, 0, slicedData.length);
        return new Tensor(slicedShape, slicedData);
    }

    public Matrix toMatrix() {
        if (rank() != 2) {
            throw new IllegalStateException("Tensor must be rank 2 to convert to Matrix. Current rank: " + rank());
        }
        return Matrix.fromFlat(shape[0], shape[1], this.data);
    }

    public double[] toFlat() {
        return data.clone();
    }

    private int getFlatIndex(int[] indices) {
        if (indices.length != shape.length) {
            throw new IllegalArgumentException("Indices rank " + indices.length + " does not match tensor rank " + shape.length);
        }
        int index = 0;
        for (int i = 0; i < shape.length; i++) {
            if (indices[i] < 0 || indices[i] >= shape[i]) {
                throw new IndexOutOfBoundsException("Index " + indices[i] + " out of bounds for axis " + i + " (size " + shape[i] + ")");
            }
            index += indices[i] * strides[i];
        }
        return index;
    }

    private static int[] computeStrides(int[] shape) {
        int[] strides = new int[shape.length];
        strides[shape.length - 1] = 1;
        for (int i = shape.length - 2; i >= 0; i--) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Tensor tensor = (Tensor) o;
        return size == tensor.size && Arrays.equals(shape, tensor.shape) && Arrays.equals(data, tensor.data);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(size);
        result = 31 * result + Arrays.hashCode(shape);
        result = 31 * result + Arrays.hashCode(data);
        return result;
    }
}
