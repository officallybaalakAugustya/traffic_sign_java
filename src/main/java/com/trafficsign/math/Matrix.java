package com.trafficsign.math;

import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.function.DoubleUnaryOperator;

/**
 * Pure Java 2D Matrix implementation with strict dimension validation,
 * defensive copying of mutable arrays, and linear algebra operations.
 */
public final class Matrix {

    private final int rows;
    private final int cols;
    private final double[][] data;

    public Matrix(int rows, int cols) {
        if (rows <= 0 || cols <= 0) {
            throw new IllegalArgumentException("Matrix dimensions must be strictly positive. Received: " + rows + "x" + cols);
        }
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(double[][] values) {
        Objects.requireNonNull(values, "Initial values array must not be null");
        if (values.length == 0 || values[0].length == 0) {
            throw new IllegalArgumentException("Matrix cannot be initialized from empty 2D array");
        }
        this.rows = values.length;
        this.cols = values[0].length;
        this.data = new double[rows][cols];

        for (int r = 0; r < rows; r++) {
            if (values[r] == null || values[r].length != cols) {
                throw new IllegalArgumentException("Ragged 2D array: row " + r + " has length "
                        + (values[r] == null ? "null" : values[r].length) + ", expected " + cols);
            }
            System.arraycopy(values[r], 0, this.data[r], 0, cols);
        }
    }

    public static Matrix zeros(int rows, int cols) {
        return new Matrix(rows, cols);
    }

    public static Matrix fromFlat(int rows, int cols, double[] flat) {
        Objects.requireNonNull(flat, "flat array must not be null");
        if (flat.length != rows * cols) {
            throw new IllegalArgumentException("Flat array length " + flat.length + " does not match matrix size " + (rows * cols));
        }
        Matrix m = new Matrix(rows, cols);
        int idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                m.data[r][c] = flat[idx++];
            }
        }
        return m;
    }

    public static Matrix random(int rows, int cols, Random rng, double min, double max) {
        Objects.requireNonNull(rng, "Random number generator must not be null");
        Matrix m = new Matrix(rows, cols);
        double range = max - min;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                m.data[r][c] = min + rng.nextDouble() * range;
            }
        }
        return m;
    }

    public int rows() {
        return rows;
    }

    public int cols() {
        return cols;
    }

    public double get(int r, int c) {
        checkBounds(r, c);
        return data[r][c];
    }

    public void set(int r, int c, double val) {
        checkBounds(r, c);
        data[r][c] = val;
    }

    public Matrix add(Matrix other) {
        Objects.requireNonNull(other, "other matrix must not be null");
        assertSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.data[r][c] = this.data[r][c] + other.data[r][c];
            }
        }
        return result;
    }

    public Matrix subtract(Matrix other) {
        Objects.requireNonNull(other, "other matrix must not be null");
        assertSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.data[r][c] = this.data[r][c] - other.data[r][c];
            }
        }
        return result;
    }

    public Matrix multiply(double scalar) {
        Matrix result = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.data[r][c] = this.data[r][c] * scalar;
            }
        }
        return result;
    }

    public Matrix multiply(Matrix other) {
        Objects.requireNonNull(other, "other matrix must not be null");
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Incompatible dimensions for matrix multiplication: ("
                    + this.rows + "x" + this.cols + ") * (" + other.rows + "x" + other.cols + ")");
        }
        Matrix result = new Matrix(this.rows, other.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int k = 0; k < this.cols; k++) {
                double a = this.data[i][k];
                for (int j = 0; j < other.cols; j++) {
                    result.data[i][j] += a * other.data[k][j];
                }
            }
        }
        return result;
    }

    public Matrix elementMultiply(Matrix other) {
        Objects.requireNonNull(other, "other matrix must not be null");
        assertSameShape(other);
        Matrix result = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.data[r][c] = this.data[r][c] * other.data[r][c];
            }
        }
        return result;
    }

    public Matrix transpose() {
        Matrix result = new Matrix(cols, rows);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.data[c][r] = this.data[r][c];
            }
        }
        return result;
    }

    public static double dot(double[] a, double[] b) {
        Objects.requireNonNull(a, "vector a must not be null");
        Objects.requireNonNull(b, "vector b must not be null");
        if (a.length != b.length) {
            throw new IllegalArgumentException("Vectors must have identical length for dot product: " + a.length + " vs " + b.length);
        }
        double sum = 0.0;
        for (int i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }
        return sum;
    }

    public Matrix apply(DoubleUnaryOperator func) {
        Objects.requireNonNull(func, "mapping function must not be null");
        Matrix result = new Matrix(rows, cols);
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                result.data[r][c] = func.applyAsDouble(this.data[r][c]);
            }
        }
        return result;
    }

    public double sum() {
        double total = 0.0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                total += data[r][c];
            }
        }
        return total;
    }

    public double[] sumCols() {
        double[] colSums = new double[cols];
        for (int c = 0; c < cols; c++) {
            double sum = 0.0;
            for (int r = 0; r < rows; r++) {
                sum += data[r][c];
            }
            colSums[c] = sum;
        }
        return colSums;
    }

    public double[] toFlat() {
        double[] flat = new double[rows * cols];
        int idx = 0;
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                flat[idx++] = data[r][c];
            }
        }
        return flat;
    }

    public double[][] toArray() {
        double[][] copy = new double[rows][cols];
        for (int r = 0; r < rows; r++) {
            System.arraycopy(data[r], 0, copy[r], 0, cols);
        }
        return copy;
    }

    private void checkBounds(int r, int c) {
        if (r < 0 || r >= rows || c < 0 || c >= cols) {
            throw new IndexOutOfBoundsException("Matrix index (" + r + ", " + c + ") out of bounds for size (" + rows + "x" + cols + ")");
        }
    }

    private void assertSameShape(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix shape mismatch: (" + this.rows + "x" + this.cols
                    + ") vs (" + other.rows + "x" + other.cols + ")");
        }
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Matrix matrix = (Matrix) o;
        if (rows != matrix.rows || cols != matrix.cols) return false;
        for (int r = 0; r < rows; r++) {
            if (!Arrays.equals(data[r], matrix.data[r])) return false;
        }
        return true;
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(rows, cols);
        for (int r = 0; r < rows; r++) {
            result = 31 * result + Arrays.hashCode(data[r]);
        }
        return result;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("Matrix (").append(rows).append("x").append(cols).append("):\n");
        for (int r = 0; r < Math.min(rows, 10); r++) {
            sb.append("  [");
            for (int c = 0; c < Math.min(cols, 10); c++) {
                sb.append(String.format("%8.4f", data[r][c]));
                if (c < cols - 1) sb.append(", ");
            }
            if (cols > 10) sb.append(", ...");
            sb.append("]\n");
        }
        if (rows > 10) sb.append("  ...\n");
        return sb.toString();
    }
}
