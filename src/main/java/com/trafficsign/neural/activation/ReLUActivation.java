package com.trafficsign.neural.activation;

/**
 * Rectified Linear Unit (ReLU) activation:
 * f(z) = max(0, z)
 * f'(z) = 1 if z > 0 else 0
 */
public class ReLUActivation implements ActivationFunction {

    @Override
    public double activate(double z) {
        return Math.max(0.0, z);
    }

    @Override
    public double derivative(double z, double activatedValue) {
        return z > 0.0 ? 1.0 : 0.0;
    }
}
