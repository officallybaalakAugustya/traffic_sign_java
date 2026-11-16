package com.trafficsign.neural.activation;

/**
 * Interface for scalar activation functions used in hidden layers.
 */
public interface ActivationFunction {

    double activate(double z);

    double derivative(double z, double activatedValue);
}
