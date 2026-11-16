package com.trafficsign.neural.activation;

import com.trafficsign.neural.neuron.SigmoidNeuron;

/**
 * Sigmoid logistic activation function.
 */
public class SigmoidActivation implements ActivationFunction {

    @Override
    public double activate(double z) {
        return SigmoidNeuron.sigmoid(z);
    }

    @Override
    public double derivative(double z, double activatedValue) {
        return activatedValue * (1.0 - activatedValue);
    }
}
