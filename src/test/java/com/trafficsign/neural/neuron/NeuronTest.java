package com.trafficsign.neural.neuron;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

class NeuronTest {

    @ParameterizedTest
    @CsvSource({
            "0, 0, 0",
            "0, 1, 0",
            "1, 0, 0",
            "1, 1, 1"
    })
    @DisplayName("McCulloch-Pitts AND gate verification")
    void shouldVerifyMcCullochPittsAndGate(int x1, int x2, int expected) {
        McCullochPittsNeuron andNeuron = McCullochPittsNeuron.andGate();
        assertThat(andNeuron.activate(new int[]{x1, x2})).isEqualTo(expected);
    }

    @ParameterizedTest
    @CsvSource({
            "0, 0, 0",
            "0, 1, 1",
            "1, 0, 1",
            "1, 1, 1"
    })
    @DisplayName("McCulloch-Pitts OR gate verification")
    void shouldVerifyMcCullochPittsOrGate(int x1, int x2, int expected) {
        McCullochPittsNeuron orNeuron = McCullochPittsNeuron.orGate();
        assertThat(orNeuron.activate(new int[]{x1, x2})).isEqualTo(expected);
    }

    @Test
    @DisplayName("McCulloch-Pitts NOT gate and non-binary input rejection")
    void shouldVerifyMcCullochPittsNotGateAndInputRejection() {
        McCullochPittsNeuron notNeuron = McCullochPittsNeuron.notGate();
        assertThat(notNeuron.activate(new int[]{0})).isEqualTo(1);
        assertThat(notNeuron.activate(new int[]{1})).isEqualTo(0);

        assertThatThrownBy(() -> notNeuron.activate(new int[]{2}))
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("strictly binary inputs");
    }

    @Test
    @DisplayName("Sigmoid neuron evaluates smooth output and accurate derivative at z=0")
    void shouldVerifySigmoidNeuronAtZero() {
        SigmoidNeuron neuron = new SigmoidNeuron(new double[]{1.0, -1.0}, 0.0);
        double output = neuron.activate(new double[]{2.0, 2.0}); // z = 0.0
        assertThat(output).isEqualTo(0.5);

        double derivative = SigmoidNeuron.sigmoidDerivative(0.0);
        assertThat(derivative).isEqualTo(0.25);
    }

    @Test
    @DisplayName("Sigmoid neuron numerically stable under extreme positive and negative inputs")
    void shouldBeNumericallyStableUnderExtremes() {
        // Large positive input (+500.0) -> asymptotically 1.0, not NaN or overflow
        double extremePos = SigmoidNeuron.sigmoid(500.0);
        assertThat(extremePos).isEqualTo(1.0);

        // Large negative input (-500.0) -> asymptotically 0.0, not NaN or overflow
        double extremeNeg = SigmoidNeuron.sigmoid(-500.0);
        assertThat(extremeNeg).isEqualTo(0.0);

        // Derivatives at extremes should approach 0.0
        assertThat(SigmoidNeuron.sigmoidDerivative(500.0)).isEqualTo(0.0);
        assertThat(SigmoidNeuron.sigmoidDerivative(-500.0)).isCloseTo(0.0, org.assertj.core.data.Offset.offset(1e-15));

        assertThatThrownBy(() -> SigmoidNeuron.sigmoid(Double.NaN))
                .isInstanceOf(IllegalArgumentException.class);
    }
}
