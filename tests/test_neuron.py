import pytest
import math
from trafficsign.neural.neuron import McCullochPittsNeuron, SigmoidNeuron


def test_mcculloch_pitts_logic_gates():
    # AND gate
    and_gate = McCullochPittsNeuron.and_gate()
    assert and_gate.activate([0, 0]) == 0
    assert and_gate.activate([0, 1]) == 0
    assert and_gate.activate([1, 0]) == 0
    assert and_gate.activate([1, 1]) == 1

    # OR gate
    or_gate = McCullochPittsNeuron.or_gate()
    assert or_gate.activate([0, 0]) == 0
    assert or_gate.activate([0, 1]) == 1
    assert or_gate.activate([1, 0]) == 1
    assert or_gate.activate([1, 1]) == 1

    # NOT gate
    not_gate = McCullochPittsNeuron.not_gate()
    assert not_gate.activate([0]) == 1
    assert not_gate.activate([1]) == 0

    with pytest.raises(ValueError):
        and_gate.activate([0, 2])


def test_sigmoid_neuron():
    neuron = SigmoidNeuron([0.5, -0.5], bias=0.0)

    # Affine: 0.5 * 1.0 + (-0.5) * 1.0 + 0 = 0 -> sigmoid(0) = 0.5
    assert neuron.activate([1.0, 1.0]) == 0.5

    # Large values stability
    assert SigmoidNeuron.sigmoid(100.0) == 1.0
    assert SigmoidNeuron.sigmoid(-100.0) == 0.0

    # Derivative at z=0: sigmoid(0)=0.5, deriv = 0.5 * 0.5 = 0.25
    assert SigmoidNeuron.sigmoid_derivative(0.0) == 0.25
