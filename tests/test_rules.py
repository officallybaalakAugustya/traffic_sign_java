import numpy as np
from trafficsign.neural.rules import HebbianLearning, ErrorCorrectionLearning, CompetitiveLearning, HopfieldEnergyModel


def test_hebbian_learning():
    heb = HebbianLearning(input_dimension=2, learning_rate=0.1, use_oja_rule=True)
    heb.train_sample([1.0, 0.5])
    w = heb.get_weights()
    assert len(w) == 2
    assert not np.isnan(w).any()


def test_error_correction_learning():
    ec = ErrorCorrectionLearning(input_dimension=2, learning_rate=0.1)
    # Train on simple linear boundary
    for _ in range(50):
        ec.train_sample([1.0, 1.0], 1)
        ec.train_sample([-1.0, -1.0], 0)

    assert ec.predict([1.0, 1.0]) == 1
    assert ec.predict([-1.0, -1.0]) == 0


def test_competitive_learning():
    cl = CompetitiveLearning(num_prototypes=3, dimension=2, learning_rate=0.1, seed=42)
    winner = cl.train_sample([0.8, 0.9])
    assert 0 <= winner < 3


def test_hopfield_energy_model():
    hop = HopfieldEnergyModel(num_units=4)
    pattern = [1, -1, 1, -1]
    hop.store_pattern(pattern)

    # Stored pattern should be local energy minimum
    recalled = hop.recall([1, -1, 1, 1])  # one corrupted bit
    assert np.array_equal(recalled, pattern)
