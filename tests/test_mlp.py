import tempfile
from pathlib import Path
import numpy as np
from trafficsign.math.matrix import Matrix
from trafficsign.neural.mlp import MLPNetwork, MlpTrainer, MlpCheckpoint
from trafficsign.neural.optimizer import StochasticGradientDescent


def test_mlp_forward_backward_and_checkpoint():
    mlp = MLPNetwork([3, 4, 2], random_seed=42)
    x = Matrix.from_array([[0.5, 0.2, 0.1], [0.1, 0.9, 0.4]])
    y = [0, 1]

    # Forward
    fwd = mlp.forward(x)
    assert fwd.output_probabilities.shape == (2, 2)
    assert np.allclose(np.sum(fwd.output_probabilities, axis=1), [1.0, 1.0])

    # Backward
    grads = mlp.backward(fwd, y)
    assert len(grads.grad_weights) == 2
    assert grads.grad_weights[0].rows() == 3
    assert grads.grad_weights[0].cols() == 4
    assert grads.grad_weights[1].rows() == 4
    assert grads.grad_weights[1].cols() == 2

    # Train batch
    opt = StochasticGradientDescent(0.1)
    loss = mlp.train_batch(x, y, opt)
    assert loss > 0.0

    # Predictions
    preds = mlp.predict(x)
    assert len(preds) == 2

    # Checkpoint serialization
    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_path = Path(tmp_dir) / "test_ckpt.json"
        ckpt = mlp.to_checkpoint(epoch=1, val_loss=0.45, val_acc=0.9, val_f1=0.88)
        ckpt.save_json(ckpt_path)

        loaded_ckpt = MlpCheckpoint.load_json(ckpt_path)
        mlp2 = MLPNetwork([3, 4, 2], random_seed=99)
        mlp2.load_checkpoint(loaded_ckpt)

        preds2 = mlp2.predict(x)
        assert np.array_equal(preds, preds2)
