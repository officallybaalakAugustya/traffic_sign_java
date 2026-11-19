"""
Feed-Forward Multi-Layer Perceptron (MLP), Trainer, and Checkpoint
mirroring com.trafficsign.neural.mlp.
Features Xavier/Glorot parameter initialization, forward propagation,
analytical backpropagation, multi-class Softmax cross-entropy loss,
and checkpoint serialization.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from trafficsign.math.matrix import Matrix
from trafficsign.neural.activation import ActivationFunction, ReLUActivation, Softmax
from trafficsign.neural.loss import CategoricalCrossEntropyLoss
from trafficsign.neural.optimizer import Optimizer, StochasticGradientDescent


@dataclass
class Gradients:
    grad_weights: List[Matrix]
    grad_biases: List[np.ndarray]


@dataclass
class ForwardPassResult:
    z_values: List[Matrix]
    activations: List[Matrix]
    output_probabilities: np.ndarray


@dataclass
class MlpCheckpoint:
    architecture: List[int]
    num_layers: int
    weights: List[List[List[float]]]
    biases: List[List[float]]
    epoch: int
    val_loss: float
    val_accuracy: float
    val_macro_f1: float

    def save_json(self, path: Union[str, Path]) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> MlpCheckpoint:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            architecture=data["architecture"],
            num_layers=data["numLayers"] if "numLayers" in data else data.get("num_layers", len(data["architecture"]) - 1),
            weights=data["weights"],
            biases=data["biases"],
            epoch=data.get("epoch", 0),
            val_loss=data.get("valLoss", data.get("val_loss", 0.0)),
            val_accuracy=data.get("valAccuracy", data.get("val_accuracy", 0.0)),
            val_macro_f1=data.get("valMacroF1", data.get("val_macro_f1", 0.0)),
        )


class MLPNetwork:
    """
    Configurable Feed-Forward Multi-Layer Perceptron (MLP).
    """

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activations: Optional[Sequence[ActivationFunction]] = None,
        random_seed: int = 42
    ):
        if layer_sizes is None or len(layer_sizes) < 2:
            raise ValueError("MLP must have at least an input and output layer (length >= 2)")
        for s in layer_sizes:
            if s <= 0:
                raise ValueError(f"Layer size must be positive: {s}")

        self.layer_sizes = list(layer_sizes)
        self.num_layers = len(layer_sizes) - 1

        if activations is None:
            self.activations = [ReLUActivation() for _ in range(self.num_layers - 1)]
        else:
            if len(activations) != self.num_layers - 1:
                raise ValueError(
                    f"Expected {self.num_layers - 1} hidden activation functions, but received {len(activations)}"
                )
            self.activations = list(activations)

        self.weights: List[Matrix] = []
        self.biases: List[np.ndarray] = []
        self.loss_function = CategoricalCrossEntropyLoss()

        self._initialize_parameters(random_seed)

    def _initialize_parameters(self, seed: int) -> None:
        rng = random.Random(seed)
        self.weights = []
        self.biases = []

        for l in range(self.num_layers):
            fan_in = self.layer_sizes[l]
            fan_out = self.layer_sizes[l + 1]
            limit = math.sqrt(6.0 / (fan_in + fan_out))

            w = Matrix.random(fan_in, fan_out, rng, -limit, limit)
            b = np.zeros(fan_out, dtype=np.float64)
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, input_x: Matrix) -> ForwardPassResult:
        if input_x is None:
            raise ValueError("inputX must not be null")
        if input_x.cols() != self.layer_sizes[0]:
            raise ValueError(
                f"Input features {input_x.cols()} does not match network input dimension {self.layer_sizes[0]}"
            )

        batch_size = input_x.rows()
        z_values: List[Matrix] = []
        act_values: List[Matrix] = [input_x]

        for l in range(self.num_layers):
            prev_a = act_values[l]
            w = self.weights[l]
            b = self.biases[l]

            # Linear transformation Z = A * W + b
            z = prev_a.multiply(w)
            for r in range(batch_size):
                for c in range(len(b)):
                    z.set(r, c, z.get(r, c) + b[c])
            z_values.append(z)

            if l < self.num_layers - 1:
                # Hidden layer activation
                act = self.activations[l]
                a = z.apply(act.activate)
                act_values.append(a)
            else:
                # Output layer: Softmax
                z_arr = z.to_array()
                probs = Softmax.activate_batch(z_arr)
                act_values.append(Matrix.from_array(probs))

        output_probs = act_values[self.num_layers].to_array()
        return ForwardPassResult(z_values, act_values, output_probs)

    def backward(self, fwd: ForwardPassResult, true_labels: Sequence[int]) -> Gradients:
        if fwd is None or true_labels is None:
            raise ValueError("ForwardPassResult and trueLabels must not be null")

        act_values = fwd.activations
        z_values = fwd.z_values
        output_probs = fwd.output_probabilities

        batch_size = act_values[0].rows()
        grad_weights: List[Optional[Matrix]] = [None] * self.num_layers
        grad_biases: List[Optional[np.ndarray]] = [None] * self.num_layers

        # 1. Output error delta^[L] = (1 / N) * (probs - target)
        delta = self.loss_function.compute_output_gradients(output_probs, true_labels)

        # 2. Backpropagate layer by layer
        for l in range(self.num_layers - 1, -1, -1):
            prev_a = act_values[l]

            # grad_W^[l] = (A^[l-1])^T * delta^[l]
            grad_weights[l] = prev_a.transpose().multiply(delta)

            # grad_b^[l] = sum across batch of delta^[l]
            grad_biases[l] = delta.sum_cols()

            if l > 0:
                w_t = self.weights[l].transpose()
                delta_prev = delta.multiply(w_t)

                act = self.activations[l - 1]
                z_prev = z_values[l - 1]
                act_deriv = Matrix(batch_size, self.layer_sizes[l])

                for r in range(batch_size):
                    for c in range(self.layer_sizes[l]):
                        z_val = z_prev.get(r, c)
                        a_val = act_values[l].get(r, c)
                        act_deriv.set(r, c, act.derivative(z_val, a_val))

                delta = delta_prev.element_multiply(act_deriv)

        return Gradients(grad_weights, grad_biases)

    def train_batch(self, batch_x: Matrix, batch_y: Sequence[int], optimizer: Optimizer) -> float:
        fwd = self.forward(batch_x)
        loss = self.loss_function.compute_loss(fwd.output_probabilities, batch_y)
        grads = self.backward(fwd, batch_y)

        for l in range(self.num_layers):
            optimizer.update(l, self.weights[l], self.biases[l], grads.grad_weights[l], grads.grad_biases[l])

        return loss

    def evaluate_loss(self, x: Matrix, y: Sequence[int]) -> float:
        fwd = self.forward(x)
        return self.loss_function.compute_loss(fwd.output_probabilities, y)

    def predict(self, x: Matrix) -> np.ndarray:
        fwd = self.forward(x)
        probs = fwd.output_probabilities
        return np.argmax(probs, axis=1)

    def get_num_layers(self) -> int:
        return self.num_layers

    def get_layer_sizes(self) -> List[int]:
        return list(self.layer_sizes)

    def get_weights(self, layer: int) -> Matrix:
        return self.weights[layer]

    def get_biases(self, layer: int) -> np.ndarray:
        return self.biases[layer].copy()

    def set_weight(self, layer: int, r: int, c: int, val: float) -> None:
        self.weights[layer].set(r, c, val)

    def set_bias(self, layer: int, idx: int, val: float) -> None:
        self.biases[layer][idx] = float(val)

    def set_weights(self, layer: int, w: Matrix) -> None:
        if w is None:
            raise ValueError("weights matrix must not be null")
        self.weights[layer] = w

    def set_biases(self, layer: int, b: Sequence[float]) -> None:
        if b is None:
            raise ValueError("biases array must not be null")
        self.biases[layer] = np.array(b, dtype=np.float64).copy()

    def to_checkpoint(self, epoch: int = 0, val_loss: float = 0.0, val_acc: float = 0.0, val_f1: float = 0.0) -> MlpCheckpoint:
        w_list = [w.to_array().tolist() for w in self.weights]
        b_list = [b.tolist() for b in self.biases]
        return MlpCheckpoint(
            architecture=list(self.layer_sizes),
            num_layers=self.num_layers,
            weights=w_list,
            biases=b_list,
            epoch=epoch,
            val_loss=val_loss,
            val_accuracy=val_acc,
            val_macro_f1=val_f1
        )

    def load_checkpoint(self, checkpoint: MlpCheckpoint) -> None:
        self.layer_sizes = list(checkpoint.architecture)
        self.num_layers = checkpoint.num_layers
        self.weights = [Matrix.from_array(w) for w in checkpoint.weights]
        self.biases = [np.array(b, dtype=np.float64) for b in checkpoint.biases]


@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    val_macro_f1: float
    epoch_time_ms: int


@dataclass
class TrainingRunResult:
    epoch_history: List[TrainingMetrics]
    best_val_loss: float
    best_val_accuracy: float
    best_val_macro_f1: float
    best_epoch: int
    checkpoint_path: Path
    total_training_time_ms: int


class MlpTrainer:
    """
    MLP Trainer for training feed-forward models on traffic sign dataset splits.
    """

    def __init__(self, target_resolution: int = 32):
        self.target_resolution = target_resolution

    def train_epochs(
        self,
        network: MLPNetwork,
        train_x: Matrix,
        train_y: Sequence[int],
        val_x: Matrix,
        val_y: Sequence[int],
        optimizer: Optimizer,
        epochs: int,
        batch_size: int = 64,
        checkpoint_dir: Optional[Path] = None
    ) -> TrainingRunResult:
        optimizer.initialize(
            network.num_layers,
            [[w.rows(), w.cols()] for w in network.weights],
            [len(b) for b in network.biases]
        )

        n_samples = train_x.rows()
        history: List[TrainingMetrics] = []
        best_val_loss = float("inf")
        best_acc = 0.0
        best_f1 = 0.0
        best_epoch = 0
        best_ckpt_path = Path("models/checkpoints/mlp_best.json") if checkpoint_dir is None else checkpoint_dir / "mlp_best.json"

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            ep_start = time.time()
            indices = list(range(n_samples))
            random.shuffle(indices)

            epoch_loss = 0.0
            n_batches = 0

            for b_start in range(0, n_samples, batch_size):
                b_end = min(b_start + batch_size, n_samples)
                b_idx = indices[b_start:b_end]

                batch_arr = train_x.to_array()[b_idx]
                batch_labels = [train_y[i] for i in b_idx]

                b_loss = network.train_batch(Matrix.from_array(batch_arr), batch_labels, optimizer)
                epoch_loss += b_loss
                n_batches += 1

            train_loss = epoch_loss / max(1, n_batches)

            # Evaluate training accuracy on subset
            train_preds = network.predict(train_x)
            train_acc = float(np.mean(train_preds == np.array(train_y)))

            # Evaluate validation
            val_loss = network.evaluate_loss(val_x, val_y)
            val_preds = network.predict(val_x)
            val_acc = float(np.mean(val_preds == np.array(val_y)))

            # Simple macro F1
            classes = np.unique(val_y)
            f1_scores = []
            for c in classes:
                tp = np.sum((val_preds == c) & (np.array(val_y) == c))
                fp = np.sum((val_preds == c) & (np.array(val_y) != c))
                fn = np.sum((val_preds != c) & (np.array(val_y) == c))
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1_val = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
                f1_scores.append(f1_val)
            val_macro_f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0

            ep_time = int((time.time() - ep_start) * 1000)

            metric = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                val_macro_f1=val_macro_f1,
                epoch_time_ms=ep_time
            )
            history.append(metric)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_acc = val_acc
                best_f1 = val_macro_f1
                best_epoch = epoch
                if checkpoint_dir is not None:
                    ckpt = network.to_checkpoint(epoch, val_loss, val_acc, val_macro_f1)
                    ckpt.save_json(best_ckpt_path)

        total_time = int((time.time() - start_time) * 1000)
        return TrainingRunResult(
            epoch_history=history,
            best_val_loss=best_val_loss,
            best_val_accuracy=best_acc,
            best_val_macro_f1=best_f1,
            best_epoch=best_epoch,
            checkpoint_path=best_ckpt_path,
            total_training_time_ms=total_time
        )
