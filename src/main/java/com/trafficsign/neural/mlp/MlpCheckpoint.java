package com.trafficsign.neural.mlp;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;

public record MlpCheckpoint(
        @JsonProperty("architecture") int[] architecture,
        @JsonProperty("numLayers") int numLayers,
        @JsonProperty("weights") List<double[][]> weights,
        @JsonProperty("biases") List<double[]> biases,
        @JsonProperty("epoch") int epoch,
        @JsonProperty("valLoss") double valLoss,
        @JsonProperty("valAccuracy") double valAccuracy,
        @JsonProperty("valMacroF1") double valMacroF1
) {
}
