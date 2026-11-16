package com.trafficsign.data.sampling;

import java.util.*;

public class BalancedSampler {

    public static int[] generateBalancedIndices(int[] labels, int numClasses, long seed) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (labels.length == 0) {
            return new int[0];
        }

        Map<Integer, List<Integer>> classToIndices = new TreeMap<>();
        for (int i = 0; i < labels.length; i++) {
            classToIndices.computeIfAbsent(labels[i], k -> new ArrayList<>()).add(i);
        }

        Random rng = new Random(seed);
        int totalTargetSamples = labels.length;
        int activeClasses = classToIndices.size();
        int samplesPerClass = Math.max(1, totalTargetSamples / activeClasses);

        List<Integer> balancedList = new ArrayList<>(totalTargetSamples);

        for (List<Integer> idxList : classToIndices.values()) {
            int n = idxList.size();
            for (int k = 0; k < samplesPerClass; k++) {
                balancedList.add(idxList.get(rng.nextInt(n)));
            }
        }

        while (balancedList.size() < totalTargetSamples) {
            int randomClass = rng.nextInt(activeClasses);
            var entry = (Map.Entry<Integer, List<Integer>>) classToIndices.entrySet().toArray()[randomClass];
            List<Integer> list = entry.getValue();
            balancedList.add(list.get(rng.nextInt(list.size())));
        }

        Collections.shuffle(balancedList, rng);

        int[] result = new int[balancedList.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = balancedList.get(i);
        }
        return result;
    }

    public static int[] generateSmoothedIndices(int[] labels, int numClasses, double power, long seed) {
        Objects.requireNonNull(labels, "labels must not be null");
        if (labels.length == 0) {
            return new int[0];
        }

        Map<Integer, List<Integer>> classToIndices = new TreeMap<>();
        for (int i = 0; i < labels.length; i++) {
            classToIndices.computeIfAbsent(labels[i], k -> new ArrayList<>()).add(i);
        }

        Random rng = new Random(seed);
        int totalTargetSamples = labels.length;

        double totalWeight = 0.0;
        Map<Integer, Double> classWeights = new HashMap<>();
        for (Map.Entry<Integer, List<Integer>> entry : classToIndices.entrySet()) {
            double w = Math.pow(entry.getValue().size(), power);
            classWeights.put(entry.getKey(), w);
            totalWeight += w;
        }

        List<Integer> balancedList = new ArrayList<>(totalTargetSamples);
        int allocated = 0;
        for (Map.Entry<Integer, List<Integer>> entry : classToIndices.entrySet()) {
            int c = entry.getKey();
            List<Integer> idxList = entry.getValue();
            int n = idxList.size();
            double prob = classWeights.get(c) / totalWeight;
            int count = (int) Math.round(prob * totalTargetSamples);
            if (count < 1) count = 1;
            for (int k = 0; k < count && allocated < totalTargetSamples; k++) {
                balancedList.add(idxList.get(rng.nextInt(n)));
                allocated++;
            }
        }

        List<Integer> activeClassKeys = new ArrayList<>(classToIndices.keySet());
        while (balancedList.size() < totalTargetSamples) {
            int c = activeClassKeys.get(rng.nextInt(activeClassKeys.size()));
            List<Integer> list = classToIndices.get(c);
            balancedList.add(list.get(rng.nextInt(list.size())));
        }

        while (balancedList.size() > totalTargetSamples) {
            balancedList.remove(balancedList.size() - 1);
        }

        Collections.shuffle(balancedList, rng);

        int[] result = new int[balancedList.size()];
        for (int i = 0; i < result.length; i++) {
            result[i] = balancedList.get(i);
        }
        return result;
    }

    public static float[] computeCappedWeights(int[] labels, int numClasses, float maxCap) {
        Objects.requireNonNull(labels, "labels must not be null");
        int total = labels.length;
        int[] counts = new int[numClasses];
        for (int l : labels) {
            if (l >= 0 && l < numClasses) {
                counts[l]++;
            }
        }

        float[] weights = new float[numClasses];
        for (int c = 0; c < numClasses; c++) {
            if (counts[c] > 0) {
                float w = (float) total / (float) (numClasses * counts[c]);
                weights[c] = Math.min(w, maxCap);
            } else {
                weights[c] = 1.0f;
            }
        }

        float sum = 0f;
        for (float w : weights) sum += w;
        float mean = sum / numClasses;
        for (int c = 0; c < numClasses; c++) {
            weights[c] /= mean;
        }

        return weights;
    }
}
