package com.trafficsign.data;

import java.util.*;

/**
 * Audit record documenting class distributions, sample frequencies, severe class imbalances,
 * and unknown/unmapped label encounters in a dataset.
 */
public record ClassFrequencyReport(
        int totalSamples,
        int totalClassesInTaxonomy,
        Map<Integer, Integer> classFrequencies,
        Map<String, Integer> unmappedLabelCounts,
        int minSupport,
        int maxSupport,
        double imbalanceRatio,
        List<Integer> missingClassIds
) {
    public static ClassFrequencyReport generate(
            List<ImageSample> samples,
            TrafficSignTaxonomy taxonomy,
            Map<String, Integer> unmappedLabels
    ) {
        Objects.requireNonNull(samples, "samples must not be null");
        Objects.requireNonNull(taxonomy, "taxonomy must not be null");

        Map<Integer, Integer> freq = new TreeMap<>();
        for (int i = 0; i < taxonomy.totalClasses(); i++) {
            freq.put(i, 0);
        }

        for (ImageSample s : samples) {
            freq.put(s.classId(), freq.getOrDefault(s.classId(), 0) + 1);
        }

        int total = samples.size();
        List<Integer> missing = new ArrayList<>();
        int min = Integer.MAX_VALUE;
        int max = 0;

        for (var entry : freq.entrySet()) {
            int count = entry.getValue();
            if (count == 0) {
                missing.add(entry.getKey());
            } else {
                if (count < min) min = count;
                if (count > max) max = count;
            }
        }

        if (min == Integer.MAX_VALUE) {
            min = 0;
        }

        double ratio = min > 0 ? (double) max / min : (max > 0 ? Double.POSITIVE_INFINITY : 1.0);

        return new ClassFrequencyReport(
                total,
                taxonomy.totalClasses(),
                Collections.unmodifiableMap(freq),
                unmappedLabels != null ? Collections.unmodifiableMap(unmappedLabels) : Collections.emptyMap(),
                min,
                max,
                ratio,
                Collections.unmodifiableList(missing)
        );
    }

    public boolean hasSevereImbalance(double thresholdRatio) {
        return imbalanceRatio > thresholdRatio || Double.isInfinite(imbalanceRatio);
    }

    public String formatSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format("Dataset Distribution: %d total samples across %d classes.\n", totalSamples, totalClassesInTaxonomy));
        sb.append(String.format("Support Range: min=%d, max=%d, Imbalance Ratio=%.2f\n", minSupport, maxSupport, imbalanceRatio));
        if (!missingClassIds.isEmpty()) {
            sb.append(String.format("Missing/Zero-support Classes (%d): %s\n", missingClassIds.size(), missingClassIds));
        }
        if (!unmappedLabelCounts.isEmpty()) {
            sb.append(String.format("Unmapped Labels Encountered: %s\n", unmappedLabelCounts));
        }
        return sb.toString();
    }
}
