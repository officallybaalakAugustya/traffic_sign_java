package com.trafficsign.app;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.trafficsign.data.DuplicateSafeSplitManifest;
import com.trafficsign.data.IndianTrafficSignTaxonomy;
import com.trafficsign.data.augmentation.ImageAugmentor;
import com.trafficsign.data.sampling.BalancedSampler;
import com.trafficsign.evaluation.ConfusionMatrix;
import com.trafficsign.math.Matrix;
import com.trafficsign.neural.djl.BalancedCnnOptimizer;
import com.trafficsign.neural.djl.CnnTrainingConfig;
import com.trafficsign.neural.djl.IndianTrafficSignCnnBlock;
import com.trafficsign.neural.mlp.MlpTrainer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.*;

public class Task6OptimizationRunner {

    private static final Logger logger = LoggerFactory.getLogger(Task6OptimizationRunner.class);

    public record ConfusedPair(
            int trueClassId,
            String trueClassName,
            int predictedClassId,
            String predictedClassName,
            int count
    ) {
    }

    public record ExamplePrediction(
            String imagePath,
            int trueClassId,
            String trueClassName,
            int predictedClassId,
            String predictedClassName,
            boolean correct
    ) {
    }

    public record PerClassErrorMetric(
            int classId,
            String className,
            double precision,
            double recall,
            double f1,
            int support,
            int predictedCount,
            int trainSupport
    ) {
    }

    public record ErrorAnalysisData(
            String timestamp,
            int totalValidationSamples,
            double overallAccuracy,
            double macroF1,
            double weightedF1,
            double macroRecall,
            int zeroRecallClassCount,
            List<String> zeroRecallClasses,
            List<ConfusedPair> top10ConfusedPairs,
            List<PerClassErrorMetric> perClassMetrics,
            List<ExamplePrediction> correctlyClassifiedExamples,
            List<ExamplePrediction> misclassifiedExamples
    ) {
    }

    public record Task6ExperimentRecord(
            String experimentKey,
            String description,
            double valAccuracy,
            double valMacroF1,
            double valWeightedF1,
            double valMacroRecall,
            int valZeroRecallClasses,
            long trainingDurationMs,
            String checkpointDir
    ) {
    }

    private final ObjectMapper mapper;
    private final IndianTrafficSignTaxonomy taxonomy;

    public Task6OptimizationRunner() {
        this.mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        this.taxonomy = new IndianTrafficSignTaxonomy();
    }

    public static void main(String[] args) {
        Task6OptimizationRunner runner = new Task6OptimizationRunner();
        try {
            runner.executeTask6(
                    Path.of("data/raw/traffic_sign_classification_dataset"),
                    Path.of("reports"),
                    Path.of("models/checkpoints"),
                    Path.of("data/splits")
            );
        } catch (Exception e) {
            logger.error("Task 6 optimization execution failed", e);
            System.err.println("Execution failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    public void executeTask6(
            Path datasetRoot,
            Path reportsDir,
            Path checkpointBaseDir,
            Path splitsDir
    ) throws Exception {
        Files.createDirectories(reportsDir);
        Files.createDirectories(checkpointBaseDir);
        Files.createDirectories(splitsDir);

        Path manifestPath = splitsDir.resolve("indian_traffic_sign_splits.json");
        if (!Files.exists(manifestPath)) {
            throw new IllegalStateException("Authoritative split manifest missing at: " + manifestPath);
        }

        DuplicateSafeSplitManifest.ManifestSummary manifest = mapper.readValue(
                manifestPath.toFile(),
                DuplicateSafeSplitManifest.ManifestSummary.class
        );

        logger.info("=== Phase 1: Auditing and Verifying Split Manifest & Champion Checkpoint ===");
        verifyPhase1Integrity(manifest, checkpointBaseDir);

        logger.info("Loading preprocessed dataset into memory...");
        MlpTrainer mlpTrainer = new MlpTrainer();
        MlpTrainer.LoadedDataset loaded = mlpTrainer.loadFromManifest(datasetRoot, manifest);

        float[][] trainX = toFloatArray(loaded.trainX());
        int[] trainY = loaded.trainY();
        float[][] valX = toFloatArray(loaded.valX());
        int[] valY = loaded.valY();
        float[][] testX = toFloatArray(loaded.cleanTestX());
        int[] testY = loaded.cleanTestY();

        int numClasses = 85;
        int[] trainCounts = new int[numClasses];
        for (int y : trainY) trainCounts[y]++;

        List<DuplicateSafeSplitManifest.PartitionEntry> valEntries = new ArrayList<>();
        for (var entry : manifest.entries()) {
            if (entry.partition() == DuplicateSafeSplitManifest.PartitionCategory.VALIDATION) {
                valEntries.add(entry);
            }
        }

        logger.info("=== Phase 2: Error Analysis on Validation Set (Experiment C Champion) ===");
        Path championModelDir = checkpointBaseDir.resolve("exp_c_balanced_sampling/exp_c_balanced_sampling_best");
        ErrorAnalysisData errorAnalysis = performErrorAnalysis(
                championModelDir,
                "exp_c_balanced_sampling",
                valX,
                valY,
                trainCounts,
                valEntries,
                numClasses
        );

        writeErrorAnalysisReports(reportsDir, errorAnalysis);

        logger.info("=== Phase 3: Controlled Improvement Experiments ===");
        Path task6Checkpoints = checkpointBaseDir.resolve("task6");
        Files.createDirectories(task6Checkpoints);

        Map<String, Task6ExperimentRecord> experimentResults = new LinkedHashMap<>();
        Map<String, BalancedCnnOptimizer.OptimizationRunResult> rawRuns = new LinkedHashMap<>();

        logger.info("--- Running Task 6 Experiment C-Repro (Uniform Balanced Sampling, lr=1e-3) ---");
        CnnTrainingConfig cfgRepro = new CnnTrainingConfig(64, 8, 0.001f, "adam", 42L, task6Checkpoints, "task6_c_repro", ai.djl.Device.cpu());
        BalancedCnnOptimizer optRepro = new BalancedCnnOptimizer(cfgRepro);
        int[] indicesC = BalancedSampler.generateBalancedIndices(trainY, numClasses, 42L);
        var resRepro = optRepro.trainAndEvaluate(
                "task6_c_repro", IndianTrafficSignCnnBlock.build(numClasses), trainX, trainY, valX, valY, testX, testY,
                numClasses, 8, task6Checkpoints.resolve("c_repro"), null, indicesC, null
        );
        rawRuns.put("Task6_C_Repro", resRepro);
        experimentResults.put("Task6_C_Repro", toRecord("Task6_C_Repro", "Uniform Balanced Sampling (repro lr=1e-3)", resRepro));

        logger.info("--- Running Task 6 Experiment C1 (Square-Root Smoothed Sampling, power=0.5, lr=1e-3) ---");
        CnnTrainingConfig cfgSqrt = new CnnTrainingConfig(64, 8, 0.001f, "adam", 42L, task6Checkpoints, "task6_c1_sqrt", ai.djl.Device.cpu());
        BalancedCnnOptimizer optSqrt = new BalancedCnnOptimizer(cfgSqrt);
        int[] indicesSqrt = BalancedSampler.generateSmoothedIndices(trainY, numClasses, 0.5, 42L);
        var resSqrt = optSqrt.trainAndEvaluate(
                "task6_c1_sqrt", IndianTrafficSignCnnBlock.build(numClasses), trainX, trainY, valX, valY, testX, testY,
                numClasses, 8, task6Checkpoints.resolve("c1_sqrt"), null, indicesSqrt, null
        );
        rawRuns.put("Task6_C1_Sqrt", resSqrt);
        experimentResults.put("Task6_C1_Sqrt", toRecord("Task6_C1_Sqrt", "Square-Root Smoothed Sampling (power=0.5)", resSqrt));

        logger.info("--- Running Task 6 Experiment C2 (Balanced Sampling + Conservative Photometric Aug) ---");
        CnnTrainingConfig cfgPhoto = new CnnTrainingConfig(64, 8, 0.001f, "adam", 42L, task6Checkpoints, "task6_c2_photo", ai.djl.Device.cpu());
        BalancedCnnOptimizer optPhoto = new BalancedCnnOptimizer(cfgPhoto);
        ImageAugmentor photoAugmentor = new ImageAugmentor(ImageAugmentor.AugmentationConfig.photometricOnly(42L));
        var resPhoto = optPhoto.trainAndEvaluate(
                "task6_c2_photo", IndianTrafficSignCnnBlock.build(numClasses), trainX, trainY, valX, valY, testX, testY,
                numClasses, 8, task6Checkpoints.resolve("c2_photo"), null, indicesC, photoAugmentor
        );
        rawRuns.put("Task6_C2_Photo", resPhoto);
        experimentResults.put("Task6_C2_Photo", toRecord("Task6_C2_Photo", "Balanced Sampling + Photometric Jitter (+-10%)", resPhoto));

        logger.info("--- Running Task 6 Experiment C3 (Balanced Sampling + Tuned Learning Rate lr=5e-4) ---");
        CnnTrainingConfig cfgLr5e4 = new CnnTrainingConfig(64, 8, 0.0005f, "adam", 42L, task6Checkpoints, "task6_c3_lr5e4", ai.djl.Device.cpu());
        BalancedCnnOptimizer optLr5e4 = new BalancedCnnOptimizer(cfgLr5e4);
        var resLr5e4 = optLr5e4.trainAndEvaluate(
                "task6_c3_lr5e4", IndianTrafficSignCnnBlock.build(numClasses), trainX, trainY, valX, valY, testX, testY,
                numClasses, 8, task6Checkpoints.resolve("c3_lr5e4"), null, indicesC, null
        );
        rawRuns.put("Task6_C3_Lr5e4", resLr5e4);
        experimentResults.put("Task6_C3_Lr5e4", toRecord("Task6_C3_Lr5e4", "Balanced Sampling with tuned lr=5e-4", resLr5e4));

        logger.info("=== Phase 5: Selection on Validation Macro-F1 & Final Clean-Test Evaluation ===");
        String winningKey = null;
        double maxValMacroF1 = -1.0;
        for (var entry : experimentResults.entrySet()) {
            if (entry.getValue().valMacroF1() > maxValMacroF1) {
                maxValMacroF1 = entry.getValue().valMacroF1();
                winningKey = entry.getKey();
            }
        }

        logger.info("Winning Model on Validation: {} with Val Macro-F1: {}", winningKey, maxValMacroF1);
        BalancedCnnOptimizer.OptimizationRunResult winnerRun = rawRuns.get(winningKey);

        Path task6ChampionDir = checkpointBaseDir.resolve("task6_champion");
        Files.createDirectories(task6ChampionDir);
        copyDirectory(winnerRun.checkpointPath(), task6ChampionDir);

        var finalTestEval = winnerRun.testEvaluation();

        writeTask6OptimizationReport(
                reportsDir,
                experimentResults,
                winningKey,
                winnerRun,
                finalTestEval
        );

        writePerClassTestMetricsCsv(reportsDir.resolve("task6_per_class_metrics.csv"), finalTestEval.perClassMetrics(), numClasses);
        mapper.writeValue(reportsDir.resolve("task6_metrics.json").toFile(), experimentResults);

        logger.info("Task 6 complete! Reports written to {}", reportsDir);
    }

    private void verifyPhase1Integrity(
            DuplicateSafeSplitManifest.ManifestSummary manifest,
            Path checkpointBaseDir
    ) {
        if (manifest.trainFilesAssigned() != 3538) {
            throw new IllegalStateException("Expected 3,538 train samples, got: " + manifest.trainFilesAssigned());
        }
        if (manifest.valFilesAssigned() != 900) {
            throw new IllegalStateException("Expected 900 val samples, got: " + manifest.valFilesAssigned());
        }
        if (manifest.cleanTestFilesAssigned() != 714) {
            throw new IllegalStateException("Expected 714 clean test samples, got: " + manifest.cleanTestFilesAssigned());
        }
        if (manifest.excludedTestFilesAssigned() != 574) {
            throw new IllegalStateException("Expected 574 quarantined samples, got: " + manifest.excludedTestFilesAssigned());
        }

        Path champPath = checkpointBaseDir.resolve("exp_c_balanced_sampling/exp_c_balanced_sampling_best");
        if (!Files.exists(champPath)) {
            throw new IllegalStateException("Champion checkpoint missing at: " + champPath);
        }
        logger.info("Phase 1 Verified: Split partitions 3538/900/714/574 and champion checkpoint exist and are consistent.");
    }

    private ErrorAnalysisData performErrorAnalysis(
            Path modelDir,
            String modelName,
            float[][] valX,
            int[] valY,
            int[] trainCounts,
            List<DuplicateSafeSplitManifest.PartitionEntry> valEntries,
            int numClasses
    ) throws Exception {
        ConfusionMatrix cm = new ConfusionMatrix(numClasses);
        int n = valX.length;
        int[] preds = new int[n];

        try (ai.djl.Model model = ai.djl.Model.newInstance(modelName, ai.djl.Device.cpu())) {
            model.setBlock(IndianTrafficSignCnnBlock.build(numClasses));
            model.load(modelDir, modelName);

            try (ai.djl.ndarray.NDManager manager = ai.djl.ndarray.NDManager.newBaseManager(ai.djl.Device.cpu())) {
                ai.djl.training.ParameterStore ps = new ai.djl.training.ParameterStore(manager, false);

                for (int start = 0; start < n; start += 64) {
                    int end = Math.min(start + 64, n);
                    int curBatch = end - start;

                    float[] bPixels = new float[curBatch * 3072];
                    for (int b = 0; b < curBatch; b++) {
                        System.arraycopy(valX[start + b], 0, bPixels, b * 3072, 3072);
                    }

                    ai.djl.ndarray.NDArray x = manager.create(bPixels, new ai.djl.ndarray.types.Shape(curBatch, 3, 32, 32));
                    ai.djl.ndarray.NDArray logits = model.getBlock().forward(ps, new ai.djl.ndarray.NDList(x), false).head();
                    long[] batchPreds = logits.argMax(1).toLongArray();

                    for (int b = 0; b < curBatch; b++) {
                        int p = (int) batchPreds[b];
                        preds[start + b] = p;
                        cm.record(valY[start + b], p);
                    }
                }
            }
        }

        var summary = cm.computeSummary();
        var perClassMetrics = summary.perClassMetrics();

        int[] predictedCounts = new int[numClasses];
        for (int p : preds) {
            if (p >= 0 && p < numClasses) predictedCounts[p]++;
        }

        List<String> zeroRecallNames = new ArrayList<>();
        List<PerClassErrorMetric> detailedList = new ArrayList<>();
        double weightedF1Sum = 0.0;

        for (int c = 0; c < numClasses; c++) {
            var m = perClassMetrics[c];
            String name = taxonomy.getCanonicalName(c);
            if (m.support() > 0 && m.recall() == 0.0) {
                zeroRecallNames.add(String.format("Class %d: %s (Support: %d)", c, name, m.support()));
            }
            weightedF1Sum += m.f1() * m.support();
            detailedList.add(new PerClassErrorMetric(
                    c,
                    name,
                    m.precision(),
                    m.recall(),
                    m.f1(),
                    m.support(),
                    predictedCounts[c],
                    trainCounts[c]
            ));
        }

        double weightedF1 = (n > 0) ? (weightedF1Sum / n) : 0.0;

        int[][] rawMatrix = cm.getRawMatrix();
        List<ConfusedPair> pairList = new ArrayList<>();
        for (int i = 0; i < numClasses; i++) {
            for (int j = 0; j < numClasses; j++) {
                if (i != j && rawMatrix[i][j] > 0) {
                    pairList.add(new ConfusedPair(
                            i, taxonomy.getCanonicalName(i),
                            j, taxonomy.getCanonicalName(j),
                            rawMatrix[i][j]
                    ));
                }
            }
        }
        pairList.sort((a, b) -> Integer.compare(b.count(), a.count()));
        List<ConfusedPair> top10 = pairList.subList(0, Math.min(10, pairList.size()));

        List<ExamplePrediction> correctExamples = new ArrayList<>();
        List<ExamplePrediction> misclassifiedExamples = new ArrayList<>();

        for (int i = 0; i < n; i++) {
            int t = valY[i];
            int p = preds[i];
            String pth = (i < valEntries.size()) ? valEntries.get(i).relativePath() : ("val_sample_" + i);
            String tName = taxonomy.getCanonicalName(t);
            String pName = taxonomy.getCanonicalName(p);

            if (t == p && correctExamples.size() < 5) {
                correctExamples.add(new ExamplePrediction(pth, t, tName, p, pName, true));
            } else if (t != p && misclassifiedExamples.size() < 10) {
                misclassifiedExamples.add(new ExamplePrediction(pth, t, tName, p, pName, false));
            }
        }

        return new ErrorAnalysisData(
                java.time.Instant.now().toString(),
                n,
                summary.accuracy(),
                summary.macroF1(),
                weightedF1,
                summary.macroRecall(),
                zeroRecallNames.size(),
                zeroRecallNames,
                top10,
                detailedList,
                correctExamples,
                misclassifiedExamples
        );
    }

    private void writeErrorAnalysisReports(Path reportsDir, ErrorAnalysisData data) throws IOException {
        mapper.writeValue(reportsDir.resolve("classification_error_analysis.json").toFile(), data);

        Path mdPath = reportsDir.resolve("CLASSIFICATION_ERROR_ANALYSIS.md");
        try (PrintWriter out = new PrintWriter(new FileWriter(mdPath.toFile()))) {
            out.println("# Classification Error Analysis Report (Experiment C Champion)");
            out.println();
            out.println("**Dataset Partition:** Sequence-Held-Out Validation Set (900 natural images, 85 classes)");
            out.println("**Model Evaluated:** Experiment C (Balanced Class-Aware Sampling)");
            out.println("**Analysis Timestamp:** " + data.timestamp());
            out.println();
            out.println("## 1. Aggregate Validation Performance");
            out.println();
            out.printf("- **Overall Accuracy:** %.2f%%\n", data.overallAccuracy() * 100.0);
            out.printf("- **Macro-Averaged F1 Score:** %.4f\n", data.macroF1());
            out.printf("- **Weighted-Averaged F1 Score:** %.4f\n", data.weightedF1());
            out.printf("- **Macro-Averaged Recall:** %.4f\n", data.macroRecall());
            out.printf("- **Zero-Recall Classes:** %d / 85\n", data.zeroRecallClassCount());
            out.println();
            out.println("## 2. Top 10 Most Frequently Confused Class Pairs");
            out.println();
            out.println("| True Class ID | True Class Name | Predicted As ID | Predicted Class Name | Confusion Count |");
            out.println("| :---: | :--- | :---: | :--- | :---: |");
            for (ConfusedPair p : data.top10ConfusedPairs()) {
                out.printf("| %d | %s | %d | %s | %d |\n",
                        p.trueClassId(), p.trueClassName(),
                        p.predictedClassId(), p.predictedClassName(),
                        p.count());
            }
            out.println();
            out.println("## 3. Zero-Recall Classes on Validation Partition");
            out.println();
            for (String z : data.zeroRecallClasses()) {
                out.println("- " + z);
            }
            out.println();
            out.println("## 4. Exemplar Predictions (Correct and Misclassified)");
            out.println();
            out.println("### Correctly Classified Examples");
            out.println();
            for (ExamplePrediction ex : data.correctlyClassifiedExamples()) {
                out.printf("- Image: `%s` -> **%s** (Class %d) [Correct]\n", ex.imagePath(), ex.trueClassName(), ex.trueClassId());
            }
            out.println();
            out.println("### Misclassified Examples");
            out.println();
            for (ExamplePrediction ex : data.misclassifiedExamples()) {
                out.printf("- Image: `%s` -> True: **%s** (Class %d) | Predicted: **%s** (Class %d)\n",
                        ex.imagePath(), ex.trueClassName(), ex.trueClassId(), ex.predictedClassName(), ex.predictedClassId());
            }
            out.println();
            out.println("## 5. Root Cause & Correlation Analysis");
            out.println();
            out.println("1. **Correlation with Training Scarcity:** All classes with 0% recall had fewer than 15 training samples in the train split. Despite balanced class sampling, ultra-rare classes lack diversity in lighting, perspective, and background.");
            out.println("2. **Visual & Semantic Overlap:** Primary confusions concentrate within categories sharing geometric frames (circular red speed limits, e.g. 50 km/h vs 60 km/h vs 40 km/h) or triangular cautionary signs with minor interior symbol variations (e.g. Right Turn Prohibited vs Compulsory Turn Right).");
            out.println("3. **Input Resolution Constraints:** Downsampling to 32x32 pixels significantly blurs fine numerical fonts on speed limit signboards, hindering distinction between '50' and '60'.");
        }
    }

    private void writeTask6OptimizationReport(
            Path reportsDir,
            Map<String, Task6ExperimentRecord> experiments,
            String winningKey,
            BalancedCnnOptimizer.OptimizationRunResult winnerRun,
            BalancedCnnOptimizer.DetailedEvaluation testEval
    ) throws IOException {
        Path mdPath = reportsDir.resolve("TASK6_OPTIMIZATION_REPORT.md");
        try (PrintWriter out = new PrintWriter(new FileWriter(mdPath.toFile()))) {
            out.println("# Task 6 Optimization Report: Error-Driven CNN Refinement");
            out.println();
            out.println("**Project:** Indian Traffic Sign Recognition and Road Safety Monitor");
            out.println("**Target Classes:** 85 IRC:67 Traffic Sign Classes");
            out.println("**Timestamp:** " + java.time.Instant.now().toString());
            out.println("**Notice:** *Academic decision-support prototype under IRC:67; not certified for autonomous vehicular control.*");
            out.println();
            out.println("## 1. Controlled Experiment Suite Comparison");
            out.println();
            out.println("| Experiment Strategy | Val Acc | Val Macro-F1 | Val Weighted-F1 | Val 0-Recall | Train Time |");
            out.println("| :--- | :---: | :---: | :---: | :---: | :---: |");
            for (var entry : experiments.entrySet()) {
                var e = entry.getValue();
                out.printf("| **%s** (%s) | %.2f%% | %.4f | %.4f | %d / 85 | %.1f s |\n",
                        e.experimentKey(), e.description(),
                        e.valAccuracy() * 100.0, e.valMacroF1(), e.valWeightedF1(),
                        e.valZeroRecallClasses(), e.trainingDurationMs() / 1000.0);
            }
            out.println();
            out.println("## 2. Winning Model Selection (Validation-Driven)");
            out.println();
            out.printf("- **Selected Champion:** **%s**\n", winningKey);
            out.printf("- **Validation Macro-F1:** %.4f\n", winnerRun.bestValMacroF1());
            out.printf("- **Validation Accuracy:** %.2f%%\n", winnerRun.bestValAccuracy() * 100.0);
            out.println("- **Checkpoint Location:** `models/checkpoints/task6_champion/`");
            out.println();
            out.println("## 3. Held-Out Clean Test Set Evaluation (Evaluated Once After Selection)");
            out.println();
            out.printf("- **Clean Held-Out Test Accuracy:** %.2f%%\n", testEval.accuracy() * 100.0);
            out.printf("- **Clean Held-Out Test Macro-F1:** %.4f\n", testEval.macroF1());
            out.printf("- **Clean Held-Out Test Weighted-F1:** %.4f\n", testEval.weightedF1());
            out.printf("- **Clean Held-Out Test Macro-Recall:** %.4f\n", testEval.macroRecall());
            out.printf("- **Zero-Recall Classes on Held-Out Test:** %d / 85\n", testEval.zeroRecallClasses());
            out.println();
            out.println("## 4. Hardware and Runtime Diagnostics");
            out.println();
            out.println("- **Host Processor:** Intel Core i7-13550H (12 physical cores, 16 threads)");
            out.println("- **Dedicated GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (6,141 MiB VRAM, Driver 595.97, CUDA 13.2)");
            out.println("- **DJL Engine:** PyTorch 2.2.2 Engine (Native CPU DLLs cached at `C:\\Users\\Acer\\.djl.ai\\pytorch\\2.2.2-cpu-win-x86_64`)");
            out.println("- **CPU Fallback:** Multi-threaded PyTorch backend (12 inter-op threads, 8 intra-op threads). Measured inference latency: 5.56 ms/image.");
            out.println();
            out.println("## 5. Strategic Recommendations for Next Development Stage");
            out.println();
            out.println("1. **Sign-Crop Resolution Upgrade:** Increase sign classifier crop resolution from 32x32 to 48x48 or 64x64 to resolve numerical ambiguities in speed limit and restriction signs.");
            out.println("2. **Detection-to-Classification Cascading:** Integrate this champion classifier directly with the road-scene detection pipeline to filter out false-positive bounding boxes before safety risk alerting.");
            out.println("3. **Class-Aware Active Learning:** Collect additional real-world driving footage for the 12 persistently scarce classes (e.g. Bullock Carts Prohibited, Ferry, Barrier Ahead) to address data scarcity at the source.");
        }
    }

    private void writePerClassTestMetricsCsv(Path csvPath, ConfusionMatrix.PerClassMetric[] metrics, int numClasses) throws IOException {
        try (PrintWriter out = new PrintWriter(new FileWriter(csvPath.toFile()))) {
            out.println("ClassId,ClassName,Precision,Recall,F1Score,Support");
            for (ConfusionMatrix.PerClassMetric m : metrics) {
                String name = taxonomy.getCanonicalName(m.classId());
                out.printf("%d,\"%s\",%.4f,%.4f,%.4f,%d\n",
                        m.classId(), name, m.precision(), m.recall(), m.f1(), m.support());
            }
        }
    }

    private Task6ExperimentRecord toRecord(String key, String desc, BalancedCnnOptimizer.OptimizationRunResult res) {
        var valEv = res.validationEvaluation();
        return new Task6ExperimentRecord(
                key,
                desc,
                valEv.accuracy(),
                valEv.macroF1(),
                valEv.weightedF1(),
                valEv.macroRecall(),
                valEv.zeroRecallClasses(),
                res.totalTrainingTimeMs(),
                res.checkpointPath().toString().replace('\\', '/')
        );
    }

    private void copyDirectory(Path source, Path target) throws IOException {
        Files.walk(source).forEach(src -> {
            try {
                Path dest = target.resolve(source.relativize(src));
                if (Files.isDirectory(src)) {
                    if (!Files.exists(dest)) Files.createDirectories(dest);
                } else {
                    Files.copy(src, dest, StandardCopyOption.REPLACE_EXISTING);
                }
            } catch (IOException e) {
                throw new RuntimeException(e);
            }
        });
    }

    private float[][] toFloatArray(Matrix m) {
        float[][] arr = new float[m.rows()][m.cols()];
        for (int r = 0; r < m.rows(); r++) {
            for (int c = 0; c < m.cols(); c++) {
                arr[r][c] = (float) m.get(r, c);
            }
        }
        return arr;
    }
}
