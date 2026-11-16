package com.trafficsign.app;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.trafficsign.data.ClassificationDatasetValidator;
import com.trafficsign.data.ClassificationValidationResult;
import com.trafficsign.data.IndianTrafficSignTaxonomy;
import com.trafficsign.data.detection.FrameDetectionRecord;
import com.trafficsign.data.detection.RoadDrivingTarReader;
import com.trafficsign.data.detection.TarInspectionSummary;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

public class IndianRoadPipelineCli {

    private static final Logger logger = LoggerFactory.getLogger(IndianRoadPipelineCli.class);

    private final ObjectMapper objectMapper;
    private final IndianTrafficSignTaxonomy taxonomy;

    public IndianRoadPipelineCli() {
        this.objectMapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        this.taxonomy = new IndianTrafficSignTaxonomy();
    }

    public static void main(String[] args) {
        IndianRoadPipelineCli cli = new IndianRoadPipelineCli();
        try {
            cli.run(args);
        } catch (Exception e) {
            logger.error("CLI Execution failed: {}", e.getMessage(), e);
            System.err.println("Execution failed: " + e.getMessage());
            System.exit(1);
        }
    }

    public void run(String[] args) throws IOException {
        String command = args != null && args.length > 0 ? args[0].toLowerCase() : "dataset-summary";
        Map<String, String> params = parseArgs(args);

        Path reportsDir = Path.of(params.getOrDefault("reports-dir", "reports"));
        if (!Files.exists(reportsDir)) {
            Files.createDirectories(reportsDir);
        }

        switch (command) {
            case "validate-classification" -> runClassificationValidation(params, reportsDir);
            case "inspect-tar" -> runTarInspection(params, reportsDir);
            case "dataset-summary" -> {
                runClassificationValidation(params, reportsDir);
                runTarInspection(params, reportsDir);
                generateMasterSummary(reportsDir);
            }
            default -> {
                System.out.println("Unknown command: " + command);
                System.out.println("Available commands: validate-classification, inspect-tar, dataset-summary");
                System.out.println("Options:");
                System.out.println("  --classification-root <path> (default: data/raw/traffic_sign_classification_dataset)");
                System.out.println("  --tar-dir <path>             (default: data/raw/indian-road-driving-sample)");
                System.out.println("  --limit <N>                  (default: 10)");
                System.out.println("  --seed <long>                (default: 42)");
                System.out.println("  --val-ratio <double>         (default: 0.20)");
            }
        }
    }

    private void runClassificationValidation(Map<String, String> params, Path reportsDir) throws IOException {
        Path classRoot = Path.of(params.getOrDefault("classification-root", "data/raw/traffic_sign_classification_dataset"));
        long seed = Long.parseLong(params.getOrDefault("seed", "42"));
        double valRatio = Double.parseDouble(params.getOrDefault("val-ratio", "0.20"));
        boolean verifyHeaders = Boolean.parseBoolean(params.getOrDefault("verify-headers", "true"));

        logger.info("Executing classification validation on {}", classRoot);
        ClassificationDatasetValidator validator = new ClassificationDatasetValidator(taxonomy);
        ClassificationValidationResult result = validator.validate(classRoot, seed, valRatio, verifyHeaders);

        Path jsonOut = reportsDir.resolve("classification_validation_report.json");
        Files.writeString(jsonOut, validator.generateJsonReport(result));

        Path mdOut = reportsDir.resolve("CLASSIFICATION_DATASET_REPORT.md");
        Files.writeString(mdOut, validator.generateMarkdownReport(result));

        logger.info("Classification validation complete. JSON: {}, Markdown: {}", jsonOut, mdOut);
        System.out.println("Classification validation finished successfully.");
        System.out.println("  Train images: " + result.totalTrainImages() + ", Test images: " + result.totalTestImages());
        System.out.println("  Class consistency: " + (result.isClassConsistent() ? "OK" : "WARNING"));
        System.out.println("  Train imbalance ratio: " + String.format("%.2f", result.trainImbalanceRatio()));
        System.out.println("  Derived Train/Val split: " + result.trainValSplitTrainCount() + " / " + result.trainValSplitValCount());
        System.out.println("  Reports written to: " + mdOut);
    }

    private void runTarInspection(Map<String, String> params, Path reportsDir) throws IOException {
        Path tarDir = Path.of(params.getOrDefault("tar-dir", "data/raw/indian-road-driving-sample"));
        int limit = Integer.parseInt(params.getOrDefault("limit", "10"));

        logger.info("Executing TAR inspection on {} (limit: {} archives)", tarDir, limit);
        RoadDrivingTarReader reader = new RoadDrivingTarReader();
        TarInspectionSummary summary = reader.inspectDirectory(tarDir, limit);

        Path jsonOut = reportsDir.resolve("tar_inspection_report.json");
        Files.writeString(jsonOut, objectMapper.writeValueAsString(summary));

        Path mdOut = reportsDir.resolve("TAR_INSPECTION_REPORT.md");
        Files.writeString(mdOut, generateTarMarkdown(summary, tarDir));

        logger.info("TAR inspection complete. JSON: {}, Markdown: {}", jsonOut, mdOut);
        System.out.println("TAR inspection finished successfully.");
        System.out.println("  Archives inspected: " + summary.totalArchivesProcessed());
        System.out.println("  Total frames: " + summary.totalFrames() + ", Matched pairs: " + summary.totalMatchedPairs());
        System.out.println("  PNG entries safely ignored: " + summary.totalPngIgnored());
        System.out.println("  Total objects: " + summary.totalObjects() + ", Generic traffic signs: " + summary.totalTrafficSigns());
        System.out.println("  Reports written to: " + mdOut);
    }

    private String generateTarMarkdown(TarInspectionSummary s, Path tarDir) {
        StringBuilder sb = new StringBuilder();
        sb.append("# Indian Road Driving TAR Archives Inspection Report\n\n");
        sb.append("**Source Directory:** `").append(tarDir).append("`\n\n");
        sb.append("## Archive Processing Summary\n\n");
        sb.append("| Metric | Observed Value |\n");
        sb.append("| :--- | :---: |\n");
        sb.append("| **Archives Inspected** | ").append(s.totalArchivesProcessed()).append(" |\n");
        sb.append("| **Total Frame Stems** | ").append(s.totalFrames()).append(" |\n");
        sb.append("| **Matched JPG + JSON Pairs** | ").append(s.totalMatchedPairs()).append(" |\n");
        sb.append("| **JPG Only (Missing Annotation)** | ").append(s.totalJpgOnly()).append(" |\n");
        sb.append("| **JSON Only (Missing Image Frame)** | ").append(s.totalJsonOnly()).append(" |\n");
        sb.append("| **PNG Files Ignored** | ").append(s.totalPngIgnored()).append(" (Semantic purpose unverified; excluded) |\n");
        sb.append("| **Total Annotated Objects** | ").append(s.totalObjects()).append(" |\n");
        sb.append("| **Generic Traffic Sign Annotations** | ").append(s.totalTrafficSigns()).append(" |\n\n");

        sb.append("## Detection vs Classification Distinction Notice\n\n");
        sb.append("> [!IMPORTANT]\n");
        sb.append("> The TAR archive annotations contain generic `traffic sign` bounding boxes ($x_1, y_1, x_2, y_2$). ");
        sb.append("These boxes indicate spatial locations of traffic signs in driving scenes but **DO NOT** provide ");
        sb.append("fine-grained sign classification categories (such as Stop, Speed Limit 40, or Cattle). ");
        sb.append("They must **NEVER** be conflated with the 85-class IRC:67 classification ground truth.\n\n");

        sb.append("## Object Category Distribution Across Inspected Archives\n\n");
        sb.append("| Category | Object Count | Percentage of Total |\n");
        sb.append("| :--- | :---: | :---: |\n");
        for (var entry : s.categoryDistribution().entrySet()) {
            double pct = s.totalObjects() > 0 ? (entry.getValue() * 100.0) / s.totalObjects() : 0.0;
            sb.append("| `").append(entry.getKey()).append("` | ").append(entry.getValue()).append(" | ").append(String.format("%.2f%%", pct)).append(" |\n");
        }
        sb.append("\n");

        sb.append("## Data Quality & Error Log\n\n");
        if (s.annotationErrors().isEmpty()) {
            sb.append("- Zero annotation errors or coordinate anomalies detected across inspected frames.\n\n");
        } else {
            sb.append("Total recorded anomalies: ").append(s.annotationErrors().size()).append("\n\n");
            int displayLimit = Math.min(25, s.annotationErrors().size());
            for (int i = 0; i < displayLimit; i++) {
                sb.append("- ").append(s.annotationErrors().get(i)).append("\n");
            }
            if (s.annotationErrors().size() > displayLimit) {
                sb.append("- ... and ").append(s.annotationErrors().size() - displayLimit).append(" more errors recorded in JSON report.\n");
            }
            sb.append("\n");
        }

        return sb.toString();
    }

    private void generateMasterSummary(Path reportsDir) throws IOException {
        Path classJson = reportsDir.resolve("classification_validation_report.json");
        Path tarJson = reportsDir.resolve("tar_inspection_report.json");

        if (!Files.exists(classJson) || !Files.exists(tarJson)) {
            return;
        }

        StringBuilder sb = new StringBuilder();
        sb.append("# Indian Traffic Sign & Road Safety Pipeline: Master Data Summary\n\n");
        sb.append("**Generated At:** ").append(java.time.Instant.now()).append("\n\n");
        sb.append("## 1. Indian Sign Classification Pipeline (IRC:67 85 Classes)\n");
        sb.append("- Detailed report: [`CLASSIFICATION_DATASET_REPORT.md`](CLASSIFICATION_DATASET_REPORT.md)\n");
        sb.append("- Machine-readable report: [`classification_validation_report.json`](classification_validation_report.json)\n\n");
        sb.append("## 2. Indian Road Driving Detection Pipeline (Generic Signs in Road Scenes)\n");
        sb.append("- Detailed report: [`TAR_INSPECTION_REPORT.md`](TAR_INSPECTION_REPORT.md)\n");
        sb.append("- Machine-readable report: [`tar_inspection_report.json`](tar_inspection_report.json)\n\n");
        sb.append("## 3. Separation of Responsibilities\n");
        sb.append("- **Classification Target:** 85-class IRC:67 sign recognition trained on pure cropped signboards.\n");
        sb.append("- **Detection Target:** Full-frame bounding box localization for `traffic sign` category.\n");
        sb.append("- **Road Safety Monitor:** Decision-support alerts combining detection localization with fine-grained classification.\n");

        Files.writeString(reportsDir.resolve("DATASET_PIPELINE_SUMMARY.md"), sb.toString());
    }

    private Map<String, String> parseArgs(String[] args) {
        Map<String, String> map = new HashMap<>();
        if (args == null) return map;
        for (int i = 1; i < args.length; i++) {
            if (args[i].startsWith("--") && i + 1 < args.length && !args[i + 1].startsWith("--")) {
                String key = args[i].substring(2);
                String val = args[i + 1];
                map.put(key, val);
                i++;
            }
        }
        return map;
    }
}
