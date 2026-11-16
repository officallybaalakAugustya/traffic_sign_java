package com.trafficsign.app;

import com.trafficsign.data.DatasetAcquisitionManager;
import com.trafficsign.data.DatasetValidator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Path;

/**
 * Command-line runner for downloading and safely extracting the official GTSRB archives
 * using parallel HTTP Range streams.
 */
public class DatasetDownloadCli {

    private static final Logger logger = LoggerFactory.getLogger(DatasetDownloadCli.class);

    public static void main(String[] args) {
        Path rawDir = Path.of("data", "raw", "GTSRB");
        int concurrency = 12;

        try {
            logger.info("============================================================");
            logger.info(" GTSRB Java Multi-Stream Acquisition Engine (Java 21)");
            logger.info(" Destination : {}", rawDir.toAbsolutePath());
            logger.info(" Concurrency : {} parallel HTTP range streams", concurrency);
            logger.info("============================================================");

            DatasetAcquisitionManager.acquireAndExtractAll(rawDir, concurrency);

            logger.info("Acquisition and extraction finished. Executing deep validation...");
            DatasetValidator.DetailedValidationReport report = DatasetValidator.validate(rawDir);

            logger.info("=== Dataset Validation Results ===");
            logger.info("  Total Training Samples : {}", report.totalTrainingSamples());
            logger.info("  Total Physical Tracks  : {}", report.totalTracks());
            logger.info("  Detected Classes       : {} / 43", report.detectedClasses());
            logger.info("  Official Test Samples  : {}", report.totalTestSamples());
            logger.info("  Aligned Test Labels    : {}", report.testLabelsAligned());
            logger.info("  Corrupt Images Count   : {}", report.corruptImagesCount());
            logger.info("  Duplicate Files Count  : {}", report.duplicateFilesCount());
            logger.info("  Inverted ROIs Count    : {}", report.invertedRoiCount());
            logger.info("  Is Dataset Healthy     : {}", report.isDatasetHealthy());

            if (!report.isDatasetHealthy()) {
                logger.warn("Anomalies detected: {}", report.anomalyMessages());
            }

        } catch (Exception e) {
            logger.error("Dataset acquisition failed: {}", e.getMessage(), e);
            System.exit(1);
        }
    }
}
