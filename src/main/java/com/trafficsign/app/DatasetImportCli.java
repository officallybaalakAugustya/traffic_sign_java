package com.trafficsign.app;

import com.trafficsign.data.DatasetImportManager;

import java.nio.file.Path;
import java.util.Arrays;

public final class DatasetImportCli {
    private DatasetImportCli() {
    }

    public static void main(String[] args) {
        try {
            if (args.length == 0) {
                throw new IllegalArgumentException("Usage: audit --input <path> | plan --input <path> | import --plan <path> [--confirm]");
            }
            String command = args[0];
            if ("audit".equals(command) || "plan".equals(command)) {
                Path input = valueAfter(args, "--input");
                DatasetImportManager.Report report = DatasetImportManager.audit(Path.of("."), input);
                try {
                    DatasetImportManager.writeReports(Path.of("."), report, "plan".equals(command));
                } catch (Exception e) {
                    System.err.println("Reports could not be persisted: " + e.getClass().getSimpleName() + ": " + e.getMessage());
                }
                System.out.println(report.markdown());
                return;
            }
            if ("import".equals(command)) {
                Path plan = valueAfter(args, "--plan");
                if (!Arrays.asList(args).contains("--confirm")) {
                    throw new IllegalArgumentException("Import requires explicit --confirm");
                }
                DatasetImportManager.ImportResult result = DatasetImportManager.importPlan(Path.of("."), plan);
                System.out.println(result.markdown());
                return;
            }
            throw new IllegalArgumentException("Unknown command: " + command);
        } catch (Exception e) {
            System.err.println("Dataset import failed: " + e.getMessage());
            System.exit(2);
        }
    }

    private static Path valueAfter(String[] args, String option) {
        for (int i = 0; i + 1 < args.length; i++) {
            if (option.equals(args[i])) {
                return Path.of(args[i + 1]);
            }
        }
        throw new IllegalArgumentException("Missing " + option);
    }
}
