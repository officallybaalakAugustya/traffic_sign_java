package com.trafficsign.app;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {
    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    public static final String APP_NAME = "Traffic Sign Recognition and Road Safety Monitor";
    public static final String APP_VERSION = "1.0.0-SNAPSHOT";

    public static void main(String[] args) {
        if (args != null && args.length > 0) {
            if ("gui".equalsIgnoreCase(args[0]) || "monitor".equalsIgnoreCase(args[0])) {
                com.trafficsign.ui.RoadSafetyMonitorApp.main(new String[0]);
                return;
            }
            if ("demo".equalsIgnoreCase(args[0]) || "demonstrate".equalsIgnoreCase(args[0])) {
                RoadSafetyMonitorDemonstrator.main(new String[0]);
                return;
            }
            if ("audit".equals(args[0]) || "plan".equals(args[0]) || "import".equals(args[0])) {
                DatasetImportCli.main(args);
                return;
            }
            IndianRoadPipelineCli.main(args);
            return;
        }
        logger.info("Initializing {} v{}", APP_NAME, APP_VERSION);
        logger.info("Java Runtime Version: {}", System.getProperty("java.version"));
        logger.info("Current Milestone: Stage 1 (Indian Road Real-Data Pipeline)");
        System.out.println("Traffic Sign Recognition and Road Safety Monitor v" + APP_VERSION);
        System.out.println("Available commands:");
        System.out.println("  gui / monitor         : Launch Desktop Road Safety Decision-Support Monitor (IRC:67)");
        System.out.println("  demo / demonstrate    : Execute multi-scenario road safety demonstration");
        System.out.println("  validate-classification / inspect-tar / dataset-summary : Dataset audit tools");
    }
}
