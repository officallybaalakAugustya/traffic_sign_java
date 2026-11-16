package com.trafficsign.safety;

import ai.djl.Device;
import com.trafficsign.inference.RoadSceneInferencePipeline;
import org.junit.jupiter.api.Test;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;

public class Task17DiagnosticsTest {

    @Test
    void testDiagnosticPrediction() throws Exception {
        Path detDir = Path.of("models/checkpoints/detector_task15_champion");
        Path clsDir = Path.of("models/checkpoints/task7_champion");

        try (RoadSceneInferencePipeline pipeline = new RoadSceneInferencePipeline(detDir, clsDir, Device.cpu())) {
            pipeline.setDetectionConfidenceThreshold(0.20f);
            pipeline.setClassificationConfidenceThreshold(0.20f);
            pipeline.setNmsIouThreshold(0.25);
            pipeline.setMaxProposalsPerFrame(25);

            File emptyRoadFile = new File("data/raw/indian-road-driving-sample/inspection/000db725-8180-4770-8b6a-4eb74aeda9f9_0004.jpg");
            if (emptyRoadFile.exists()) {
                BufferedImage emptyRoadImg = ImageIO.read(emptyRoadFile);
                RoadSceneInferencePipeline.PipelineResult resEmpty = pipeline.processImage(emptyRoadImg);
                System.out.println("=== 0004 Empty Road Detections (Det=0.20, Cls=0.20) ===");
                System.out.println("Total Detections: " + resEmpty.detections().size());
                for (var d : resEmpty.detections()) {
                    System.out.printf("Empty Det: classId=%d, name='%s', detConf=%.4f, clsConf=%.4f, state=%s, box=%s%n",
                            d.predictedClassId(), d.predictedClassName(), d.detectionConfidence(),
                            d.classificationConfidence(), d.uncertaintyState(), d.normalizedBox());
                }
            }

            File speed40File = new File("data/raw/traffic_sign_classification_dataset/test/SPEED_LIMIT_40/66000.jpg");
            if (speed40File.exists()) {
                BufferedImage speed40Img = ImageIO.read(speed40File);
                BufferedImage syntheticRoad = new BufferedImage(800, 600, BufferedImage.TYPE_INT_RGB);
                java.awt.Graphics2D g = syntheticRoad.createGraphics();
                g.setColor(new java.awt.Color(50, 70, 90));
                g.fillRect(0, 0, 800, 300);
                g.setColor(new java.awt.Color(70, 70, 70));
                g.fillRect(0, 300, 800, 300);
                g.drawImage(speed40Img, 400, 200, 100, 100, null);
                g.dispose();

                RoadSceneInferencePipeline.PipelineResult res = pipeline.processImage(syntheticRoad);
                System.out.println("=== Synthetic 40km/h Road Detections ===");
                System.out.println("Total Detections: " + res.detections().size());
                for (var d : res.detections()) {
                    System.out.printf("Speed40 Det: classId=%d, name='%s', detConf=%.4f, clsConf=%.4f, state=%s, box=%s%n",
                            d.predictedClassId(), d.predictedClassName(), d.detectionConfidence(),
                            d.classificationConfidence(), d.uncertaintyState(), d.normalizedBox());
                }
            }
        }
    }
}
