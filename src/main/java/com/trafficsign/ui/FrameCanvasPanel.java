package com.trafficsign.ui;

import com.trafficsign.data.BoundingBox;
import com.trafficsign.inference.RoadSceneInferencePipeline;
import com.trafficsign.safety.SafetyEvent;
import com.trafficsign.safety.SafetyState;

import javax.swing.JPanel;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.geom.RoundRectangle2D;
import java.awt.image.BufferedImage;
import java.util.List;

public class FrameCanvasPanel extends JPanel {

    private static final Color COLOR_REGULATORY = new Color(220, 20, 30);
    private static final Color COLOR_CAUTIONARY = new Color(245, 160, 0);
    private static final Color COLOR_INFORMATORY = new Color(0, 120, 230);
    private static final Color COLOR_UNCERTAIN = new Color(140, 140, 140);
    private static final Color COLOR_SELECTED = new Color(255, 0, 220);

    private BufferedImage currentFrame;
    private RoadSceneInferencePipeline.PipelineResult currentResult;
    private SafetyState safetyState;
    private SafetyEvent selectedEvent;
    private double currentFps = 0.0;
    private long lastTotalLatencyMs = 0;
    private int currentFrameIndex = 0;
    private String hardwareDeviceLabel = "CPU";

    public FrameCanvasPanel() {
        setBackground(new Color(24, 26, 32));
        setPreferredSize(new Dimension(800, 500));
    }

    public synchronized void updateFrame(
            int frameIndex,
            BufferedImage frame,
            RoadSceneInferencePipeline.PipelineResult result,
            SafetyState state,
            double fps,
            long latencyMs
    ) {
        this.currentFrameIndex = frameIndex;
        this.currentFrame = frame;
        this.currentResult = result;
        this.safetyState = state;
        this.currentFps = fps;
        this.lastTotalLatencyMs = latencyMs;
        repaint();
    }

    public synchronized void setSelectedEvent(SafetyEvent event) {
        this.selectedEvent = event;
        repaint();
    }

    public void setHardwareDeviceLabel(String label) {
        this.hardwareDeviceLabel = label;
        repaint();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g.create();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);

        int panelW = getWidth();
        int panelH = getHeight();

        if (currentFrame == null) {
            drawEmptyState(g2, panelW, panelH);
            g2.dispose();
            return;
        }

        int imgW = currentFrame.getWidth();
        int imgH = currentFrame.getHeight();

        double scale = Math.min((double) panelW / imgW, (double) panelH / imgH);
        int drawW = (int) Math.round(imgW * scale);
        int drawH = (int) Math.round(imgH * scale);
        int offsetX = (panelW - drawW) / 2;
        int offsetY = (panelH - drawH) / 2;

        g2.drawImage(currentFrame, offsetX, offsetY, drawW, drawH, null);

        if (currentResult != null && currentResult.detections() != null) {
            for (var det : currentResult.detections()) {
                drawDetectionBox(g2, det, offsetX, offsetY, scale, imgW, imgH);
            }
        }

        if (selectedEvent != null && selectedEvent.normalizedBox() != null) {
            drawSelectedHighlight(g2, selectedEvent.normalizedBox(), offsetX, offsetY, scale, imgW, imgH);
        }

        drawActiveSafetyHud(g2, panelW, panelH);
        drawTelemetryFooter(g2, panelW, panelH, imgW, imgH);

        g2.dispose();
    }

    private void drawEmptyState(Graphics2D g2, int w, int h) {
        g2.setColor(new Color(180, 185, 195));
        g2.setFont(new Font("Segoe UI", Font.BOLD, 18));
        String msg = "No Frame Loaded - Select Single Frame or Video Frame Sequence";
        FontMetrics fm = g2.getFontMetrics();
        g2.drawString(msg, (w - fm.stringWidth(msg)) / 2, h / 2);

        g2.setFont(new Font("Segoe UI", Font.PLAIN, 13));
        String sub = "Supported formats: JPG, PNG. IRC:67 Certified Academic Decision-Support System";
        FontMetrics fmSub = g2.getFontMetrics();
        g2.drawString(sub, (w - fmSub.stringWidth(sub)) / 2, (h / 2) + 30);
    }

    private void drawDetectionBox(
            Graphics2D g2,
            RoadSceneInferencePipeline.RoadSceneDetection det,
            int offsetX,
            int offsetY,
            double scale,
            int imgW,
            int imgH
    ) {
        BoundingBox norm = det.normalizedBox();
        int bx1 = offsetX + (int) Math.round(norm.x1() * imgW * scale);
        int by1 = offsetY + (int) Math.round(norm.y1() * imgH * scale);
        int bx2 = offsetX + (int) Math.round(norm.x2() * imgW * scale);
        int by2 = offsetY + (int) Math.round(norm.y2() * imgH * scale);
        int bw = Math.max(2, bx2 - bx1);
        int bh = Math.max(2, by2 - by1);

        Color boxColor = switch (det.category() != null ? det.category().toUpperCase() : "") {
            case "MANDATORY", "REGULATORY" -> COLOR_REGULATORY;
            case "CAUTIONARY" -> COLOR_CAUTIONARY;
            case "INFORMATORY" -> COLOR_INFORMATORY;
            default -> COLOR_UNCERTAIN;
        };

        if (det.uncertaintyState() != RoadSceneInferencePipeline.UncertaintyState.CONFIDENT) {
            boxColor = COLOR_UNCERTAIN;
            float[] dash = {6.0f, 4.0f};
            g2.setStroke(new BasicStroke(2.0f, BasicStroke.CAP_BUTT, BasicStroke.JOIN_MITER, 10.0f, dash, 0.0f));
        } else {
            g2.setStroke(new BasicStroke(2.5f));
        }

        g2.setColor(boxColor);
        g2.drawRect(bx1, by1, bw, bh);

        String labelText = String.format("%s (%.0f%%)", det.displayLabel(), det.classificationConfidence() * 100.0f);
        g2.setFont(new Font("Segoe UI", Font.BOLD, 11));
        FontMetrics fm = g2.getFontMetrics();
        int textW = fm.stringWidth(labelText);
        int textH = fm.getHeight();

        int pillX = bx1;
        int pillY = Math.max(offsetY, by1 - textH - 4);
        int pillW = textW + 10;
        int pillH = textH + 4;

        g2.setColor(new Color(20, 20, 25, 210));
        g2.fill(new RoundRectangle2D.Float(pillX, pillY, pillW, pillH, 6, 6));
        g2.setColor(boxColor);
        g2.draw(new RoundRectangle2D.Float(pillX, pillY, pillW, pillH, 6, 6));

        g2.setColor(Color.WHITE);
        g2.drawString(labelText, pillX + 5, pillY + fm.getAscent() + 1);
    }

    private void drawSelectedHighlight(
            Graphics2D g2,
            BoundingBox norm,
            int offsetX,
            int offsetY,
            double scale,
            int imgW,
            int imgH
    ) {
        int bx1 = offsetX + (int) Math.round(norm.x1() * imgW * scale);
        int by1 = offsetY + (int) Math.round(norm.y1() * imgH * scale);
        int bx2 = offsetX + (int) Math.round(norm.x2() * imgW * scale);
        int by2 = offsetY + (int) Math.round(norm.y2() * imgH * scale);
        int bw = Math.max(2, bx2 - bx1);
        int bh = Math.max(2, by2 - by1);

        g2.setStroke(new BasicStroke(3.5f));
        g2.setColor(COLOR_SELECTED);
        g2.drawRect(bx1 - 2, by1 - 2, bw + 4, bh + 4);
    }

    private void drawActiveSafetyHud(Graphics2D g2, int panelW, int panelH) {
        if (safetyState == null) {
            return;
        }

        int hudX = 16;
        int hudY = 16;
        Integer speedLimit = safetyState.getCurrentPostedSpeedLimit();

        int circleSize = 64;
        g2.setColor(Color.WHITE);
        g2.fillOval(hudX, hudY, circleSize, circleSize);
        g2.setColor(new Color(220, 20, 30));
        g2.setStroke(new BasicStroke(6.0f));
        g2.drawOval(hudX + 3, hudY + 3, circleSize - 6, circleSize - 6);

        if (speedLimit != null) {
            g2.setColor(Color.BLACK);
            g2.setFont(new Font("Segoe UI", Font.BOLD, 22));
            String slText = String.valueOf(speedLimit);
            FontMetrics fmSl = g2.getFontMetrics();
            g2.drawString(slText, hudX + (circleSize - fmSl.stringWidth(slText)) / 2, hudY + 36);

            g2.setFont(new Font("Segoe UI", Font.BOLD, 9));
            String unitText = "km/h";
            FontMetrics fmUnit = g2.getFontMetrics();
            g2.drawString(unitText, hudX + (circleSize - fmUnit.stringWidth(unitText)) / 2, hudY + 50);
        } else {
            g2.setColor(new Color(100, 100, 100));
            g2.setFont(new Font("Segoe UI", Font.BOLD, 12));
            String noneText = "DEFAULT";
            FontMetrics fmNone = g2.getFontMetrics();
            g2.drawString(noneText, hudX + (circleSize - fmNone.stringWidth(noneText)) / 2, hudY + 36);
        }

        int badgeY = hudY + circleSize + 10;
        for (String restriction : safetyState.getActiveRestrictions()) {
            g2.setFont(new Font("Segoe UI", Font.BOLD, 11));
            FontMetrics fmRes = g2.getFontMetrics();
            int rW = fmRes.stringWidth(restriction) + 16;
            int rH = 22;

            g2.setColor(new Color(200, 30, 30, 220));
            g2.fill(new RoundRectangle2D.Float(hudX, badgeY, rW, rH, 6, 6));
            g2.setColor(Color.WHITE);
            g2.drawString(restriction, hudX + 8, badgeY + fmRes.getAscent() + 2);
            badgeY += rH + 6;
        }

        List<String> hazards = safetyState.getActiveHazards();
        if (!hazards.isEmpty()) {
            String primaryHazard = hazards.get(hazards.size() - 1);
            String bannerText = "HAZARD ADVISORY: " + primaryHazard.toUpperCase();
            g2.setFont(new Font("Segoe UI", Font.BOLD, 13));
            FontMetrics fmH = g2.getFontMetrics();
            int bW = fmH.stringWidth(bannerText) + 24;
            int bH = 28;
            int bX = (panelW - bW) / 2;
            int bY = 16;

            g2.setColor(new Color(230, 140, 0, 230));
            g2.fill(new RoundRectangle2D.Float(bX, bY, bW, bH, 8, 8));
            g2.setColor(Color.BLACK);
            g2.drawString(bannerText, bX + 12, bY + fmH.getAscent() + 5);
        }
    }

    private void drawTelemetryFooter(Graphics2D g2, int panelW, int panelH, int imgW, int imgH) {
        int footH = 26;
        int footY = panelH - footH - 4;
        int footX = 10;
        int footW = panelW - 20;

        g2.setColor(new Color(15, 17, 22, 220));
        g2.fill(new RoundRectangle2D.Float(footX, footY, footW, footH, 6, 6));

        g2.setColor(new Color(210, 215, 225));
        g2.setFont(new Font("Consolas", Font.PLAIN, 12));
        FontMetrics fm = g2.getFontMetrics();

        long detMs = currentResult != null ? currentResult.detectionLatencyMs() : 0;
        long clsMs = currentResult != null ? currentResult.classificationLatencyMs() : 0;
        int detsCount = currentResult != null && currentResult.detections() != null ? currentResult.detections().size() : 0;

        String telemetry = String.format(
                " Frame: #%04d (%dx%d) | FPS: %.1f | Det: %d ms | Cls: %d ms | Total: %d ms | Detections: %d | Device: %s",
                currentFrameIndex, imgW, imgH, currentFps, detMs, clsMs, lastTotalLatencyMs, detsCount, hardwareDeviceLabel
        );

        g2.drawString(telemetry, footX + 10, footY + fm.getAscent() + 4);
    }
}
