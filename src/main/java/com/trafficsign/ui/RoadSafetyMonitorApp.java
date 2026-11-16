package com.trafficsign.ui;

import ai.djl.Device;
import com.trafficsign.data.BoundingBox;
import com.trafficsign.inference.RoadSceneInferencePipeline;
import com.trafficsign.safety.SafetyEvent;
import com.trafficsign.safety.SafetyEventEngine;
import com.trafficsign.safety.SafetyState;
import com.trafficsign.safety.SignRuleCatalogue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.ImageIO;
import javax.swing.BorderFactory;
import javax.swing.Box;
import javax.swing.BoxLayout;
import javax.swing.ImageIcon;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSlider;
import javax.swing.JSplitPane;
import javax.swing.JTable;
import javax.swing.JTextField;
import javax.swing.JToolBar;
import javax.swing.ListSelectionModel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import javax.swing.border.EmptyBorder;
import javax.swing.border.TitledBorder;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.FlowLayout;
import java.awt.Font;
import java.awt.Graphics2D;
import java.awt.GridLayout;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.event.ActionEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;

public class RoadSafetyMonitorApp extends JFrame {

    private static final Logger logger = LoggerFactory.getLogger(RoadSafetyMonitorApp.class);

    private final SignRuleCatalogue catalogue;
    private final SafetyEventEngine safetyEngine;
    private final SafetyEventTableModel tableModel;
    private RoadSceneInferencePipeline pipeline;

    private final FrameCanvasPanel canvasPanel;
    private final JTable eventTable;
    private final JLabel previewImageLabel;
    private final JLabel signTitleLabel;
    private final JLabel signCategoryLabel;
    private final JLabel signAdvisoryLabel;
    private final JLabel signVerificationLabel;
    private final JComboBox<SafetyEvent.OperatorStatus> operatorStatusCombo;
    private final JTextField operatorNoteField;

    private final JTextField detectorPathField;
    private final JTextField classifierPathField;
    private final JSlider detConfidenceSlider;
    private final JSlider clsConfidenceSlider;
    private final JSlider nmsIouSlider;
    private final JSlider maxProposalsSlider;
    private final JLabel detConfValueLabel;
    private final JLabel clsConfValueLabel;
    private final JLabel nmsIouValueLabel;
    private final JLabel proposalCapValueLabel;

    private final JButton playPauseButton;
    private final JButton stepButton;
    private final JButton restartButton;
    private final JComboBox<String> speedComboBox;
    private final JCheckBox loopCheckBox;

    private final List<Path> frameSequencePaths = new ArrayList<>();
    private int currentFrameIndex = 0;
    private Timer playbackTimer;
    private final AtomicBoolean isPlaying = new AtomicBoolean(false);
    private final AtomicBoolean isProcessing = new AtomicBoolean(false);

    public RoadSafetyMonitorApp() {
        super("Indian Traffic Sign Recognition and Road Safety Decision-Support Monitor (IRC:67)");
        this.catalogue = new SignRuleCatalogue();
        this.safetyEngine = new SafetyEventEngine(this.catalogue, new SafetyState());
        this.tableModel = new SafetyEventTableModel();
        this.canvasPanel = new FrameCanvasPanel();
        this.eventTable = new JTable(tableModel);

        this.previewImageLabel = new JLabel();
        this.signTitleLabel = new JLabel("No Sign Selected");
        this.signCategoryLabel = new JLabel("Category: -");
        this.signAdvisoryLabel = new JLabel("Advisory: -");
        this.signVerificationLabel = new JLabel("Status: -");
        this.operatorStatusCombo = new JComboBox<>(SafetyEvent.OperatorStatus.values());
        this.operatorNoteField = new JTextField(16);

        this.detectorPathField = new JTextField("models/checkpoints/detector_task15_champion", 20);
        this.classifierPathField = new JTextField("models/checkpoints/task7_champion", 20);
        this.detConfidenceSlider = new JSlider(5, 95, 20);
        this.clsConfidenceSlider = new JSlider(5, 95, 25);
        this.nmsIouSlider = new JSlider(5, 90, 25);
        this.maxProposalsSlider = new JSlider(5, 60, 25);

        this.detConfValueLabel = new JLabel("0.20");
        this.clsConfValueLabel = new JLabel("0.25");
        this.nmsIouValueLabel = new JLabel("0.25");
        this.proposalCapValueLabel = new JLabel("25");

        detConfValueLabel.setFont(new Font("Consolas", Font.BOLD, 12));
        clsConfValueLabel.setFont(new Font("Consolas", Font.BOLD, 12));
        nmsIouValueLabel.setFont(new Font("Consolas", Font.BOLD, 12));
        proposalCapValueLabel.setFont(new Font("Consolas", Font.BOLD, 12));

        this.safetyEngine.setMinDetectionConfidence(detConfidenceSlider.getValue() / 100.0f);
        this.safetyEngine.setMinClassificationConfidence(clsConfidenceSlider.getValue() / 100.0f);
        this.safetyEngine.setTrackIouThreshold(nmsIouSlider.getValue() / 100.0);

        this.playPauseButton = new JButton("Play");
        this.stepButton = new JButton("Step Forward");
        this.restartButton = new JButton("Restart");
        this.speedComboBox = new JComboBox<>(new String[]{"1.0x Realtime", "2.0x Fast", "Max Speed (Analysis)"});
        this.loopCheckBox = new JCheckBox("Loop Playback", true);

        initPipelineSafely();
        initUiLayout();
        initSafetyListeners();
        initPlaybackTimer();
    }

    private void initPipelineSafely() {
        try {
            Path detDir = Path.of(detectorPathField.getText().trim());
            Path clsDir = Path.of(classifierPathField.getText().trim());
            if (Files.exists(detDir) && Files.exists(clsDir)) {
                this.pipeline = new RoadSceneInferencePipeline(detDir, clsDir, Device.cpu());
                this.pipeline.setDetectionConfidenceThreshold(detConfidenceSlider.getValue() / 100.0f);
                this.pipeline.setClassificationConfidenceThreshold(clsConfidenceSlider.getValue() / 100.0f);
                this.pipeline.setNmsIouThreshold(nmsIouSlider.getValue() / 100.0);
                this.pipeline.setMaxProposalsPerFrame(maxProposalsSlider.getValue());
                canvasPanel.setHardwareDeviceLabel("PyTorch (CPU)");
                logger.info("Pipeline initialized with detector: {} and classifier: {}", detDir, clsDir);
            } else {
                logger.warn("Checkpoint directories not found. Application running in standby mode.");
                canvasPanel.setHardwareDeviceLabel("Standby (No Weights Loaded)");
            }
        } catch (Exception e) {
            logger.warn("Pipeline initialization warning: {}", e.getMessage());
            canvasPanel.setHardwareDeviceLabel("Standby (Error loading weights)");
        }
    }

    private void initUiLayout() {
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setMinimumSize(new Dimension(1280, 800));
        setLayout(new BorderLayout());

        JPanel headerPanel = buildDisclaimerBanner();
        add(headerPanel, BorderLayout.NORTH);

        JToolBar mainToolBar = buildControlToolBar();

        JPanel centerPanel = new JPanel(new BorderLayout());
        centerPanel.add(mainToolBar, BorderLayout.NORTH);

        JSplitPane mainSplitPane = new JSplitPane(JSplitPane.HORIZONTAL_SPLIT);
        mainSplitPane.setResizeWeight(0.60);
        mainSplitPane.setLeftComponent(canvasPanel);

        JPanel rightPane = buildRightInspectionPanel();
        mainSplitPane.setRightComponent(rightPane);

        centerPanel.add(mainSplitPane, BorderLayout.CENTER);
        add(centerPanel, BorderLayout.CENTER);

        JPanel bottomSettingsPanel = buildSettingsPanel();
        add(bottomSettingsPanel, BorderLayout.SOUTH);
    }

    private JPanel buildDisclaimerBanner() {
        JPanel banner = new JPanel(new FlowLayout(FlowLayout.CENTER, 10, 8));
        banner.setBackground(new Color(170, 20, 20));
        JLabel noticeLabel = new JLabel("ACADEMIC DECISION-SUPPORT PROTOTYPE ONLY (IRC:67) - NOT CERTIFIED FOR AUTONOMOUS VEHICULAR CONTROL");
        noticeLabel.setForeground(Color.WHITE);
        noticeLabel.setFont(new Font("Segoe UI", Font.BOLD, 13));
        banner.add(noticeLabel);
        return banner;
    }

    private JToolBar buildControlToolBar() {
        JToolBar toolBar = new JToolBar();
        toolBar.setFloatable(false);
        toolBar.setLayout(new FlowLayout(FlowLayout.LEFT, 8, 4));

        JButton openImageBtn = new JButton("Open Single Frame");
        openImageBtn.addActionListener(this::handleOpenSingleImage);
        toolBar.add(openImageBtn);

        JButton openFolderBtn = new JButton("Open Sequence Folder");
        openFolderBtn.addActionListener(this::handleOpenSequenceFolder);
        toolBar.add(openFolderBtn);

        toolBar.addSeparator();

        playPauseButton.addActionListener(e -> togglePlayPause());
        toolBar.add(playPauseButton);

        stepButton.addActionListener(e -> stepForward());
        toolBar.add(stepButton);

        restartButton.addActionListener(e -> restartSequence());
        toolBar.add(restartButton);

        toolBar.addSeparator();
        toolBar.add(new JLabel("Speed: "));
        speedComboBox.addActionListener(e -> updatePlaybackSpeed());
        toolBar.add(speedComboBox);

        toolBar.add(loopCheckBox);

        return toolBar;
    }

    private JPanel buildRightInspectionPanel() {
        JPanel panel = new JPanel(new BorderLayout(5, 5));
        panel.setBorder(new EmptyBorder(5, 5, 5, 5));

        JPanel inspectionCard = new JPanel(new BorderLayout(10, 10));
        inspectionCard.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createLineBorder(new Color(80, 90, 110)),
                "Selected Sign Observation & Operator Review",
                TitledBorder.LEADING,
                TitledBorder.TOP,
                new Font("Segoe UI", Font.BOLD, 12)
        ));

        previewImageLabel.setPreferredSize(new Dimension(110, 110));
        previewImageLabel.setHorizontalAlignment(JLabel.CENTER);
        previewImageLabel.setBorder(BorderFactory.createLineBorder(new Color(60, 65, 75)));
        inspectionCard.add(previewImageLabel, BorderLayout.WEST);

        JPanel detailsPanel = new JPanel();
        detailsPanel.setLayout(new BoxLayout(detailsPanel, BoxLayout.Y_AXIS));

        signTitleLabel.setFont(new Font("Segoe UI", Font.BOLD, 14));
        signCategoryLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));
        signVerificationLabel.setFont(new Font("Segoe UI", Font.ITALIC, 11));
        signAdvisoryLabel.setFont(new Font("Segoe UI", Font.PLAIN, 12));

        detailsPanel.add(signTitleLabel);
        detailsPanel.add(Box.createVerticalStrut(4));
        detailsPanel.add(signCategoryLabel);
        detailsPanel.add(Box.createVerticalStrut(4));
        detailsPanel.add(signVerificationLabel);
        detailsPanel.add(Box.createVerticalStrut(6));
        detailsPanel.add(signAdvisoryLabel);
        detailsPanel.add(Box.createVerticalStrut(8));

        JPanel reviewControls = new JPanel(new FlowLayout(FlowLayout.LEFT, 5, 2));
        reviewControls.add(new JLabel("Action:"));
        reviewControls.add(operatorStatusCombo);
        reviewControls.add(new JLabel("Note:"));
        reviewControls.add(operatorNoteField);

        JButton saveReviewBtn = new JButton("Save Review");
        saveReviewBtn.addActionListener(this::handleSaveReview);
        reviewControls.add(saveReviewBtn);

        detailsPanel.add(reviewControls);
        inspectionCard.add(detailsPanel, BorderLayout.CENTER);

        JPanel tablePanel = new JPanel(new BorderLayout(4, 4));
        tablePanel.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createLineBorder(new Color(80, 90, 110)),
                "Live Road-Safety Event Log (Deduplicated)",
                TitledBorder.LEADING,
                TitledBorder.TOP,
                new Font("Segoe UI", Font.BOLD, 12)
        ));

        JPanel filterBar = new JPanel(new FlowLayout(FlowLayout.LEFT, 8, 4));
        filterBar.add(new JLabel("Severity:"));
        JComboBox<String> sevFilter = new JComboBox<>(new String[]{"ALL", "CRITICAL", "WARNING", "INFO"});
        sevFilter.addActionListener(e -> {
            String sel = (String) sevFilter.getSelectedItem();
            tableModel.setSeverityFilter("ALL".equals(sel) ? null : SignRuleCatalogue.SeverityLevel.valueOf(sel));
        });
        filterBar.add(sevFilter);

        filterBar.add(new JLabel("Category:"));
        JComboBox<String> catFilter = new JComboBox<>(new String[]{"ALL", "REGULATORY", "CAUTIONARY", "INFORMATORY"});
        catFilter.addActionListener(e -> {
            String sel = (String) catFilter.getSelectedItem();
            tableModel.setCategoryFilter("ALL".equals(sel) ? null : SignRuleCatalogue.RuleCategory.valueOf(sel));
        });
        filterBar.add(catFilter);

        JButton exportCsvBtn = new JButton("Export CSV");
        exportCsvBtn.addActionListener(e -> handleExportCsv());
        filterBar.add(exportCsvBtn);

        JButton exportJsonBtn = new JButton("Export JSON");
        exportJsonBtn.addActionListener(e -> handleExportJson());
        filterBar.add(exportJsonBtn);

        JButton clearBtn = new JButton("Clear");
        clearBtn.addActionListener(e -> {
            safetyEngine.clear();
            tableModel.clear();
            canvasPanel.repaint();
        });
        filterBar.add(clearBtn);

        tablePanel.add(filterBar, BorderLayout.NORTH);

        eventTable.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        eventTable.getSelectionModel().addListSelectionListener(e -> {
            if (!e.getValueIsAdjusting()) {
                handleTableSelectionChanged();
            }
        });
        JScrollPane scrollPane = new JScrollPane(eventTable);
        tablePanel.add(scrollPane, BorderLayout.CENTER);

        panel.add(inspectionCard, BorderLayout.NORTH);
        panel.add(tablePanel, BorderLayout.CENTER);

        return panel;
    }

    private JPanel buildSettingsPanel() {
        JPanel settings = new JPanel(new GridLayout(2, 1, 4, 4));
        settings.setBorder(BorderFactory.createTitledBorder("Model Checkpoints & Tunable Operating Parameters"));

        JPanel row1 = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 4));
        row1.add(new JLabel("Detector:"));
        row1.add(detectorPathField);
        JButton browseDet = new JButton("Browse");
        browseDet.addActionListener(e -> chooseDirectory(detectorPathField));
        row1.add(browseDet);

        row1.add(new JLabel("Classifier:"));
        row1.add(classifierPathField);
        JButton browseCls = new JButton("Browse");
        browseCls.addActionListener(e -> chooseDirectory(classifierPathField));
        row1.add(browseCls);

        JButton reloadBtn = new JButton("Reload Models");
        reloadBtn.addActionListener(e -> handleReloadModels());
        row1.add(reloadBtn);

        JPanel row2 = new JPanel(new FlowLayout(FlowLayout.LEFT, 10, 4));
        row2.add(new JLabel("Det Conf:"));
        detConfidenceSlider.setPreferredSize(new Dimension(85, 20));
        detConfidenceSlider.addChangeListener(e -> {
            float val = detConfidenceSlider.getValue() / 100.0f;
            detConfValueLabel.setText(String.format("%.2f", val));
            if (pipeline != null) {
                pipeline.setDetectionConfidenceThreshold(val);
            }
            safetyEngine.setMinDetectionConfidence(val);
        });
        row2.add(detConfidenceSlider);
        row2.add(detConfValueLabel);

        row2.add(new JLabel("Cls Conf:"));
        clsConfidenceSlider.setPreferredSize(new Dimension(85, 20));
        clsConfidenceSlider.addChangeListener(e -> {
            float val = clsConfidenceSlider.getValue() / 100.0f;
            clsConfValueLabel.setText(String.format("%.2f", val));
            if (pipeline != null) {
                pipeline.setClassificationConfidenceThreshold(val);
            }
            safetyEngine.setMinClassificationConfidence(val);
        });
        row2.add(clsConfidenceSlider);
        row2.add(clsConfValueLabel);

        row2.add(new JLabel("NMS IoU:"));
        nmsIouSlider.setPreferredSize(new Dimension(85, 20));
        nmsIouSlider.addChangeListener(e -> {
            double val = nmsIouSlider.getValue() / 100.0;
            nmsIouValueLabel.setText(String.format("%.2f", val));
            if (pipeline != null) {
                pipeline.setNmsIouThreshold(val);
            }
            safetyEngine.setTrackIouThreshold(val);
        });
        row2.add(nmsIouSlider);
        row2.add(nmsIouValueLabel);

        row2.add(new JLabel("Proposal Cap:"));
        maxProposalsSlider.setPreferredSize(new Dimension(85, 20));
        maxProposalsSlider.addChangeListener(e -> {
            int val = maxProposalsSlider.getValue();
            proposalCapValueLabel.setText(String.valueOf(val));
            if (pipeline != null) {
                pipeline.setMaxProposalsPerFrame(val);
            }
        });
        row2.add(maxProposalsSlider);
        row2.add(proposalCapValueLabel);

        settings.add(row1);
        settings.add(row2);
        return settings;
    }

    private void chooseDirectory(JTextField targetField) {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            targetField.setText(chooser.getSelectedFile().getAbsolutePath());
        }
    }

    private void initSafetyListeners() {
        safetyEngine.addListener(new SafetyEventEngine.SafetyEventListener() {
            @Override
            public void onSafetyEvent(SafetyEvent event) {
                SwingUtilities.invokeLater(() -> tableModel.addEvent(event));
            }

            @Override
            public void onSafetyStateUpdated(SafetyState state) {
                SwingUtilities.invokeLater(canvasPanel::repaint);
            }
        });
    }

    private void initPlaybackTimer() {
        playbackTimer = new Timer(200, (ActionEvent e) -> {
            if (isPlaying.get() && !isProcessing.get()) {
                processNextSequenceFrame();
            }
        });
    }

    private void updatePlaybackSpeed() {
        int idx = speedComboBox.getSelectedIndex();
        int delay = switch (idx) {
            case 1 -> 80;
            case 2 -> 5;
            default -> 200;
        };
        playbackTimer.setDelay(delay);
    }

    private void togglePlayPause() {
        if (frameSequencePaths.isEmpty()) {
            JOptionPane.showMessageDialog(this, "Please load a frame sequence directory first.", "Notice", JOptionPane.INFORMATION_MESSAGE);
            return;
        }
        if (isPlaying.get()) {
            isPlaying.set(false);
            playbackTimer.stop();
            playPauseButton.setText("Play");
        } else {
            isPlaying.set(true);
            playbackTimer.start();
            playPauseButton.setText("Pause");
        }
    }

    private void stepForward() {
        if (isPlaying.get()) {
            togglePlayPause();
        }
        processNextSequenceFrame();
    }

    private void restartSequence() {
        currentFrameIndex = 0;
        safetyEngine.clear();
        tableModel.clear();
        canvasPanel.setSelectedEvent(null);
        if (!frameSequencePaths.isEmpty()) {
            processFrameAtPath(frameSequencePaths.get(0), 0);
        }
    }

    private void processNextSequenceFrame() {
        if (frameSequencePaths.isEmpty()) {
            return;
        }
        if (currentFrameIndex >= frameSequencePaths.size()) {
            if (loopCheckBox.isSelected()) {
                currentFrameIndex = 0;
            } else {
                isPlaying.set(false);
                playbackTimer.stop();
                playPauseButton.setText("Play");
                return;
            }
        }
        Path path = frameSequencePaths.get(currentFrameIndex);
        int frameIdx = currentFrameIndex;
        currentFrameIndex++;
        processFrameAtPath(path, frameIdx);
    }

    public void processFrameAtPath(Path imagePath, int frameIdx) {
        if (imagePath == null || !Files.exists(imagePath)) {
            return;
        }
        isProcessing.set(true);
        new Thread(() -> {
            try {
                BufferedImage img = ImageIO.read(imagePath.toFile());
                if (img != null) {
                    processSingleFrame(img, frameIdx);
                }
            } catch (Exception e) {
                logger.error("Error reading frame: " + imagePath, e);
            } finally {
                isProcessing.set(false);
            }
        }).start();
    }

    public synchronized void processSingleFrame(BufferedImage frame, int frameIdx) {
        long t0 = System.currentTimeMillis();
        RoadSceneInferencePipeline.PipelineResult result = null;
        if (pipeline != null) {
            result = pipeline.processImage(frame);
        } else {
            result = new RoadSceneInferencePipeline.PipelineResult(frame.getWidth(), frame.getHeight(), 10, 10, List.of());
        }
        long latency = System.currentTimeMillis() - t0;
        double fps = latency > 0 ? (1000.0 / latency) : 0.0;

        safetyEngine.processFrame(frameIdx, System.currentTimeMillis(), frame, result);

        RoadSceneInferencePipeline.PipelineResult finalResult = result;
        SwingUtilities.invokeLater(() -> {
            canvasPanel.updateFrame(frameIdx, frame, finalResult, safetyEngine.getSafetyState(), fps, latency);
        });
    }

    private void handleOpenSingleImage(ActionEvent e) {
        JFileChooser chooser = new JFileChooser();
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            Path file = chooser.getSelectedFile().toPath();
            frameSequencePaths.clear();
            frameSequencePaths.add(file);
            currentFrameIndex = 0;
            safetyEngine.clear();
            tableModel.clear();
            canvasPanel.setSelectedEvent(null);
            processFrameAtPath(file, 0);
        }
    }

    private void handleOpenSequenceFolder(ActionEvent e) {
        JFileChooser chooser = new JFileChooser();
        chooser.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
        if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            File dir = chooser.getSelectedFile();
            loadSequenceFromFolder(dir.toPath());
        }
    }

    public void loadSequenceFromFolder(Path folderPath) {
        try {
            List<Path> files = Files.list(folderPath)
                    .filter(p -> {
                        String name = p.getFileName().toString().toLowerCase();
                        return name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".png");
                    })
                    .sorted(Comparator.comparing(Path::getFileName))
                    .toList();

            frameSequencePaths.clear();
            frameSequencePaths.addAll(files);
            currentFrameIndex = 0;
            safetyEngine.clear();
            tableModel.clear();
            canvasPanel.setSelectedEvent(null);

            if (!frameSequencePaths.isEmpty()) {
                processFrameAtPath(frameSequencePaths.get(0), 0);
                JOptionPane.showMessageDialog(this, "Loaded " + frameSequencePaths.size() + " frames into sequence player.", "Folder Loaded", JOptionPane.INFORMATION_MESSAGE);
            } else {
                JOptionPane.showMessageDialog(this, "No image files (.jpg, .png) found in selected directory.", "Empty Directory", JOptionPane.WARNING_MESSAGE);
            }
        } catch (Exception ex) {
            logger.error("Failed to load sequence folder", ex);
            JOptionPane.showMessageDialog(this, "Error loading directory: " + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
        }
    }

    private void handleTableSelectionChanged() {
        int selectedRow = eventTable.getSelectedRow();
        if (selectedRow < 0) {
            return;
        }
        SafetyEvent event = tableModel.getEventAt(selectedRow);
        setSelectedEvent(event);
    }

    public void setSelectedEvent(SafetyEvent event) {
        if (event == null) {
            return;
        }
        canvasPanel.setSelectedEvent(event);
        signTitleLabel.setText(event.displayName() + " [" + event.canonicalLabel() + "]");
        signCategoryLabel.setText("Category: " + event.category() + " | Severity: " + event.severity() + " | Conf: " + String.format("%.1f%%", event.classificationConfidence() * 100.0f));
        signVerificationLabel.setText("Standard: IRC:67 Code of Practice for Road Signs");
        signAdvisoryLabel.setText("Advisory: " + event.driverAdvisory());
        operatorStatusCombo.setSelectedItem(event.operatorStatus());
        operatorNoteField.setText(event.operatorNote() != null ? event.operatorNote() : "");

        String evidencePath = event.evidenceImagePath();
        if (evidencePath != null && !evidencePath.isBlank() && Files.exists(Path.of(evidencePath))) {
            try {
                BufferedImage crop = ImageIO.read(new File(evidencePath));
                if (crop != null) {
                    Image scaled = crop.getScaledInstance(100, 100, Image.SCALE_SMOOTH);
                    previewImageLabel.setIcon(new ImageIcon(scaled));
                    previewImageLabel.setText("");
                }
            } catch (Exception ignored) {
                previewImageLabel.setIcon(null);
                previewImageLabel.setText("Crop N/A");
            }
        } else {
            previewImageLabel.setIcon(null);
            previewImageLabel.setText("No Crop");
        }
    }

    private void handleSaveReview(ActionEvent e) {
        int selectedRow = eventTable.getSelectedRow();
        if (selectedRow < 0) {
            return;
        }
        SafetyEvent event = tableModel.getEventAt(selectedRow);
        if (event == null) {
            return;
        }
        SafetyEvent.OperatorStatus status = (SafetyEvent.OperatorStatus) operatorStatusCombo.getSelectedItem();
        String note = operatorNoteField.getText().trim();
        safetyEngine.updateOperatorReview(event.eventId(), status, note);
        SafetyEvent updated = event.withOperatorReview(status, note);
        tableModel.updateEvent(updated);
        JOptionPane.showMessageDialog(this, "Review saved for event " + event.eventId(), "Saved", JOptionPane.INFORMATION_MESSAGE);
    }

    private void handleExportCsv() {
        JFileChooser chooser = new JFileChooser();
        chooser.setSelectedFile(new File("reports/safety_events_export.csv"));
        if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                safetyEngine.exportEventsToCsv(chooser.getSelectedFile().toPath());
                JOptionPane.showMessageDialog(this, "Events successfully exported to CSV.", "Export Complete", JOptionPane.INFORMATION_MESSAGE);
            } catch (IOException ex) {
                JOptionPane.showMessageDialog(this, "Export failed: " + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private void handleExportJson() {
        JFileChooser chooser = new JFileChooser();
        chooser.setSelectedFile(new File("reports/safety_events_export.json"));
        if (chooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                safetyEngine.exportEventsToJson(chooser.getSelectedFile().toPath());
                JOptionPane.showMessageDialog(this, "Events successfully exported to JSON.", "Export Complete", JOptionPane.INFORMATION_MESSAGE);
            } catch (IOException ex) {
                JOptionPane.showMessageDialog(this, "Export failed: " + ex.getMessage(), "Error", JOptionPane.ERROR_MESSAGE);
            }
        }
    }

    private void handleReloadModels() {
        if (pipeline != null) {
            try {
                pipeline.close();
            } catch (Exception ignored) {
            }
            pipeline = null;
        }
        initPipelineSafely();
        JOptionPane.showMessageDialog(this, "Models reloaded successfully.", "Reload", JOptionPane.INFORMATION_MESSAGE);
    }

    public BufferedImage captureUiSnapshot(int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g2 = image.createGraphics();
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        this.addNotify();
        this.setSize(width, height);
        this.setPreferredSize(new Dimension(width, height));
        this.pack();
        this.setSize(width, height);
        this.getContentPane().setSize(width, height);
        this.getContentPane().doLayout();
        this.getContentPane().validate();
        this.getContentPane().printAll(g2);
        g2.dispose();
        return image;
    }

    public SafetyEventEngine getSafetyEngine() {
        return safetyEngine;
    }

    public FrameCanvasPanel getCanvasPanel() {
        return canvasPanel;
    }

    public SafetyEventTableModel getTableModel() {
        return tableModel;
    }

    public void setPipeline(RoadSceneInferencePipeline pipeline) {
        this.pipeline = pipeline;
        if (pipeline != null) {
            float detConf = detConfidenceSlider.getValue() / 100.0f;
            float clsConf = clsConfidenceSlider.getValue() / 100.0f;
            double nms = nmsIouSlider.getValue() / 100.0;
            int cap = maxProposalsSlider.getValue();

            pipeline.setDetectionConfidenceThreshold(detConf);
            pipeline.setClassificationConfidenceThreshold(clsConf);
            pipeline.setNmsIouThreshold(nms);
            pipeline.setMaxProposalsPerFrame(cap);

            safetyEngine.setMinDetectionConfidence(detConf);
            safetyEngine.setMinClassificationConfidence(clsConf);
            safetyEngine.setTrackIouThreshold(nms);
        }
    }

    public JSlider getDetConfidenceSlider() {
        return detConfidenceSlider;
    }

    public JSlider getClsConfidenceSlider() {
        return clsConfidenceSlider;
    }

    public JSlider getNmsIouSlider() {
        return nmsIouSlider;
    }

    public JSlider getMaxProposalsSlider() {
        return maxProposalsSlider;
    }

    public JLabel getDetConfValueLabel() {
        return detConfValueLabel;
    }

    public JLabel getClsConfValueLabel() {
        return clsConfValueLabel;
    }

    public JLabel getNmsIouValueLabel() {
        return nmsIouValueLabel;
    }

    public JLabel getProposalCapValueLabel() {
        return proposalCapValueLabel;
    }

    public RoadSceneInferencePipeline getPipeline() {
        return pipeline;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            RoadSafetyMonitorApp app = new RoadSafetyMonitorApp();
            app.pack();
            app.setLocationRelativeTo(null);
            app.setVisible(true);
        });
    }
}
