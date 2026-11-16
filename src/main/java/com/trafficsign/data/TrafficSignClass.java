package com.trafficsign.data;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.Optional;
import java.util.function.Function;
import java.util.stream.Collectors;

/**
 * Standard 43 traffic-sign classes defined by the German Traffic Sign Recognition
 * Benchmark (GTSRB) and Vienna Convention standards.
 */
public enum TrafficSignClass {
    SPEED_LIMIT_20(0, "Speed limit (20km/h)", SignCategory.PROHIBITORY, 20),
    SPEED_LIMIT_30(1, "Speed limit (30km/h)", SignCategory.PROHIBITORY, 30),
    SPEED_LIMIT_50(2, "Speed limit (50km/h)", SignCategory.PROHIBITORY, 50),
    SPEED_LIMIT_60(3, "Speed limit (60km/h)", SignCategory.PROHIBITORY, 60),
    SPEED_LIMIT_70(4, "Speed limit (70km/h)", SignCategory.PROHIBITORY, 70),
    SPEED_LIMIT_80(5, "Speed limit (80km/h)", SignCategory.PROHIBITORY, 80),
    END_OF_SPEED_LIMIT_80(6, "End of speed limit (80km/h)", SignCategory.OTHER, 80),
    SPEED_LIMIT_100(7, "Speed limit (100km/h)", SignCategory.PROHIBITORY, 100),
    SPEED_LIMIT_120(8, "Speed limit (120km/h)", SignCategory.PROHIBITORY, 120),
    NO_PASSING(9, "No passing", SignCategory.PROHIBITORY, null),
    NO_PASSING_VEHICLES_OVER_3_5T(10, "No passing vehicles over 3.5 metric tons", SignCategory.PROHIBITORY, null),
    RIGHT_OF_WAY_AT_INTERSECTION(11, "Right-of-way at intersection", SignCategory.DANGER, null),
    PRIORITY_ROAD(12, "Priority road", SignCategory.OTHER, null),
    YIELD(13, "Yield", SignCategory.DANGER, null),
    STOP(14, "Stop", SignCategory.PROHIBITORY, null),
    NO_VEHICLES(15, "No vehicles", SignCategory.PROHIBITORY, null),
    VEHICLES_OVER_3_5T_PROHIBITED(16, "Vehicles over 3.5 metric tons prohibited", SignCategory.PROHIBITORY, null),
    NO_ENTRY(17, "No entry", SignCategory.PROHIBITORY, null),
    GENERAL_CAUTION(18, "General caution", SignCategory.DANGER, null),
    DANGEROUS_CURVE_TO_THE_LEFT(19, "Dangerous curve to the left", SignCategory.DANGER, null),
    DANGEROUS_CURVE_TO_THE_RIGHT(20, "Dangerous curve to the right", SignCategory.DANGER, null),
    DOUBLE_CURVE(21, "Double curve", SignCategory.DANGER, null),
    BUMPY_ROAD(22, "Bumpy road", SignCategory.DANGER, null),
    SLIPPERY_ROAD(23, "Slippery road", SignCategory.DANGER, null),
    ROAD_NARROWS_ON_THE_RIGHT(24, "Road narrows on the right", SignCategory.DANGER, null),
    ROAD_WORK(25, "Road work", SignCategory.DANGER, null),
    TRAFFIC_SIGNALS(26, "Traffic signals", SignCategory.DANGER, null),
    PEDESTRIANS(27, "Pedestrians", SignCategory.DANGER, null),
    CHILDREN_CROSSING(28, "Children crossing", SignCategory.DANGER, null),
    BICYCLES_CROSSING(29, "Bicycles crossing", SignCategory.DANGER, null),
    BEWARE_OF_ICE_SNOW(30, "Beware of ice/snow", SignCategory.DANGER, null),
    WILD_ANIMALS_CROSSING(31, "Wild animals crossing", SignCategory.DANGER, null),
    END_OF_ALL_SPEED_AND_PASSING_LIMITS(32, "End of all speed and passing limits", SignCategory.OTHER, null),
    TURN_RIGHT_AHEAD(33, "Turn right ahead", SignCategory.MANDATORY, null),
    TURN_LEFT_AHEAD(34, "Turn left ahead", SignCategory.MANDATORY, null),
    AHEAD_ONLY(35, "Ahead only", SignCategory.MANDATORY, null),
    GO_STRAIGHT_OR_RIGHT(36, "Go straight or right", SignCategory.MANDATORY, null),
    GO_STRAIGHT_OR_LEFT(37, "Go straight or left", SignCategory.MANDATORY, null),
    KEEP_RIGHT(38, "Keep right", SignCategory.MANDATORY, null),
    KEEP_LEFT(39, "Keep left", SignCategory.MANDATORY, null),
    ROUNDABOUT_MANDATORY(40, "Roundabout mandatory", SignCategory.MANDATORY, null),
    END_OF_NO_PASSING(41, "End of no passing", SignCategory.OTHER, null),
    END_OF_NO_PASSING_VEHICLES_OVER_3_5T(42, "End of no passing vehicles over 3.5 metric tons", SignCategory.OTHER, null);

    public enum SignCategory {
        PROHIBITORY,
        DANGER,
        MANDATORY,
        OTHER
    }

    private final int id;
    private final String labelName;
    private final SignCategory category;
    private final Integer speedLimitKmH;

    private static final Map<Integer, TrafficSignClass> BY_ID = Collections.unmodifiableMap(
            Arrays.stream(values()).collect(Collectors.toMap(TrafficSignClass::getId, Function.identity()))
    );

    TrafficSignClass(int id, String labelName, SignCategory category, Integer speedLimitKmH) {
        this.id = id;
        this.labelName = labelName;
        this.category = category;
        this.speedLimitKmH = speedLimitKmH;
    }

    public int getId() {
        return id;
    }

    public String getLabelName() {
        return labelName;
    }

    public SignCategory getCategory() {
        return category;
    }

    public Optional<Integer> getSpeedLimitKmH() {
        return Optional.ofNullable(speedLimitKmH);
    }

    public static Optional<TrafficSignClass> fromId(int id) {
        return Optional.ofNullable(BY_ID.get(id));
    }

    public static TrafficSignClass requireById(int id) {
        TrafficSignClass signClass = BY_ID.get(id);
        if (signClass == null) {
            throw new IllegalArgumentException("Unknown traffic sign class id: " + id + ". Valid range is 0 to 42.");
        }
        return signClass;
    }

    public static int totalClasses() {
        return values().length;
    }
}
