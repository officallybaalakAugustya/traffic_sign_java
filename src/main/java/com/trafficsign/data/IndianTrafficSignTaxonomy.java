package com.trafficsign.data;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.InputStream;
import java.util.*;

/**
 * Concrete implementation of TrafficSignTaxonomy for Indian traffic signs
 * conforming to IRC:67 Code of Practice for Road Signs (MoRTH).
 * Provides robust label normalization, fallback mappings, and safety risk descriptions.
 */
public class IndianTrafficSignTaxonomy implements TrafficSignTaxonomy {

    public record IndianSignClassInfo(
            int id,
            String canonicalName,
            String datasetLabel,
            String category,
            String ircCode,
            Integer speedLimitKmH,
            String safetyImplication
    ) {
    }

    private static final int DEFAULT_TOTAL_CLASSES = 85;
    private final Map<Integer, IndianSignClassInfo> classesById = new LinkedHashMap<>();
    private final Map<String, Integer> labelToId = new HashMap<>();

    public IndianTrafficSignTaxonomy() {
        boolean loaded = loadFromClasspath();
        if (!loaded || classesById.isEmpty()) {
            loadBuiltInDefaults();
        }
    }

    private boolean loadFromClasspath() {
        try (InputStream is = getClass().getResourceAsStream("/taxonomies/indian_traffic_signs_irc67.json")) {
            if (is == null) {
                return false;
            }
            ObjectMapper mapper = new ObjectMapper();
            JsonNode root = mapper.readTree(is);
            JsonNode classesNode = root.get("classes");
            if (classesNode != null && classesNode.isArray()) {
                for (JsonNode node : classesNode) {
                    int id = node.get("id").asInt();
                    String canonical = node.get("canonicalName").asText();
                    String datasetLabel = node.get("datasetLabel").asText();
                    String category = node.get("category").asText();
                    String ircCode = node.has("ircCode") ? node.get("ircCode").asText() : "IRC:67";
                    Integer speedLimit = node.hasNonNull("speedLimitKmH") ? node.get("speedLimitKmH").asInt() : null;
                    String safety = node.has("safetyImplication") ? node.get("safetyImplication").asText() : "";

                    IndianSignClassInfo info = new IndianSignClassInfo(id, canonical, datasetLabel, category, ircCode, speedLimit, safety);
                    registerClass(info);
                }
                return true;
            }
        } catch (Exception ignored) {
            // Fall back to built-in definitions
        }
        return false;
    }

    private void loadBuiltInDefaults() {
        // Core 66 standard IRC:67 classes
        addClass(0, "All Motor Vehicles Prohibited", "ALL_MOTOR_VEHICLE_PROHIBITED", "MANDATORY", null, "Entry strictly prohibited.");
        addClass(1, "Axle Load Limit", "AXLE_LOAD_LIMIT", "MANDATORY", null, "Weight limit for axles.");
        addClass(2, "Barrier Ahead", "BARRIER_AHEAD", "CAUTIONARY", null, "Toll or police barrier ahead.");
        addClass(3, "Bullock Carts Prohibited", "BULLOCK_PROHIBITED", "MANDATORY", null, "Slow animal transport banned.");
        addClass(4, "Cattle Crossing", "CATTLE", "CAUTIONARY", null, "Livestock hazard on road.");
        addClass(5, "Compulsory Ahead", "COMPULSARY_AHEAD", "MANDATORY", null, "Straight traffic only.");
        addClass(6, "Compulsory Ahead or Turn Right", "COMPULSARY_AHEAD_OR_TURN_RIGHT", "MANDATORY", null, "Straight or right turn only.");
        addClass(7, "Compulsory Cycle Track", "COMPULSARY_CYCLE_TRACK", "MANDATORY", null, "Cycles only.");
        addClass(8, "Compulsory Keep Left", "COMPULSARY_KEEP_LEFT", "MANDATORY", null, "Keep left of median.");
        addClass(9, "Compulsory Keep Right", "COMPULSARY_KEEP_RIGHT", "MANDATORY", null, "Keep right of obstruction.");
        addClass(10, "Compulsory Minimum Speed", "COMPULSARY_MINIMUM_SPEED", "MANDATORY", null, "Maintain minimum speed.");
        addClass(11, "Compulsory Sound Horn", "COMPULSARY_SOUND_HORN", "MANDATORY", null, "Sound horn on blind curve.");
        addClass(12, "Compulsory Turn Left", "COMPULSARY_TURN_LEFT", "MANDATORY", null, "Mandatory left turn.");
        addClass(13, "Compulsory Turn Right Ahead", "COMPULSARY_TURN_RIGHT_AHEAD", "MANDATORY", null, "Turn right ahead.");
        addClass(14, "Cross Road", "CROSS_ROAD", "CAUTIONARY", null, "4-way intersection hazard.");
        addClass(15, "Cycles Prohibited", "CYCLE_PROHIBITED", "MANDATORY", null, "No bicycles.");
        addClass(16, "Dangerous Dip", "DANGEROUS_DIP", "CAUTIONARY", null, "Road depression / causeway.");
        addClass(17, "Direction", "DIRECTION", "INFORMATORY", null, "Navigational arrow.");
        addClass(18, "Falling Rocks", "FALLING_ROCKS", "CAUTIONARY", null, "Landslide warning.");
        addClass(19, "Gap in Median", "GAP_IN_MEDIAN", "CAUTIONARY", null, "Median opening for U-turns.");
        addClass(20, "Give Way", "GIVE_WAY", "MANDATORY", null, "Yield right-of-way.");
        addClass(21, "Guarded Level Crossing", "GUARDED_LEVEL_CROSSING", "CAUTIONARY", null, "Railway gate ahead.");
        addClass(22, "Handcarts Prohibited", "HANDCART_PROHIBITED", "MANDATORY", null, "No handcarts.");
        addClass(23, "Height Limit", "HEIGHT_LIMIT", "MANDATORY", null, "Overhead clearance limit.");
        addClass(24, "Horn Prohibited", "HORN_PROHIBITED", "MANDATORY", null, "Silence zone. Do not honk.");
        addClass(25, "Hump or Rough Road", "HUMP_OR_ROUGH_ROAD", "CAUTIONARY", null, "Speed breaker ahead.");
        addClass(26, "Left Reverse Bend", "LEFT_REVERSE_BEND", "CAUTIONARY", null, "Double bend left first.");
        addClass(27, "Left Turn Prohibited", "LEFT_TURN_PROHIBITED", "MANDATORY", null, "No left turn.");
        addClass(28, "Loose Gravel", "LOOSE_GRAVEL", "CAUTIONARY", null, "Loose stone hazard.");
        addClass(29, "Men at Work", "MEN_AT_WORK", "CAUTIONARY", null, "Construction workers in lane.");
        addClass(30, "Narrow Road Ahead", "NARROW_ROAD_AHEAD", "CAUTIONARY", null, "Carriageway narrows.");
        addClass(31, "No Entry", "NO_ENTRY", "MANDATORY", null, "Entry strictly forbidden.");
        addClass(32, "No Parking", "NO_PARKING", "MANDATORY", null, "No vehicle parking.");
        addClass(33, "No Stopping or Standing", "NO_STOPPING_OR_STANDING", "MANDATORY", null, "No stopping on carriageway.");
        addClass(34, "Overtaking Prohibited", "OVERTAKING_PROHIBITED", "MANDATORY", null, "No overtaking allowed.");
        addClass(35, "Pass Either Side", "PASS_EITHER_SIDE", "INFORMATORY", null, "Pass obstacle left or right.");
        addClass(36, "Pedestrian Crossing", "PEDESTRIAN_CROSSING", "CAUTIONARY", null, "Zebra crossing. Yield to pedestrians.");
        addClass(37, "Pedestrians Prohibited", "PEDESTRIAN_PROHIBITED", "MANDATORY", null, "Pedestrians forbidden.");
        addClass(38, "Priority for Oncoming Vehicles", "PRIORITY_FOR_ONCOMING_VEHICLES", "MANDATORY", null, "Yield to oncoming stream.");
        addClass(39, "Quay Side or River Bank", "QUAY_SIDE_OR_RIVER_BANK", "CAUTIONARY", null, "Unprotected water edge.");
        addClass(40, "Restriction Ends", "RESTRICTION_ENDS", "MANDATORY", null, "Speed/passing limits end.");
        addClass(41, "Right Hairpin Bend", "RIGHT_HAIR_PIN_BEND", "CAUTIONARY", null, "Sharp 180-degree right turn.");
        addClass(42, "Right Hand Curve", "RIGHT_HAND_CURVE", "CAUTIONARY", null, "Curvature to right.");
        addClass(43, "Roundabout", "ROUNDABOUT", "CAUTIONARY", null, "Traffic circle ahead.");
        addClass(44, "School Ahead", "SCHOOL_AHEAD", "CAUTIONARY", null, "School zone. Drive under 25 km/h.");
        addClass(45, "Side Road Left", "SIDE_ROAD_LEFT", "CAUTIONARY", null, "Side road joins left.");
        addClass(46, "Side Road Right", "SIDE_ROAD_RIGHT", "CAUTIONARY", null, "Side road joins right.");
        addClass(47, "Slippery Road", "SLIPPERY_ROAD", "CAUTIONARY", null, "Wet or low-traction road.");
        addClass(48, "Speed Limit 5 km/h", "SPEED_LIMIT_5", "MANDATORY", 5, "Crawl speed 5 km/h.");
        addClass(49, "Speed Limit 15 km/h", "SPEED_LIMIT_15", "MANDATORY", 15, "Low speed 15 km/h.");
        addClass(50, "Speed Limit 30 km/h", "SPEED_LIMIT_30", "MANDATORY", 30, "Speed limit 30 km/h.");
        addClass(51, "Speed Limit 40 km/h", "SPEED_LIMIT_40", "MANDATORY", 40, "Speed limit 40 km/h.");
        addClass(52, "Speed Limit 50 km/h", "SPEED_LIMIT_50", "MANDATORY", 50, "Speed limit 50 km/h.");
        addClass(53, "Speed Limit 60 km/h", "SPEED_LIMIT_60", "MANDATORY", 60, "Speed limit 60 km/h.");
        addClass(54, "Speed Limit 70 km/h", "SPEED_LIMIT_70", "MANDATORY", 70, "Speed limit 70 km/h.");
        addClass(55, "Speed Limit 80 km/h", "SPEED_LIMIT_80", "MANDATORY", 80, "Speed limit 80 km/h.");
        addClass(56, "Staggered Intersection", "STAGGERED_INTERSECTION", "CAUTIONARY", null, "Offset junction hazard.");
        addClass(57, "Steep Ascent", "STEEP_ASCENT", "CAUTIONARY", null, "Steep uphill climb.");
        addClass(58, "Steep Descent", "STEEP_DESCENT", "CAUTIONARY", null, "Steep downhill descent.");
        addClass(59, "Stop", "STOP", "MANDATORY", null, "Mandatory stop at line.");
        addClass(60, "Straight Prohibited", "STRAIGHT_PROHIBITED", "MANDATORY", null, "Through traffic banned.");
        addClass(61, "T-Intersection", "T_INTERSECTION", "CAUTIONARY", null, "Perpendicular dead end.");
        addClass(62, "Trucks Prohibited", "TRUCK_PROHIBITED", "MANDATORY", null, "Commercial trucks banned.");
        addClass(63, "Turn Right", "TURN_RIGHT", "MANDATORY", null, "Turn right mandatory.");
        addClass(64, "U-Turn Prohibited", "U_TURN_PROHIBITED", "MANDATORY", null, "No U-turns.");
        addClass(65, "Y-Intersection", "Y_INTERSECTION", "CAUTIONARY", null, "Fork in roadway.");
        addClass(66, "Bullock and Handcart Prohibited", "BULLOCK_AND_HANDCART_PROHIBITED", "MANDATORY", null, "Non-motorized freight transport prohibited.");
        addClass(67, "Compulsory Ahead or Turn Left", "COMPULSARY_AHEAD_OR_TURN_LEFT", "MANDATORY", null, "Maneuver restricted to straight or left turn.");
        addClass(68, "Compulsory Turn Left Ahead", "COMPULSARY_TURN_LEFT_AHEAD", "MANDATORY", null, "Turn left ahead mandatory.");
        addClass(69, "Compulsory Turn Right", "COMPULSARY_TURN_RIGHT", "MANDATORY", null, "Turn right mandatory.");
        addClass(70, "Cycle Crossing", "CYCLE_CROSSING", "CAUTIONARY", null, "Bicycle crossing ahead; yield to cyclists.");
        addClass(71, "Ferry", "FERRY", "CAUTIONARY", null, "Ferry boat service ahead.");
        addClass(72, "Left Hairpin Bend", "LEFT_HAIR_PIN_BEND", "CAUTIONARY", null, "Sharp 180-degree left hairpin turn on ghat roads.");
        addClass(73, "Left Hand Curve", "LEFT_HAND_CURVE", "CAUTIONARY", null, "Sharp left curve ahead; decelerate.");
        addClass(74, "Length Limit", "LENGTH_LIMIT", "MANDATORY", null, "Vehicle length restriction ahead.");
        addClass(75, "Load Limit", "LOAD_LIMIT", "MANDATORY", null, "Total vehicle gross weight limit.");
        addClass(76, "Narrow Bridge", "NARROW_BRIDGE", "CAUTIONARY", null, "Road narrows to bridge width; do not overtake.");
        addClass(77, "Right Reverse Bend", "RIGHT_REVERSE_BEND", "CAUTIONARY", null, "Double bend right first.");
        addClass(78, "Right Turn Prohibited", "RIGHT_TURN_PROHIBITED", "MANDATORY", null, "Right turn prohibited.");
        addClass(79, "Road Widens Ahead", "ROAD_WIDENS_AHEAD", "CAUTIONARY", null, "Carriageway broadens ahead.");
        addClass(80, "Speed Limit 20 km/h", "SPEED_LIMIT_20", "MANDATORY", 20, "Speed limit 20 km/h.");
        addClass(81, "Tonga Prohibited", "TONGA_PROHIBITED", "MANDATORY", null, "Horse-drawn passenger tongas prohibited.");
        addClass(82, "Traffic Signal", "TRAFFIC_SIGNAL", "CAUTIONARY", null, "Traffic light signals ahead; prepare to stop on amber/red.");
        addClass(83, "Unguarded Level Crossing", "UNGUARDED_LEVEL_CROSSING", "CAUTIONARY", null, "Railway crossing without gates; stop, look, and listen.");
        addClass(84, "Width Limit", "WIDTH_LIMIT", "MANDATORY", null, "Maximum vehicle width restriction.");
    }

    private void addClass(int id, String name, String label, String cat, Integer speed, String safety) {
        registerClass(new IndianSignClassInfo(id, name, label, cat, "IRC:67", speed, safety));
    }

    private void registerClass(IndianSignClassInfo info) {
        classesById.put(info.id(), info);
        String normalizedLabel = normalize(info.datasetLabel());
        labelToId.put(normalizedLabel, info.id());

        // Also index canonical name and variants without prefix digits
        labelToId.put(normalize(info.canonicalName()), info.id());
        String stripped = normalizedLabel.replaceFirst("^\\d+", "");
        if (!stripped.isEmpty()) {
            labelToId.put(stripped, info.id());
        }
    }

    private String normalize(String s) {
        return s.toUpperCase().replaceAll("[^A-Z0-9]", "_").replaceAll("_+", "_").replaceAll("^_|_$", "");
    }

    @Override
    public int totalClasses() {
        return Math.max(DEFAULT_TOTAL_CLASSES, classesById.size());
    }

    @Override
    public String getCanonicalName(int classId) {
        IndianSignClassInfo info = classesById.get(classId);
        return info != null ? info.canonicalName() : "Unknown Indian Sign (" + classId + ")";
    }

    @Override
    public String getCategory(int classId) {
        IndianSignClassInfo info = classesById.get(classId);
        return info != null ? info.category() : "OTHER";
    }

    @Override
    public Optional<Integer> getSpeedLimitKmH(int classId) {
        IndianSignClassInfo info = classesById.get(classId);
        return info != null ? Optional.ofNullable(info.speedLimitKmH()) : Optional.empty();
    }

    @Override
    public Optional<Integer> findClassIdByLabel(String label) {
        if (label == null || label.isBlank()) {
            return Optional.empty();
        }
        String norm = normalize(label);
        Integer id = labelToId.get(norm);
        if (id != null) {
            return Optional.of(id);
        }
        // Try stripping numeric prefix if formatted like "0ALL_MOTOR_VEHICLE_PROHIBITED"
        String stripped = norm.replaceFirst("^\\d+", "");
        return Optional.ofNullable(labelToId.get(stripped));
    }

    public Optional<IndianSignClassInfo> getClassInfo(int classId) {
        return Optional.ofNullable(classesById.get(classId));
    }

    public Optional<String> getSafetyAlert(int classId) {
        IndianSignClassInfo info = classesById.get(classId);
        return info != null ? Optional.of(info.safetyImplication()) : Optional.empty();
    }
}
