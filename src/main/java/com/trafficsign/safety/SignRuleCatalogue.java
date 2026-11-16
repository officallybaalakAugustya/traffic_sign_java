package com.trafficsign.safety;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Optional;

public class SignRuleCatalogue {

    public enum RuleCategory {
        REGULATORY,
        CAUTIONARY,
        INFORMATORY,
        UNKNOWN
    }

    public enum SeverityLevel {
        INFO,
        WARNING,
        CRITICAL
    }

    public enum EventType {
        SPEED_LIMIT_POSTED,
        STOP_MANDATE,
        GIVE_WAY_MANDATE,
        PROHIBITION_ALERT,
        HAZARD_WARNING,
        RESTRICTION_ALERT,
        RESTRICTION_ENDS,
        TRAFFIC_SIGNAL_AHEAD,
        LEVEL_CROSSING_ALERT,
        GENERAL_INFO,
        UNCERTAIN_DETECTION
    }

    public record SignRule(
            int classId,
            String canonicalLabel,
            String displayName,
            RuleCategory category,
            SeverityLevel severity,
            EventType eventType,
            String driverAdvisory,
            Double numericalValue,
            String unit,
            String verificationStatus
    ) {
        public boolean isSpeedLimit() {
            return eventType == EventType.SPEED_LIMIT_POSTED && numericalValue != null;
        }

        public boolean isCritical() {
            return severity == SeverityLevel.CRITICAL;
        }

        public boolean isRegulatory() {
            return category == RuleCategory.REGULATORY;
        }
    }

    private static final String VERIFIED_STATUS = "IRC:67-2012 Verified";
    private static final String FALLBACK_STATUS = "Unverified Fallback";

    private final Map<Integer, SignRule> rulesByClassId = new HashMap<>();
    private final Map<String, SignRule> rulesByLabel = new HashMap<>();
    private final SignRule defaultUnknownRule;

    public SignRuleCatalogue() {
        this.defaultUnknownRule = new SignRule(
                -1,
                "UNKNOWN_SIGN",
                "Unrecognized Traffic Sign",
                RuleCategory.UNKNOWN,
                SeverityLevel.INFO,
                EventType.UNCERTAIN_DETECTION,
                "Sign detected with uncertain classification; exercise general caution and adhere to prevailing road conditions.",
                null,
                null,
                FALLBACK_STATUS
        );
        populateCatalogue();
    }

    public SignRule getRule(int classId) {
        return rulesByClassId.getOrDefault(classId, defaultUnknownRule);
    }

    public SignRule getRuleByLabel(String label) {
        if (label == null || label.isBlank()) {
            return defaultUnknownRule;
        }
        String normalized = label.trim().toUpperCase().replace("-", "_").replace(" ", "_");
        return rulesByLabel.getOrDefault(normalized, defaultUnknownRule);
    }

    public Optional<SignRule> findRule(int classId) {
        return Optional.ofNullable(rulesByClassId.get(classId));
    }

    public Map<Integer, SignRule> getAllRules() {
        return Collections.unmodifiableMap(rulesByClassId);
    }

    public SignRule getDefaultUnknownRule() {
        return defaultUnknownRule;
    }

    public int totalRegisteredRules() {
        return rulesByClassId.size();
    }

    private void register(SignRule rule) {
        rulesByClassId.put(rule.classId(), rule);
        rulesByLabel.put(rule.canonicalLabel().toUpperCase(), rule);
        String altLabel = rule.displayName().toUpperCase().replace(" ", "_").replace("-", "_");
        rulesByLabel.put(altLabel, rule);
    }

    private void populateCatalogue() {
        register(new SignRule(0, "ALL_MOTOR_VEHICLE_PROHIBITED", "All Motor Vehicles Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                "Do not enter with any motorized vehicle. Turn around or select an alternate authorized route.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(1, "AXLE_LOAD_LIMIT", "Axle Load Limit",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                "Ensure vehicle axle load does not exceed posted statutory limit before proceeding.",
                null, "tonnes", VERIFIED_STATUS));

        register(new SignRule(2, "BARRIER_AHEAD", "Barrier Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Physical barrier, toll booth, or checkpoint ahead. Decelerate smoothly and prepare to stop.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(3, "BULLOCK_PROHIBITED", "Bullock Carts Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                "Animal-drawn slow vehicles prohibited on this roadway section.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(4, "CATTLE", "Cattle Crossing",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Livestock hazard area. Reduce speed, scan road shoulders, and prepare for sudden crossing.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(5, "COMPULSARY_AHEAD", "Compulsory Ahead",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Mandatory straight movement. Do not make left or right turns at this intersection.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(6, "COMPULSARY_AHEAD_OR_TURN_RIGHT", "Compulsory Ahead or Turn Right",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Movement restricted to straight through or right turn only. Left turns prohibited.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(7, "COMPULSARY_CYCLE_TRACK", "Compulsory Cycle Track",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Designated lane for non-motorized cycles only. Keep motor vehicles clear.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(8, "COMPULSARY_KEEP_LEFT", "Compulsory Keep Left",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Pass road divider or island on the left side only.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(9, "COMPULSARY_KEEP_RIGHT", "Compulsory Keep Right",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Pass obstruction or traffic divider on the right side only.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(10, "COMPULSARY_MINIMUM_SPEED", "Compulsory Minimum Speed",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                "Maintain speed at or above posted minimum threshold to prevent rear-end collision hazard.",
                null, "km/h", VERIFIED_STATUS));

        register(new SignRule(11, "COMPULSARY_SOUND_HORN", "Compulsory Sound Horn",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                "Sound audible horn before entering blind turn or ghat hairpin curve.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(12, "COMPULSARY_TURN_LEFT", "Compulsory Turn Left",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Mandatory left turn. Straight progression and right turns prohibited.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(13, "COMPULSARY_TURN_RIGHT_AHEAD", "Compulsory Turn Right Ahead",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Prepare to turn right at the approaching junction as designated.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(14, "CROSS_ROAD", "Cross Road Intersection",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Four-way crossroads ahead. Decelerate, scan crossing traffic, and yield as required.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(15, "CYCLE_PROHIBITED", "Cycles Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                "Bicycle traffic prohibited beyond this point.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(16, "DANGEROUS_DIP", "Dangerous Dip Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.CRITICAL, EventType.HAZARD_WARNING,
                "Abrupt carriageway depression or causeway ahead. Brake immediately to prevent vehicle bottoming.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(17, "DIRECTION", "Direction Guide",
                RuleCategory.INFORMATORY, SeverityLevel.INFO, EventType.GENERAL_INFO,
                "Navigational directional guidance for upcoming junction.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(18, "FALLING_ROCKS", "Falling Rocks Hazard",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Hill road section prone to rockfalls and debris. Stay alert and avoid unnecessary stopping.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(19, "GAP_IN_MEDIAN", "Gap in Median",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Median crossover ahead. Watch for U-turning and merging cross traffic.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(20, "GIVE_WAY", "Give Way / Yield",
                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.GIVE_WAY_MANDATE,
                "Yield right of way to priority traffic on intersecting carriageway.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(21, "GUARDED_LEVEL_CROSSING", "Guarded Level Crossing",
                RuleCategory.CAUTIONARY, SeverityLevel.CRITICAL, EventType.LEVEL_CROSSING_ALERT,
                "Railway crossing with barrier ahead. Slow down and stop if signals or gates activate.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(22, "HANDCART_PROHIBITED", "Handcarts Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                "Manual handcarts and pushcarts prohibited beyond this point.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(23, "HEIGHT_LIMIT", "Height Limit",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                "Overhead clearance restriction ahead. Check vehicle total height including cargo.",
                null, "metres", VERIFIED_STATUS));

        register(new SignRule(24, "HORN_PROHIBITED", "Horn Prohibited (Silence Zone)",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Silence zone (hospital/school/court). Do not sound horn except to avert imminent accident.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(25, "HUMP_OR_ROUGH_ROAD", "Speed Breaker / Rough Road",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Speed hump or uneven pavement ahead. Decelerate to avoid severe chassis impact.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(26, "LEFT_REVERSE_BEND", "Left Reverse Bend",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Double curve commencing with left bend. Decelerate and maintain lane discipline.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(27, "LEFT_TURN_PROHIBITED", "Left Turn Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                "Left turn strictly prohibited at this intersection. Continue straight or turn right.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(28, "LOOSE_GRAVEL", "Loose Gravel",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Loose stones and reduced tire traction. Increase following distance to prevent windshield damage.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(29, "MEN_AT_WORK", "Men at Work Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Road maintenance crew active on carriageway. Slow down and watch for construction personnel.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(30, "NARROW_ROAD_AHEAD", "Narrow Road Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Carriageway width reduces ahead. Yield to oncoming vehicles and merge cautiously.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(31, "NO_ENTRY", "No Entry",
                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                "Entry prohibited for all vehicles. One-way traffic or closed passage ahead.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(32, "NO_PARKING", "No Parking",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Vehicle parking prohibited along this roadside corridor.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(33, "NO_STOPPING_OR_STANDING", "No Stopping or Standing",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                "Stopping or waiting is strictly banned. Keep moving to prevent severe flow obstruction.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(34, "OVERTAKING_PROHIBITED", "Overtaking Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                "Do not overtake any moving vehicle. High risk of head-on collision on this sector.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(35, "PASS_EITHER_SIDE", "Pass Either Side",
                RuleCategory.INFORMATORY, SeverityLevel.INFO, EventType.GENERAL_INFO,
                "Traffic may pass traffic island or obstacle on either left or right.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(36, "PEDESTRIAN_CROSSING", "Pedestrian Crossing Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Zebra crossing ahead. Slow down and be prepared to yield completely to pedestrians.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(37, "PEDESTRIAN_PROHIBITED", "Pedestrians Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                "Foot traffic prohibited on this expressway / high-speed corridor.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(38, "PRIORITY_FOR_ONCOMING_VEHICLES", "Priority for Oncoming Vehicles",
                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.GIVE_WAY_MANDATE,
                "Oncoming traffic has absolute priority across single-lane bottleneck. Yield before entry.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(39, "QUAY_SIDE_OR_RIVER_BANK", "Quayside or River Bank Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Roadway runs adjacent to unbarricaded water body. Steer with caution.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(40, "RESTRICTION_ENDS", "All Restrictions End",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ENDS,
                "Previously imposed speed and passing restrictions end. Default rules resume.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(41, "RIGHT_HAIR_PIN_BEND", "Right Hairpin Bend Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Acute 180-degree right turn ahead. Downshift, reduce speed, and sound horn on blind curve.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(42, "RIGHT_HAND_CURVE", "Right Hand Curve Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Road curves sharply right. Decelerate smoothly and maintain position within lane.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(43, "ROUNDABOUT", "Roundabout Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Traffic circle ahead. Slow down and yield to circulating traffic on the right.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(44, "SCHOOL_AHEAD", "School Zone Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "School zone. Reduce speed to at most 25 km/h and watch for children crossing.",
                25.0, "km/h", VERIFIED_STATUS));

        register(new SignRule(45, "SIDE_ROAD_LEFT", "Side Road Left Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Junction with side road entering from left. Watch for joining traffic.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(46, "SIDE_ROAD_RIGHT", "Side Road Right Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Junction with side road entering from right. Scan right approach.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(47, "SLIPPERY_ROAD", "Slippery Road Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Road surface has low skid resistance. Avoid abrupt braking or rapid steering inputs.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(48, "SPEED_LIMIT_5", "Speed Limit 5 km/h",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                "Maximum permissible speed is 5 km/h. Crawl speed required.",
                5.0, "km/h", VERIFIED_STATUS));

        register(new SignRule(49, "SPEED_LIMIT_15", "Speed Limit 15 km/h",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                "Maximum permissible speed is 15 km/h. Maintain strict low speed.",
                15.0, "km/h", VERIFIED_STATUS));

        register(new SignRule(50, "SPEED_LIMIT_30", "Speed Limit 30 km/h",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                "Maximum permissible speed is 30 km/h. Urban / residential zone compliance.",
                30.0, "km/h", VERIFIED_STATUS));

        register(new SignRule(51, "SPEED_LIMIT_40", "Speed Limit 40 km/h",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                "Maximum permissible speed is 40 km/h. Decelerate to posted maximum.",
                40.0, "km/h", VERIFIED_STATUS));

        register(new SignRule(52, "SPEED_LIMIT_50", "Speed Limit 50 km/h",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                "Maximum permissible speed is 50 km/h. City arterial speed compliance.",
                50.0, "km/h", VERIFIED_STATUS));

        register(new SignRule(53, "SPEED_LIMIT_60", "Speed Limit 60 km/h",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                "Maximum permissible speed is 60 km/h. Sub-arterial road speed limit.",
                60.0, "km/h", VERIFIED_STATUS));

        register(new SignRule(54, "SPEED_LIMIT_70", "Speed Limit 70 km/h",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                "Maximum permissible speed is 70 km/h. Highway sector speed ceiling.",
                70.0, "km/h", VERIFIED_STATUS));

        register(new SignRule(55, "SPEED_LIMIT_80", "Speed Limit 80 km/h",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                "Maximum permissible speed is 80 km/h. Expressway standard limit.",
                80.0, "km/h", VERIFIED_STATUS));

        register(new SignRule(56, "STAGGERED_INTERSECTION", "Staggered Intersection Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Offset cross-junctions ahead. Watch for turning maneuvers from both sides.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(57, "STEEP_ASCENT", "Steep Ascent Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.INFO, EventType.HAZARD_WARNING,
                "Steep incline ahead. Select appropriate lower gear and maintain headway.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(58, "STEEP_DESCENT", "Steep Descent Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Steep downhill gradient. Utilize engine braking; avoid excessive continuous foot-braking.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(59, "STOP", "Mandatory Stop",
                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.STOP_MANDATE,
                "Come to a complete stop before stop-line. Verify clearance before entering intersection.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(60, "STRAIGHT_PROHIBITED", "Straight Progression Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                "Through traffic prohibited ahead. Divert left or right as directed.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(61, "T_INTERSECTION", "T-Intersection Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Road terminates perpendicularly at T-junction ahead. Prepare to turn left or right.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(62, "TRUCK_PROHIBITED", "Heavy Trucks Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                "Heavy goods commercial vehicles prohibited on this roadway corridor.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(63, "TURN_RIGHT", "Mandatory Turn Right",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Mandatory right turn. All vehicles must follow right direction.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(64, "U_TURN_PROHIBITED", "U-Turn Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                "U-turns strictly forbidden at this location. High collision hazard with oncoming traffic.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(65, "Y_INTERSECTION", "Y-Intersection Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Roadway forks into two branches. Choose proper lane early.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(66, "BULLOCK_AND_HANDCART_PROHIBITED", "Bullock Carts and Handcarts Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                "All non-motorized animal-drawn and manual transport prohibited on this sector.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(67, "COMPULSARY_AHEAD_OR_TURN_LEFT", "Compulsory Ahead or Turn Left",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Movement restricted to straight through or left turn only. Right turns prohibited.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(68, "COMPULSARY_TURN_LEFT_AHEAD", "Compulsory Turn Left Ahead",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Mandatory left turn required at upcoming intersection.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(69, "COMPULSARY_TURN_RIGHT", "Compulsory Turn Right",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                "Mandatory right turn. Straight movement prohibited.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(70, "CYCLE_CROSSING", "Cycle Crossing Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Bicycle crossing ahead. Decelerate and give right of way to cyclists.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(71, "FERRY", "Ferry Boat Service Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.INFO, EventType.GENERAL_INFO,
                "Road terminates at river ferry terminal. Slow down and follow marshalling.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(72, "LEFT_HAIR_PIN_BEND", "Left Hairpin Bend Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Sharp 180-degree left hairpin curve on hill sector. Reduce speed and sound horn.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(73, "LEFT_HAND_CURVE", "Left Hand Curve Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Curvature to left ahead. Decelerate and maintain lane discipline.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(74, "LENGTH_LIMIT", "Vehicle Length Limit",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                "Maximum vehicle overall length restriction ahead. Long vehicles must divert.",
                null, "metres", VERIFIED_STATUS));

        register(new SignRule(75, "LOAD_LIMIT", "Vehicle Load Limit",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                "Gross vehicle weight restriction ahead. Overweight vehicles prohibited.",
                null, "tonnes", VERIFIED_STATUS));

        register(new SignRule(76, "NARROW_BRIDGE", "Narrow Bridge Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Carriageway narrows across bridge deck. No overtaking; yield to vehicle already on bridge.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(77, "RIGHT_REVERSE_BEND", "Right Reverse Bend",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                "Double curve commencing with right bend. Decelerate and keep left.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(78, "RIGHT_TURN_PROHIBITED", "Right Turn Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                "Right turn strictly prohibited at this intersection.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(79, "ROAD_WIDENS_AHEAD", "Road Widens Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.INFO, EventType.GENERAL_INFO,
                "Additional lane or carriageway widening ahead.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(80, "SPEED_LIMIT_20", "Speed Limit 20 km/h",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                "Maximum permissible speed is 20 km/h. Reduce speed accordingly.",
                20.0, "km/h", VERIFIED_STATUS));

        register(new SignRule(81, "TONGA_PROHIBITED", "Tonga (Horse-Drawn) Prohibited",
                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                "Horse-drawn passenger tongas prohibited on this arterial sector.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(82, "TRAFFIC_SIGNAL", "Traffic Signals Ahead",
                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.TRAFFIC_SIGNAL_AHEAD,
                "Traffic light intersection ahead. Prepare to decelerate and stop on amber or red.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(83, "UNGUARDED_LEVEL_CROSSING", "Unguarded Railway Crossing",
                RuleCategory.CAUTIONARY, SeverityLevel.CRITICAL, EventType.LEVEL_CROSSING_ALERT,
                "Railway crossing without barrier gates ahead. Mandatory stop, look both directions, and listen.",
                null, null, VERIFIED_STATUS));

        register(new SignRule(84, "WIDTH_LIMIT", "Vehicle Width Limit",
                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                "Maximum vehicle width restriction ahead. Wide-load vehicles prohibited.",
                null, "metres", VERIFIED_STATUS));
    }
}
