"""
Sign Rule Catalogue mapping IRC:67 traffic sign classes to statutory driver advisories,
severity levels, and road safety event types.
Mirroring com.trafficsign.safety.SignRuleCatalogue.
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional


class RuleCategory(str, Enum):
    REGULATORY = "REGULATORY"
    CAUTIONARY = "CAUTIONARY"
    INFORMATORY = "INFORMATORY"
    UNKNOWN = "UNKNOWN"


class SeverityLevel(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


class EventType(str, Enum):
    SPEED_LIMIT_POSTED = "SPEED_LIMIT_POSTED"
    STOP_MANDATE = "STOP_MANDATE"
    GIVE_WAY_MANDATE = "GIVE_WAY_MANDATE"
    PROHIBITION_ALERT = "PROHIBITION_ALERT"
    HAZARD_WARNING = "HAZARD_WARNING"
    RESTRICTION_ALERT = "RESTRICTION_ALERT"
    RESTRICTION_ENDS = "RESTRICTION_ENDS"
    TRAFFIC_SIGNAL_AHEAD = "TRAFFIC_SIGNAL_AHEAD"
    LEVEL_CROSSING_ALERT = "LEVEL_CROSSING_ALERT"
    GENERAL_INFO = "GENERAL_INFO"
    UNCERTAIN_DETECTION = "UNCERTAIN_DETECTION"


@dataclass(frozen=True)
class SignRule:
    class_id: int
    canonical_label: str
    display_name: str
    category: RuleCategory
    severity: SeverityLevel
    event_type: EventType
    driver_advisory: str
    numerical_value: Optional[float] = None
    unit: Optional[str] = None
    verification_status: str = "IRC:67-2012 Verified"

    def is_speed_limit(self) -> bool:
        return self.event_type == EventType.SPEED_LIMIT_POSTED and self.numerical_value is not None

    def is_critical(self) -> bool:
        return self.severity == SeverityLevel.CRITICAL

    def is_regulatory(self) -> bool:
        return self.category == RuleCategory.REGULATORY


class SignRuleCatalogue:
    """
    Statutory rule catalogue mapping traffic sign classes to driver advisories and event types.
    """

    VERIFIED_STATUS = "IRC:67-2012 Verified"
    FALLBACK_STATUS = "Unverified Fallback"

    def __init__(self):
        self._rules_by_class_id: Dict[int, SignRule] = {}
        self._rules_by_label: Dict[str, SignRule] = {}

        self.default_unknown_rule = SignRule(
            class_id=-1,
            canonical_label="UNKNOWN_SIGN",
            display_name="Unrecognized Traffic Sign",
            category=RuleCategory.UNKNOWN,
            severity=SeverityLevel.INFO,
            event_type=EventType.UNCERTAIN_DETECTION,
            driver_advisory="Sign detected with uncertain classification; exercise general caution and adhere to prevailing road conditions.",
            numerical_value=None,
            unit=None,
            verification_status=self.FALLBACK_STATUS
        )

        self._populate_catalogue()

    def _register(self, rule: SignRule) -> None:
        self._rules_by_class_id[rule.class_id] = rule
        self._rules_by_label[rule.canonical_label.upper()] = rule
        alt = rule.display_name.upper().replace(" ", "_").replace("-", "_")
        self._rules_by_label[alt] = rule

    def get_rule(self, class_id: int) -> SignRule:
        return self._rules_by_class_id.get(class_id, self.default_unknown_rule)

    def get_rule_by_label(self, label: str) -> SignRule:
        if not label:
            return self.default_unknown_rule
        norm = label.strip().upper().replace("-", "_").replace(" ", "_")
        return self._rules_by_label.get(norm, self.default_unknown_rule)

    def total_registered_rules(self) -> int:
        return len(self._rules_by_class_id)

    def _populate_catalogue(self) -> None:
        self._register(SignRule(0, "ALL_MOTOR_VEHICLE_PROHIBITED", "All Motor Vehicles Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                                "Do not enter with any motorized vehicle. Turn around or select an alternate authorized route."))

        self._register(SignRule(1, "AXLE_LOAD_LIMIT", "Axle Load Limit",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                                "Ensure vehicle axle load does not exceed posted statutory limit before proceeding.",
                                unit="tonnes"))

        self._register(SignRule(2, "BARRIER_AHEAD", "Barrier Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Physical barrier, toll booth, or checkpoint ahead. Decelerate smoothly and prepare to stop."))

        self._register(SignRule(3, "BULLOCK_PROHIBITED", "Bullock Carts Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                                "Animal-drawn slow vehicles prohibited on this roadway section."))

        self._register(SignRule(4, "CATTLE", "Cattle Crossing",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Livestock hazard area. Reduce speed, scan road shoulders, and prepare for sudden crossing."))

        self._register(SignRule(5, "COMPULSARY_AHEAD", "Compulsory Ahead",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Mandatory straight movement. Do not make left or right turns at this intersection."))

        self._register(SignRule(6, "COMPULSARY_AHEAD_OR_TURN_RIGHT", "Compulsory Ahead or Turn Right",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Movement restricted to straight through or right turn only. Left turns prohibited."))

        self._register(SignRule(7, "COMPULSARY_CYCLE_TRACK", "Compulsory Cycle Track",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Designated lane for non-motorized cycles only. Keep motor vehicles clear."))

        self._register(SignRule(8, "COMPULSARY_KEEP_LEFT", "Compulsory Keep Left",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Pass road divider or island on the left side only."))

        self._register(SignRule(9, "COMPULSARY_KEEP_RIGHT", "Compulsory Keep Right",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Pass obstruction or traffic divider on the right side only."))

        self._register(SignRule(10, "COMPULSARY_MINIMUM_SPEED", "Compulsory Minimum Speed",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                                "Maintain speed at or above posted minimum threshold to prevent rear-end collision hazard.",
                                unit="km/h"))

        self._register(SignRule(11, "COMPULSARY_SOUND_HORN", "Compulsory Sound Horn",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                                "Sound audible horn before entering blind turn or ghat hairpin curve."))

        self._register(SignRule(12, "COMPULSARY_TURN_LEFT", "Compulsory Turn Left",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Mandatory left turn. Straight progression and right turns prohibited."))

        self._register(SignRule(13, "COMPULSARY_TURN_RIGHT_AHEAD", "Compulsory Turn Right Ahead",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Prepare to turn right at the approaching junction as designated."))

        self._register(SignRule(14, "CROSS_ROAD", "Cross Road Intersection",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Four-way crossroads ahead. Decelerate, scan crossing traffic, and yield as required."))

        self._register(SignRule(15, "CYCLE_PROHIBITED", "Cycles Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                                "Bicycle traffic prohibited beyond this point."))

        self._register(SignRule(16, "DANGEROUS_DIP", "Dangerous Dip Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.CRITICAL, EventType.HAZARD_WARNING,
                                "Abrupt carriageway depression or causeway ahead. Brake immediately to prevent vehicle bottoming."))

        self._register(SignRule(17, "DIRECTION", "Direction Guide",
                                RuleCategory.INFORMATORY, SeverityLevel.INFO, EventType.GENERAL_INFO,
                                "Navigational directional guidance for upcoming junction."))

        self._register(SignRule(18, "FALLING_ROCKS", "Falling Rocks Hazard",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Hill road section prone to rockfalls and debris. Stay alert and avoid unnecessary stopping."))

        self._register(SignRule(19, "GAP_IN_MEDIAN", "Gap in Median",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Median crossover ahead. Watch for U-turning and merging cross traffic."))

        self._register(SignRule(20, "GIVE_WAY", "Give Way / Yield",
                                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.GIVE_WAY_MANDATE,
                                "Yield right of way to priority traffic on intersecting carriageway."))

        self._register(SignRule(21, "GUARDED_LEVEL_CROSSING", "Guarded Level Crossing",
                                RuleCategory.CAUTIONARY, SeverityLevel.CRITICAL, EventType.LEVEL_CROSSING_ALERT,
                                "Railway crossing with barrier ahead. Slow down and stop if signals or gates activate."))

        self._register(SignRule(22, "HANDCART_PROHIBITED", "Handcarts Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                                "Manual handcarts and pushcarts prohibited beyond this point."))

        self._register(SignRule(23, "HEIGHT_LIMIT", "Height Limit",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                                "Overhead clearance restriction ahead. Check vehicle total height including cargo.",
                                unit="metres"))

        self._register(SignRule(24, "HORN_PROHIBITED", "Horn Prohibited (Silence Zone)",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Silence zone (hospital/school/court). Do not sound horn except to avert imminent accident."))

        self._register(SignRule(25, "HUMP_OR_ROUGH_ROAD", "Speed Breaker / Rough Road",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Speed hump or uneven pavement ahead. Decelerate to avoid severe chassis impact."))

        self._register(SignRule(26, "LEFT_REVERSE_BEND", "Left Reverse Bend",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Double curve commencing with left bend. Decelerate and maintain lane discipline."))

        self._register(SignRule(27, "LEFT_TURN_PROHIBITED", "Left Turn Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                                "Left turn strictly prohibited at this intersection. Continue straight or turn right."))

        self._register(SignRule(28, "LOOSE_GRAVEL", "Loose Gravel",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Loose stones and reduced tire traction. Increase following distance to prevent windshield damage."))

        self._register(SignRule(29, "MEN_AT_WORK", "Men at Work Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Road maintenance crew active on carriageway. Slow down and watch for construction personnel."))

        self._register(SignRule(30, "NARROW_ROAD_AHEAD", "Narrow Road Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Carriageway width reduces ahead. Yield to oncoming vehicles and merge cautiously."))

        self._register(SignRule(31, "NO_ENTRY", "No Entry",
                                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                                "Entry prohibited for all vehicles. One-way traffic or closed passage ahead."))

        self._register(SignRule(32, "NO_PARKING", "No Parking",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Vehicle parking prohibited along this roadside corridor."))

        self._register(SignRule(33, "NO_STOPPING_OR_STANDING", "No Stopping or Standing",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                                "Stopping or waiting is strictly banned. Keep moving to prevent severe flow obstruction."))

        self._register(SignRule(34, "OVERTAKING_PROHIBITED", "Overtaking Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                                "Do not overtake any moving vehicle. High risk of head-on collision on this sector."))

        self._register(SignRule(35, "PASS_EITHER_SIDE", "Pass Either Side",
                                RuleCategory.INFORMATORY, SeverityLevel.INFO, EventType.GENERAL_INFO,
                                "Traffic may pass traffic island or obstacle on either left or right."))

        self._register(SignRule(36, "PEDESTRIAN_CROSSING", "Pedestrian Crossing Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Zebra crossing ahead. Slow down and be prepared to yield completely to pedestrians."))

        self._register(SignRule(37, "PEDESTRIAN_PROHIBITED", "Pedestrians Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                                "Foot traffic prohibited on this expressway / high-speed corridor."))

        self._register(SignRule(38, "PRIORITY_FOR_ONCOMING_VEHICLES", "Priority for Oncoming Vehicles",
                                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.GIVE_WAY_MANDATE,
                                "Oncoming traffic has absolute priority across single-lane bottleneck. Yield before entry."))

        self._register(SignRule(39, "QUAY_SIDE_OR_RIVER_BANK", "Quayside or River Bank Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Roadway runs adjacent to unbarricaded water body. Steer with caution."))

        self._register(SignRule(40, "RESTRICTION_ENDS", "All Restrictions End",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ENDS,
                                "Previously imposed speed and passing restrictions end. Default rules resume."))

        self._register(SignRule(41, "RIGHT_HAIR_PIN_BEND", "Right Hairpin Bend Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Acute 180-degree right turn ahead. Downshift, reduce speed, and sound horn on blind curve."))

        self._register(SignRule(42, "RIGHT_HAND_CURVE", "Right Hand Curve Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Road curves sharply right. Decelerate smoothly and maintain position within lane."))

        self._register(SignRule(43, "ROUNDABOUT", "Roundabout Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Traffic circle ahead. Slow down and yield to circulating traffic on the right."))

        self._register(SignRule(44, "SCHOOL_AHEAD", "School Zone Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "School zone. Reduce speed to at most 25 km/h and watch for children crossing.",
                                numerical_value=25.0, unit="km/h"))

        self._register(SignRule(45, "SIDE_ROAD_LEFT", "Side Road Left Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Junction with side road entering from left. Watch for joining traffic."))

        self._register(SignRule(46, "SIDE_ROAD_RIGHT", "Side Road Right Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Junction with side road entering from right. Scan right approach."))

        self._register(SignRule(47, "SLIPPERY_ROAD", "Slippery Road Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Road surface has low skid resistance. Avoid abrupt braking or rapid steering inputs."))

        self._register(SignRule(48, "SPEED_LIMIT_5", "Speed Limit 5 km/h",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                                "Maximum permissible speed is 5 km/h. Crawl speed required.",
                                numerical_value=5.0, unit="km/h"))

        self._register(SignRule(49, "SPEED_LIMIT_15", "Speed Limit 15 km/h",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                                "Maximum permissible speed is 15 km/h. Maintain strict low speed.",
                                numerical_value=15.0, unit="km/h"))

        self._register(SignRule(50, "SPEED_LIMIT_30", "Speed Limit 30 km/h",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                                "Maximum permissible speed is 30 km/h. Urban / residential zone compliance.",
                                numerical_value=30.0, unit="km/h"))

        self._register(SignRule(51, "SPEED_LIMIT_40", "Speed Limit 40 km/h",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                                "Maximum permissible speed is 40 km/h. Decelerate to posted maximum.",
                                numerical_value=40.0, unit="km/h"))

        self._register(SignRule(52, "SPEED_LIMIT_50", "Speed Limit 50 km/h",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                                "Maximum permissible speed is 50 km/h. City arterial speed compliance.",
                                numerical_value=50.0, unit="km/h"))

        self._register(SignRule(53, "SPEED_LIMIT_60", "Speed Limit 60 km/h",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                                "Maximum permissible speed is 60 km/h. Sub-arterial road speed limit.",
                                numerical_value=60.0, unit="km/h"))

        self._register(SignRule(54, "SPEED_LIMIT_70", "Speed Limit 70 km/h",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                                "Maximum permissible speed is 70 km/h. Highway sector speed ceiling.",
                                numerical_value=70.0, unit="km/h"))

        self._register(SignRule(55, "SPEED_LIMIT_80", "Speed Limit 80 km/h",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                                "Maximum permissible speed is 80 km/h. Expressway standard limit.",
                                numerical_value=80.0, unit="km/h"))

        self._register(SignRule(56, "STAGGERED_INTERSECTION", "Staggered Intersection Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Offset cross-junctions ahead. Watch for turning maneuvers from both sides."))

        self._register(SignRule(57, "STEEP_ASCENT", "Steep Ascent Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.INFO, EventType.HAZARD_WARNING,
                                "Steep incline ahead. Select appropriate lower gear and maintain headway."))

        self._register(SignRule(58, "STEEP_DESCENT", "Steep Descent Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Steep downhill gradient. Utilize engine braking; avoid excessive continuous foot-braking."))

        self._register(SignRule(59, "STOP", "Mandatory Stop",
                                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.STOP_MANDATE,
                                "Come to a complete stop before stop-line. Verify clearance before entering intersection."))

        self._register(SignRule(60, "STRAIGHT_PROHIBITED", "Straight Progression Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                                "Through traffic prohibited ahead. Divert left or right as directed."))

        self._register(SignRule(61, "T_INTERSECTION", "T-Intersection Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Road terminates perpendicularly at T-junction ahead. Prepare to turn left or right."))

        self._register(SignRule(62, "TRUCK_PROHIBITED", "Heavy Trucks Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                                "Heavy goods commercial vehicles prohibited on this roadway corridor."))

        self._register(SignRule(63, "TURN_RIGHT", "Mandatory Turn Right",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Mandatory right turn. All vehicles must follow right direction."))

        self._register(SignRule(64, "U_TURN_PROHIBITED", "U-Turn Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                                "U-turns strictly forbidden at this location. High collision hazard with oncoming traffic."))

        self._register(SignRule(65, "Y_INTERSECTION", "Y-Intersection Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Roadway forks into two branches. Choose proper lane early."))

        self._register(SignRule(66, "BULLOCK_AND_HANDCART_PROHIBITED", "Bullock Carts and Handcarts Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                                "All non-motorized animal-drawn and manual transport prohibited on this sector."))

        self._register(SignRule(67, "COMPULSARY_AHEAD_OR_TURN_LEFT", "Compulsory Ahead or Turn Left",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Movement restricted to straight through or left turn only. Right turns prohibited."))

        self._register(SignRule(68, "COMPULSARY_TURN_LEFT_AHEAD", "Compulsory Turn Left Ahead",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Mandatory left turn required at upcoming intersection."))

        self._register(SignRule(69, "COMPULSARY_TURN_RIGHT", "Compulsory Turn Right",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.RESTRICTION_ALERT,
                                "Mandatory right turn. Straight movement prohibited."))

        self._register(SignRule(70, "CYCLE_CROSSING", "Cycle Crossing Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Bicycle crossing ahead. Decelerate and give right of way to cyclists."))

        self._register(SignRule(71, "FERRY", "Ferry Boat Service Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.INFO, EventType.GENERAL_INFO,
                                "Road terminates at river ferry terminal. Slow down and follow marshalling."))

        self._register(SignRule(72, "LEFT_HAIR_PIN_BEND", "Left Hairpin Bend Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Sharp 180-degree left hairpin curve on hill sector. Reduce speed and sound horn."))

        self._register(SignRule(73, "LEFT_HAND_CURVE", "Left Hand Curve Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Curvature to left ahead. Decelerate and maintain lane discipline."))

        self._register(SignRule(74, "LENGTH_LIMIT", "Vehicle Length Limit",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                                "Maximum vehicle overall length restriction ahead. Long vehicles must divert.",
                                unit="metres"))

        self._register(SignRule(75, "LOAD_LIMIT", "Vehicle Load Limit",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                                "Gross vehicle weight restriction ahead. Overweight vehicles prohibited.",
                                unit="tonnes"))

        self._register(SignRule(76, "NARROW_BRIDGE", "Narrow Bridge Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Carriageway narrows across bridge deck. No overtaking; yield to vehicle already on bridge."))

        self._register(SignRule(77, "RIGHT_REVERSE_BEND", "Right Reverse Bend",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.HAZARD_WARNING,
                                "Double curve commencing with right bend. Decelerate and keep left."))

        self._register(SignRule(78, "RIGHT_TURN_PROHIBITED", "Right Turn Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.CRITICAL, EventType.PROHIBITION_ALERT,
                                "Right turn strictly prohibited at this intersection."))

        self._register(SignRule(79, "ROAD_WIDENS_AHEAD", "Road Widens Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.INFO, EventType.GENERAL_INFO,
                                "Additional lane or carriageway widening ahead."))

        self._register(SignRule(80, "SPEED_LIMIT_20", "Speed Limit 20 km/h",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.SPEED_LIMIT_POSTED,
                                "Maximum permissible speed is 20 km/h. Reduce speed accordingly.",
                                numerical_value=20.0, unit="km/h"))

        self._register(SignRule(81, "TONGA_PROHIBITED", "Tonga (Horse-Drawn) Prohibited",
                                RuleCategory.REGULATORY, SeverityLevel.INFO, EventType.PROHIBITION_ALERT,
                                "Horse-drawn passenger tongas prohibited on this arterial sector."))

        self._register(SignRule(82, "TRAFFIC_SIGNAL", "Traffic Signals Ahead",
                                RuleCategory.CAUTIONARY, SeverityLevel.WARNING, EventType.TRAFFIC_SIGNAL_AHEAD,
                                "Traffic light intersection ahead. Prepare to decelerate and stop on amber or red."))

        self._register(SignRule(83, "UNGUARDED_LEVEL_CROSSING", "Unguarded Railway Crossing",
                                RuleCategory.CAUTIONARY, SeverityLevel.CRITICAL, EventType.LEVEL_CROSSING_ALERT,
                                "Railway crossing without barrier gates ahead. Mandatory stop, look both directions, and listen."))

        self._register(SignRule(84, "WIDTH_LIMIT", "Vehicle Width Limit",
                                RuleCategory.REGULATORY, SeverityLevel.WARNING, EventType.RESTRICTION_ALERT,
                                "Maximum vehicle width restriction ahead. Wide-load vehicles prohibited.",
                                unit="metres"))
