"""
Traffic Sign Taxonomy conforming to IRC:67 Code of Practice for Road Signs (MoRTH).
Mirroring com.trafficsign.data.IndianTrafficSignTaxonomy and TrafficSignClass.
"""

from __future__ import annotations
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class IndianSignClassInfo:
    id: int
    canonical_name: str
    dataset_label: str
    category: str
    irc_code: str
    speed_limit_kmh: Optional[int]
    safety_implication: str


@dataclass(frozen=True)
class TrafficSignClass:
    class_id: int
    name: str
    category: str
    description: str


class IndianTrafficSignTaxonomy:
    """
    Concrete implementation of TrafficSignTaxonomy for Indian traffic signs
    conforming to IRC:67 Code of Practice for Road Signs (MoRTH).
    """

    DEFAULT_TOTAL_CLASSES = 85

    def __init__(self, json_path: Optional[Union[str, Path]] = None):
        self._classes_by_id: Dict[int, IndianSignClassInfo] = {}
        self._label_to_id: Dict[str, int] = {}

        loaded = False
        candidates = []
        if json_path is not None:
            candidates.append(Path(json_path))

        # Check standard locations
        curr_dir = Path(__file__).parent
        candidates.append(curr_dir.parent / "resources" / "indian_traffic_signs_irc67.json")
        candidates.append(Path("src/main/resources/taxonomies/indian_traffic_signs_irc67.json"))
        candidates.append(Path("S:/TrafficSignRecognitionRoadSafetyMonitor/src/main/resources/taxonomies/indian_traffic_signs_irc67.json"))

        for p in candidates:
            if p.exists():
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for item in data.get("classes", []):
                        info = IndianSignClassInfo(
                            id=item["id"],
                            canonical_name=item["canonicalName"],
                            dataset_label=item["datasetLabel"],
                            category=item["category"],
                            irc_code=item.get("ircCode", "IRC:67"),
                            speed_limit_kmh=item.get("speedLimitKmH"),
                            safety_implication=item.get("safetyImplication", "")
                        )
                        self._register_class(info)
                    loaded = True
                    break
                except Exception:
                    pass

        if not loaded or len(self._classes_by_id) == 0:
            self._load_fallback_defaults()

    def _register_class(self, info: IndianSignClassInfo) -> None:
        self._classes_by_id[info.id] = info
        self._label_to_id[info.canonical_name.upper()] = info.id
        self._label_to_id[info.dataset_label.upper()] = info.id
        normalized = info.dataset_label.upper().replace("-", "_").replace(" ", "_")
        self._label_to_id[normalized] = info.id

    def _load_fallback_defaults(self) -> None:
        # Fallback 85 classes
        defaults = [
            (0, "All Motor Vehicles Prohibited", "ALL_MOTOR_VEHICLE_PROHIBITED", "MANDATORY", None),
            (1, "Axle Load Limit", "AXLE_LOAD_LIMIT", "MANDATORY", None),
            (2, "Barrier Ahead", "BARRIER_AHEAD", "CAUTIONARY", None),
            (3, "Bullock Carts Prohibited", "BULLOCK_PROHIBITED", "MANDATORY", None),
            (4, "Cattle Crossing", "CATTLE", "CAUTIONARY", None),
            (5, "Compulsory Ahead", "COMPULSARY_AHEAD", "MANDATORY", None),
            (6, "Compulsory Ahead or Turn Right", "COMPULSARY_AHEAD_OR_TURN_RIGHT", "MANDATORY", None),
            (7, "Compulsory Cycle Track", "COMPULSARY_CYCLE_TRACK", "MANDATORY", None),
            (8, "Compulsory Keep Left", "COMPULSARY_KEEP_LEFT", "MANDATORY", None),
            (9, "Compulsory Keep Right", "COMPULSARY_KEEP_RIGHT", "MANDATORY", None),
            (10, "Compulsory Minimum Speed", "COMPULSARY_MINIMUM_SPEED", "MANDATORY", None),
            (11, "Compulsory Sound Horn", "COMPULSARY_SOUND_HORN", "MANDATORY", None),
            (12, "Compulsory Turn Left", "COMPULSARY_TURN_LEFT", "MANDATORY", None),
            (13, "Compulsory Turn Right Ahead", "COMPULSARY_TURN_RIGHT_AHEAD", "MANDATORY", None),
            (14, "Cross Road", "CROSS_ROAD", "CAUTIONARY", None),
            (15, "Cycles Prohibited", "CYCLE_PROHIBITED", "MANDATORY", None),
            (16, "Dangerous Dip", "DANGEROUS_DIP", "CAUTIONARY", None),
            (17, "Direction", "DIRECTION", "INFORMATORY", None),
            (18, "Falling Rocks", "FALLING_ROCKS", "CAUTIONARY", None),
            (19, "Gap in Median", "GAP_IN_MEDIAN", "CAUTIONARY", None),
            (20, "Give Way", "GIVE_WAY", "MANDATORY", None),
            (21, "Guarded Level Crossing", "GUARDED_LEVEL_CROSSING", "CAUTIONARY", None),
            (22, "Handcarts Prohibited", "HANDCART_PROHIBITED", "MANDATORY", None),
            (23, "Height Limit", "HEIGHT_LIMIT", "MANDATORY", None),
            (24, "Horn Prohibited", "HORN_PROHIBITED", "MANDATORY", None),
            (25, "Hump or Rough Road", "HUMP_OR_ROUGH_ROAD", "CAUTIONARY", None),
            (26, "Left Reverse Bend", "LEFT_REVERSE_BEND", "CAUTIONARY", None),
            (27, "Left Turn Prohibited", "LEFT_TURN_PROHIBITED", "MANDATORY", None),
            (28, "Loose Gravel", "LOOSE_GRAVEL", "CAUTIONARY", None),
            (29, "Men at Work", "MEN_AT_WORK", "CAUTIONARY", None),
            (30, "Narrow Road Ahead", "NARROW_ROAD_AHEAD", "CAUTIONARY", None),
            (31, "No Entry", "NO_ENTRY", "MANDATORY", None),
            (32, "No Parking", "NO_PARKING", "MANDATORY", None),
            (33, "No Stopping", "NO_STOPPING", "MANDATORY", None),
            (34, "Overtaking Prohibited", "OVERTAKING_PROHIBITED", "MANDATORY", None),
            (35, "Pedestrian Crossing", "PEDESTRIAN_CROSSING", "CAUTIONARY", None),
            (36, "Pedestrians Prohibited", "PEDESTRIAN_PROHIBITED", "MANDATORY", None),
            (37, "Priority for Oncoming Traffic", "PRIORITY_FOR_ONCOMING_TRAFFIC", "MANDATORY", None),
            (38, "Quay or River Bank", "QUAY_OR_RIVER_BANK", "CAUTIONARY", None),
            (39, "Restriction Ends", "RESTRICTION_ENDS", "MANDATORY", None),
            (40, "Right Hand Curve", "RIGHT_HAND_CURVE", "CAUTIONARY", None),
            (41, "Right Reverse Bend", "RIGHT_REVERSE_BEND", "CAUTIONARY", None),
            (42, "Right Turn Prohibited", "RIGHT_TURN_PROHIBITED", "MANDATORY", None),
            (43, "Road Closes Ahead", "ROAD_CLOSES_AHEAD", "INFORMATORY", None),
            (44, "Roundabout", "ROUNDABOUT", "CAUTIONARY", None),
            (45, "School Ahead", "SCHOOL_AHEAD", "CAUTIONARY", None),
            (46, "Side Road Left", "SIDE_ROAD_LEFT", "CAUTIONARY", None),
            (47, "Side Road Right", "SIDE_ROAD_RIGHT", "CAUTIONARY", None),
            (48, "Slippery Road", "SLIPPERY_ROAD", "CAUTIONARY", None),
            (49, "Speed Limit 20", "SPEED_LIMIT_20", "MANDATORY", 20),
            (50, "Speed Limit 30", "SPEED_LIMIT_30", "MANDATORY", 30),
            (51, "Speed Limit 40", "SPEED_LIMIT_40", "MANDATORY", 40),
            (52, "Speed Limit 50", "SPEED_LIMIT_50", "MANDATORY", 50),
            (53, "Speed Limit 60", "SPEED_LIMIT_60", "MANDATORY", 60),
            (54, "Speed Limit 70", "SPEED_LIMIT_70", "MANDATORY", 70),
            (55, "Speed Limit 80", "SPEED_LIMIT_80", "MANDATORY", 80),
            (56, "Speed Limit 90", "SPEED_LIMIT_90", "MANDATORY", 90),
            (57, "Speed Limit 100", "SPEED_LIMIT_100", "MANDATORY", 100),
            (58, "Speed Limit 110", "SPEED_LIMIT_110", "MANDATORY", 110),
            (59, "Speed Limit 120", "SPEED_LIMIT_120", "MANDATORY", 120),
            (60, "Stop", "STOP", "MANDATORY", None),
            (61, "Straight Prohibited", "STRAIGHT_PROHIBITED", "MANDATORY", None),
            (62, "T-Intersection", "T_INTERSECTION", "CAUTIONARY", None),
            (63, "Traffic Signal Ahead", "TRAFFIC_SIGNAL_AHEAD", "CAUTIONARY", None),
            (64, "Trucks Prohibited", "TRUCK_PROHIBITED", "MANDATORY", None),
            (65, "U-Turn Prohibited", "U_TURN_PROHIBITED", "MANDATORY", None),
        ]
        for item in defaults:
            cid, cname, dlabel, cat, spd = item
            self._register_class(IndianSignClassInfo(
                id=cid,
                canonical_name=cname,
                dataset_label=dlabel,
                category=cat,
                irc_code="IRC:67",
                speed_limit_kmh=spd,
                safety_implication=""
            ))

    def get_class_info(self, class_id: int) -> Optional[IndianSignClassInfo]:
        return self._classes_by_id.get(class_id)

    def get_class_by_label(self, label: str) -> Optional[IndianSignClassInfo]:
        if not label:
            return None
        norm = label.strip().upper().replace("-", "_").replace(" ", "_")
        cid = self._label_to_id.get(norm)
        if cid is not None:
            return self._classes_by_id.get(cid)
        return None

    def get_total_classes(self) -> int:
        return len(self._classes_by_id)

    def get_all_classes(self) -> List[IndianSignClassInfo]:
        return list(self._classes_by_id.values())
