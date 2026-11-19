"""
Safety State machine tracking posted speed limits, active prohibitions, and hazards.
Mirroring com.trafficsign.safety.SafetyState.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Set
from trafficsign.safety.catalogue import EventType


class SafetyState:
    """
    Maintains active driver safety state including speed limits, prohibitions, and transient hazards.
    """

    def __init__(self):
        self.current_posted_speed_limit: Optional[int] = None
        self.active_restrictions: Set[str] = set()
        self.hazard_expiry_map: Dict[str, int] = {}
        self.last_alert: Optional[object] = None
        self.last_updated_frame: int = 0

    def update_with_event(self, event: object) -> None:
        if event is None:
            return

        self.last_alert = event
        self.last_updated_frame = getattr(event, "frame_index", 0)

        event_type = getattr(event, "event_type", None)
        if event_type == EventType.UNCERTAIN_DETECTION or event_type == "UNCERTAIN_DETECTION":
            return

        display_name = getattr(event, "display_name", "")
        num_val = getattr(event, "numerical_value", None)

        if event_type == EventType.SPEED_LIMIT_POSTED or event_type == "SPEED_LIMIT_POSTED":
            if num_val is not None:
                self.current_posted_speed_limit = int(round(num_val))
        elif event_type == EventType.RESTRICTION_ENDS or event_type == "RESTRICTION_ENDS":
            self.active_restrictions.clear()
        elif event_type in (EventType.PROHIBITION_ALERT, EventType.RESTRICTION_ALERT,
                            "PROHIBITION_ALERT", "RESTRICTION_ALERT"):
            self.active_restrictions.add(display_name)
        elif event_type in (EventType.HAZARD_WARNING, EventType.LEVEL_CROSSING_ALERT,
                            EventType.TRAFFIC_SIGNAL_AHEAD, EventType.STOP_MANDATE,
                            EventType.GIVE_WAY_MANDATE, "HAZARD_WARNING",
                            "LEVEL_CROSSING_ALERT", "TRAFFIC_SIGNAL_AHEAD",
                            "STOP_MANDATE", "GIVE_WAY_MANDATE"):
            self.hazard_expiry_map[display_name] = self.last_updated_frame

    def prune(self, current_frame: int, max_ttl_frames: int = 60) -> None:
        expired = [k for k, frame in self.hazard_expiry_map.items() if (current_frame - frame) > max_ttl_frames]
        for k in expired:
            del self.hazard_expiry_map[k]

    def reset(self) -> None:
        self.current_posted_speed_limit = None
        self.active_restrictions.clear()
        self.hazard_expiry_map.clear()
        self.last_alert = None
        self.last_updated_frame = 0

    def get_current_posted_speed_limit(self) -> Optional[int]:
        return self.current_posted_speed_limit

    def get_active_restrictions(self) -> Set[str]:
        return set(self.active_restrictions)

    def get_active_hazards(self) -> List[str]:
        return list(self.hazard_expiry_map.keys())

    def get_last_alert(self) -> Optional[object]:
        return self.last_alert

    def get_last_updated_frame(self) -> int:
        return self.last_updated_frame
