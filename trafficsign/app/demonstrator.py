"""
Road Safety Decision-Support Monitor CLI Demonstrator
mirroring com.trafficsign.app.RoadSafetyMonitorDemonstrator.
Simulates real-world driving scenarios (Highway speed limit, Urban intersection,
Ghat road curve, Empty background) and produces comprehensive audit exports.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path
import time
from typing import List, Optional

import numpy as np
from PIL import Image

from trafficsign.inference.pipeline import PipelineResult, RoadSceneInferencePipeline
from trafficsign.safety.catalogue import SignRuleCatalogue
from trafficsign.safety.engine import SafetyEvent, SafetyEventEngine
from trafficsign.safety.state import SafetyState


@dataclass
class ScenarioRunStats:
    scenario_name: str
    environment_type: str
    total_frames: int
    total_detections: int
    unique_tracks: int
    deduplicated_events: int
    avg_latency_ms: float
    throughput_fps: float


class RoadSafetyMonitorDemonstrator:
    """
    Simulates multi-frame driving scenarios through the inference pipeline and safety event engine.
    """

    def __init__(self):
        self.catalogue = SignRuleCatalogue()
        self.safety_state = SafetyState()
        self.engine = SafetyEventEngine(self.catalogue, self.safety_state)
        self.pipeline = RoadSceneInferencePipeline()

    def execute(self) -> List[ScenarioRunStats]:
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        evidence_dir = reports_dir / "evidence"
        evidence_dir.mkdir(parents=True, exist_ok=True)

        self.engine.evidence_directory = evidence_dir
        self.engine.save_evidence_snapshots = True

        scenarios = [
            ("Highway Speed Enforcement", "Expressway / Arterial", self._collect_sample_frames("highway")),
            ("Urban Junction & Pedestrian", "City Intersections", self._collect_sample_frames("urban")),
            ("Ghat Curve & Hazard Warning", "Hill Road / Hairpin", self._collect_sample_frames("ghat")),
            ("Empty Highway Negative", "Background Only", self._collect_sample_frames("empty")),
        ]

        stats_list: List[ScenarioRunStats] = []

        print("=" * 80)
        print("INDIAN ROAD SAFETY DECISION-SUPPORT MONITOR - DEMONSTRATION")
        print("=" * 80)

        for name, env_type, frames in scenarios:
            print(f"\n--- Running Scenario: {name} ({env_type}) ---")
            self.engine.reset()

            total_dets = 0
            latencies = []

            for idx, frame_path in enumerate(frames):
                raw_img = Image.open(frame_path).convert("RGB")
                t0 = time.time()
                res = self.pipeline.process_image(raw_img)
                dur = int((time.time() - t0) * 1000)
                latencies.append(dur)
                total_dets += len(res.detections)

                # Process safety events
                new_evts = self.engine.process_frame(idx, int(time.time() * 1000), raw_img, res.detections)

                for e in new_evts:
                    print(f"  [Frame {idx:03d}] {e.severity.name}: {e.display_name} -> {e.driver_advisory}")

            avg_lat = float(np.mean(latencies)) if latencies else 0.0
            fps = (1000.0 / avg_lat) if avg_lat > 0 else 0.0
            unique_tracks = len(set(e.track_id for e in self.engine.event_log))
            dedup_events = len(self.engine.event_log)

            stat = ScenarioRunStats(
                scenario_name=name,
                environment_type=env_type,
                total_frames=len(frames),
                total_detections=total_dets,
                unique_tracks=unique_tracks,
                deduplicated_events=dedup_events,
                avg_latency_ms=avg_lat,
                throughput_fps=fps
            )
            stats_list.append(stat)

            print(f"  Summary: {len(frames)} frames, {total_dets} detections, {dedup_events} safety events. Avg Latency: {avg_lat:.1f}ms ({fps:.1f} FPS)")

        # Export aggregate event log
        self.engine.export_to_csv("reports/safety_event_log.csv")
        self.engine.export_to_json("reports/safety_event_log.json")
        print(f"\nExported event logs to reports/safety_event_log.csv and reports/safety_event_log.json")

        return stats_list

    def _collect_sample_frames(self, tag: str) -> List[Path]:
        search_dirs = [
            Path("reports/evidence"),
            Path("reports/task16_visualizations"),
            Path("data/raw/indian-road-driving-sample/inspection"),
        ]
        all_imgs = []
        for d in search_dirs:
            if d.exists():
                all_imgs.extend(list(d.glob("*.png")) + list(d.glob("*.jpg")))

        if not all_imgs:
            # Generate synthetic frames if none exist
            dummy_dir = Path("reports/synthetic_frames")
            dummy_dir.mkdir(parents=True, exist_ok=True)
            for i in range(5):
                p = dummy_dir / f"dummy_{tag}_{i}.png"
                if not p.exists():
                    img = Image.new("RGB", (640, 480), (60 + i * 10, 70, 80))
                    img.save(p)
                all_imgs.append(p)

        return all_imgs[:6]


def main():
    demo = RoadSafetyMonitorDemonstrator()
    demo.execute()


if __name__ == "__main__":
    main()
