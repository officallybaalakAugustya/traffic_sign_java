"""
Phase 9: CLI Demonstrator for Road Scene Sign Detection & Road Safety Monitor.
Supports headless execution, batch processing, and HUD overlay rendering.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from trafficsign.inference.pipeline import RoadSceneInferencePipeline
from trafficsign.safety.catalogue import SignRuleCatalogue
from trafficsign.safety.engine import SafetyEventEngine
from trafficsign.safety.state import SafetyState


def run_demonstration(input_path: Path, output_dir: Path, detector_pt: Path = None):
    print(f"\n=======================================================")
    print(f"INDIAN ROAD TRAFFIC SIGN RECOGNITION & SAFETY MONITOR")
    print(f"=======================================================")
    print(f"Device: {'NVIDIA RTX 4050 (CUDA)' if torch.cuda.is_available() else 'CPU'}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")

    pipeline = RoadSceneInferencePipeline(
        detector_checkpoint=detector_pt,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    catalogue = SignRuleCatalogue()
    safety_state = SafetyState()
    engine = SafetyEventEngine(catalogue, safety_state)

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(
            list(input_path.glob("*.jpg")) + list(input_path.glob("*.png")) + list(input_path.glob("*.jpeg"))
        )
    else:
        print(f"Error: Path {input_path} does not exist.")
        return

    print(f"Processing {len(image_files)} frames...\n")

    total_t0 = time.time()
    all_events = []

    for idx, img_file in enumerate(image_files):
        t0 = time.time()
        result = pipeline.process_image(img_file)
        dt = (time.time() - t0) * 1000

        # Annotate
        annotated = pipeline.draw_detections(img_file, result)
        out_file = output_dir / f"hud_{img_file.name}"
        annotated.save(out_file)

        # Process through Safety Event Engine
        frame_events = []
        for det in result.detections:
            evs = engine.process_detection(
                class_id=det.predicted_class_id,
                class_confidence=det.classification_confidence,
                detection_confidence=det.detection_confidence,
                box=det.normalized_box,
                timestamp_ms=int(time.time() * 1000)
            )
            frame_events.extend(evs)
            all_events.extend(evs)

        current_speed = safety_state.current_posted_speed_limit
        active_rest = safety_state.active_restrictions

        print(f"[{idx+1}/{len(image_files)}] {img_file.name} | Latency: {dt:.1f}ms | Detections: {len(result.detections)} | Speed Limit: {current_speed or 'None'} km/h")
        for det in result.detections:
            print(f"   -> Sign: {det.predicted_class_name} ({det.category}) | Det Conf: {det.detection_confidence*100:.1f}% | Cls Conf: {det.classification_confidence*100:.1f}% | State: {det.uncertainty_state.value}")
        for ev in frame_events:
            print(f"   [!] Safety Advisory: {ev.driver_advisory} (Severity: {ev.severity.name})")

    total_elapsed = time.time() - total_t0
    avg_fps = len(image_files) / total_elapsed if total_elapsed > 0 else 0.0

    print(f"\nDemonstration Complete!")
    print(f"Total time: {total_elapsed:.2f}s | Average FPS: {avg_fps:.1f}")
    print(f"Annotated frames with HUD saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CLI Demonstrator for Road Scene Sign Monitor")
    parser.add_argument("--input", "-i", type=str, default="data/processed/test_200", help="Input image or directory")
    parser.add_argument("--output", "-o", type=str, default="reports/demo_output", help="Output directory")
    parser.add_argument("--detector", "-d", type=str, default=None, help="Detector checkpoint path (.pt or .params)")
    args = parser.parse_args()

    input_path = REPO_ROOT / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_dir = REPO_ROOT / args.output if not Path(args.output).is_absolute() else Path(args.output)
    detector_path = Path(args.detector) if args.detector else None

    run_demonstration(input_path, output_dir, detector_path)


if __name__ == "__main__":
    main()
