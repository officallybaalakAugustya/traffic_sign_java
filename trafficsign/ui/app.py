"""
Desktop Sign-Based Road Safety Decision-Support Monitor MVP
mirroring com.trafficsign.ui.RoadSafetyMonitorApp.
Built with Python, Tkinter, Pillow, and PyTorch.
Features:
- Video & image frame canvas with uncertainty-coded bounding boxes and pills.
- HUD with IRC:67 Speed Limit Circle, Driver Advisory Banner, Restrictions & Hazards.
- Deduplicated Safety Event Table with Operator Review (Confirm, Flag, Dismiss).
- Runtime configuration sliders for Detection, Classification, NMS, and Proposal Cap.
"""

from __future__ import annotations
import math
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

from trafficsign.inference.pipeline import PipelineResult, RoadSceneDetection, RoadSceneInferencePipeline, UncertaintyState
from trafficsign.safety.catalogue import EventType, SeverityLevel, SignRuleCatalogue
from trafficsign.safety.engine import OperatorStatus, SafetyEvent, SafetyEventEngine
from trafficsign.safety.state import SafetyState


class RoadSafetyMonitorApp:
    """
    Sign-Based Road Safety Decision-Support Monitor GUI Application.
    """

    def __init__(self, root: Optional[tk.Tk] = None, pipeline: Optional[RoadSceneInferencePipeline] = None):
        self.root = root or tk.Tk()
        self.root.title("Indian Road Safety Decision-Support Monitor (IRC:67)")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 750)

        # Apply dark slate theme styling
        self.root.configure(bg="#1E222A")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", background="#1E222A", foreground="#E0E6ED", font=("Segoe UI", 9))
        style.configure("TLabel", background="#1E222A", foreground="#E0E6ED")
        style.configure("TFrame", background="#1E222A")
        style.configure("TButton", background="#2D333F", foreground="#FFFFFF", borderwidth=1)
        style.map("TButton", background=[("active", "#3E4756")])
        style.configure("Treeview", background="#282C34", foreground="#FFFFFF", fieldbackground="#282C34", rowheight=24)
        style.map("Treeview", background=[("selected", "#0D6EFD")])

        # Pipeline & Engine
        self.pipeline = pipeline or RoadSceneInferencePipeline()
        self.catalogue = SignRuleCatalogue()
        self.safety_state = SafetyState()
        self.engine = SafetyEventEngine(self.catalogue, self.safety_state)

        # Playback state
        self.image_files: List[Path] = []
        self.current_frame_idx: int = 0
        self.is_playing: bool = False
        self.play_thread: Optional[threading.Thread] = None

        self._build_ui()
        self._load_default_sample()

    def _build_ui(self) -> None:
        # Top toolbar
        toolbar = ttk.Frame(self.root, padding=(10, 6))
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Image", command=self._open_image).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Load Directory / Video", command=self._open_directory).pack(side=tk.LEFT, padx=4)
        self.play_btn = ttk.Button(toolbar, text="Play", command=self._toggle_play)
        self.play_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Step Next", command=self._step_frame).pack(side=tk.LEFT, padx=4)
        ttk.Button(toolbar, text="Reset", command=self._reset_monitor).pack(side=tk.LEFT, padx=4)

        self.status_lbl = ttk.Label(toolbar, text="Ready. Load an image or folder to begin inference.", font=("Segoe UI", 10, "bold"))
        self.status_lbl.pack(side=tk.LEFT, padx=20)

        # Main splitter (Canvas Left, HUD Right)
        content_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        content_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Left: Video & Driving Scene Canvas
        canvas_frame = ttk.Frame(content_pane, padding=4)
        content_pane.add(canvas_frame, weight=3)

        self.canvas = tk.Canvas(canvas_frame, bg="#12151B", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Right: Safety HUD & Runtime Settings
        right_panel = ttk.Frame(content_pane, padding=6)
        content_pane.add(right_panel, weight=1)

        # HUD Speed Limit Sign & Advisory
        hud_box = ttk.LabelFrame(right_panel, text="Statutory Road Safety HUD", padding=10)
        hud_box.pack(fill=tk.X, pady=4)

        self.hud_speed_canvas = tk.Canvas(hud_box, width=120, height=120, bg="#1E222A", highlightthickness=0)
        self.hud_speed_canvas.pack(pady=4)
        self._draw_speed_limit_circle(None)

        self.advisory_var = tk.StringVar(value="Normal driving conditions. Obey prevailing traffic rules.")
        self.advisory_lbl = tk.Label(
            hud_box, textvariable=self.advisory_var, bg="#232731", fg="#00CC44",
            font=("Segoe UI", 10, "bold"), wraplength=280, justify=tk.CENTER, padx=8, pady=8
        )
        self.advisory_lbl.pack(fill=tk.X, pady=6)

        # Restrictions & Hazards
        rest_box = ttk.LabelFrame(right_panel, text="Active Restrictions & Hazards", padding=6)
        rest_box.pack(fill=tk.X, pady=4)

        self.restrictions_var = tk.StringVar(value="None")
        ttk.Label(rest_box, text="Prohibitions / Restrictions:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        ttk.Label(rest_box, textvariable=self.restrictions_var, foreground="#FF9900", wraplength=280).pack(anchor=tk.W, pady=(0, 4))

        self.hazards_var = tk.StringVar(value="None")
        ttk.Label(rest_box, text="Expiring Hazards:", font=("Segoe UI", 9, "bold")).pack(anchor=tk.W)
        ttk.Label(rest_box, textvariable=self.hazards_var, foreground="#FFCC00", wraplength=280).pack(anchor=tk.W)

        # Sliders Frame
        sliders_box = ttk.LabelFrame(right_panel, text="Operating Parameters", padding=6)
        sliders_box.pack(fill=tk.X, pady=6)

        # Detection Threshold
        self.det_thresh_val = tk.DoubleVar(value=0.50)
        ttk.Label(sliders_box, text="Detector Conf Threshold:").pack(anchor=tk.W)
        det_scale = ttk.Scale(sliders_box, from_=0.05, to=0.95, variable=self.det_thresh_val, command=self._on_params_changed)
        det_scale.pack(fill=tk.X)
        self.det_lbl = ttk.Label(sliders_box, text="0.50")
        self.det_lbl.pack(anchor=tk.E)

        # Classification Threshold
        self.cls_thresh_val = tk.DoubleVar(value=0.25)
        ttk.Label(sliders_box, text="Classifier Conf Threshold:").pack(anchor=tk.W)
        cls_scale = ttk.Scale(sliders_box, from_=0.05, to=0.95, variable=self.cls_thresh_val, command=self._on_params_changed)
        cls_scale.pack(fill=tk.X)
        self.cls_lbl = ttk.Label(sliders_box, text="0.25")
        self.cls_lbl.pack(anchor=tk.E)

        # NMS IoU Threshold
        self.nms_thresh_val = tk.DoubleVar(value=0.40)
        ttk.Label(sliders_box, text="NMS IoU Threshold:").pack(anchor=tk.W)
        nms_scale = ttk.Scale(sliders_box, from_=0.10, to=0.90, variable=self.nms_thresh_val, command=self._on_params_changed)
        nms_scale.pack(fill=tk.X)
        self.nms_lbl = ttk.Label(sliders_box, text="0.40")
        self.nms_lbl.pack(anchor=tk.E)

        # Max Proposals Cap
        self.prop_cap_val = tk.IntVar(value=25)
        ttk.Label(sliders_box, text="Max Proposals Cap:").pack(anchor=tk.W)
        prop_scale = ttk.Scale(sliders_box, from_=5, to=100, variable=self.prop_cap_val, command=self._on_params_changed)
        prop_scale.pack(fill=tk.X)
        self.prop_lbl = ttk.Label(sliders_box, text="25")
        self.prop_lbl.pack(anchor=tk.E)

        # Bottom Panel: Deduplicated Event Table & Operator Review
        bottom_frame = ttk.LabelFrame(self.root, text="Deduplicated Safety Event Log & Operator Review", padding=6)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=False, padx=10, pady=5)
        bottom_frame.configure(height=220)

        # Action Buttons
        btn_bar = ttk.Frame(bottom_frame)
        btn_bar.pack(side=tk.TOP, fill=tk.X, pady=(0, 4))
        ttk.Button(btn_bar, text="Confirm / Acknowledge", command=lambda: self._review_selected(OperatorStatus.CONFIRMED)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="Flag for Inspection", command=lambda: self._review_selected(OperatorStatus.FLAGGED)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="Dismiss / Reject", command=lambda: self._review_selected(OperatorStatus.REJECTED)).pack(side=tk.LEFT, padx=4)
        ttk.Button(btn_bar, text="Export CSV", command=self._export_csv).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_bar, text="Export JSON", command=self._export_json).pack(side=tk.RIGHT, padx=4)

        # Event Treeview Table
        cols = ("frame", "track", "sign", "category", "severity", "det_conf", "cls_conf", "advisory", "status")
        self.tree = ttk.Treeview(bottom_frame, columns=cols, show="headings", height=5)
        self.tree.heading("frame", text="Frame")
        self.tree.heading("track", text="Track ID")
        self.tree.heading("sign", text="Sign Name")
        self.tree.heading("category", text="Category")
        self.tree.heading("severity", text="Severity")
        self.tree.heading("det_conf", text="Det Conf")
        self.tree.heading("cls_conf", text="Cls Conf")
        self.tree.heading("advisory", text="Driver Statutory Advisory")
        self.tree.heading("status", text="Operator Status")

        self.tree.column("frame", width=60, anchor=tk.CENTER)
        self.tree.column("track", width=70, anchor=tk.CENTER)
        self.tree.column("sign", width=180, anchor=tk.W)
        self.tree.column("category", width=100, anchor=tk.CENTER)
        self.tree.column("severity", width=80, anchor=tk.CENTER)
        self.tree.column("det_conf", width=70, anchor=tk.CENTER)
        self.tree.column("cls_conf", width=70, anchor=tk.CENTER)
        self.tree.column("advisory", width=420, anchor=tk.W)
        self.tree.column("status", width=110, anchor=tk.CENTER)

        tree_scroll = ttk.Scrollbar(bottom_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _draw_speed_limit_circle(self, speed_limit: Optional[int]) -> None:
        self.hud_speed_canvas.delete("all")
        cx, cy, r = 60, 60, 50

        if speed_limit is not None:
            # Red outer circular border (IRC:67 standard)
            self.hud_speed_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#FFFFFF", outline="#E60000", width=10)
            self.hud_speed_canvas.create_text(cx, cy - 6, text=str(speed_limit), font=("Segoe UI", 26, "bold"), fill="#000000")
            self.hud_speed_canvas.create_text(cx, cy + 22, text="km/h", font=("Segoe UI", 9, "bold"), fill="#555555")
        else:
            # Grey neutral circle
            self.hud_speed_canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill="#232731", outline="#4A5260", width=4)
            self.hud_speed_canvas.create_text(cx, cy - 6, text="---", font=("Segoe UI", 22, "bold"), fill="#8892B0")
            self.hud_speed_canvas.create_text(cx, cy + 20, text="NO POSTED LIMIT", font=("Segoe UI", 7, "bold"), fill="#8892B0")

    def _on_params_changed(self, _=None) -> None:
        d_val = round(self.det_thresh_val.get(), 2)
        c_val = round(self.cls_thresh_val.get(), 2)
        n_val = round(self.nms_thresh_val.get(), 2)
        p_val = int(self.prop_cap_val.get())

        self.det_lbl.config(text=f"{d_val:.2f}")
        self.cls_lbl.config(text=f"{c_val:.2f}")
        self.nms_lbl.config(text=f"{n_val:.2f}")
        self.prop_lbl.config(text=str(p_val))

        self.pipeline.detection_confidence_threshold = d_val
        self.pipeline.classification_confidence_threshold = c_val
        self.pipeline.nms_iou_threshold = n_val
        self.pipeline.max_proposals_per_frame = p_val

    def _open_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Open Driving Scene Image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.ppm"), ("All files", "*.*")]
        )
        if file_path:
            self.image_files = [Path(file_path)]
            self.current_frame_idx = 0
            self._process_and_display_current()

    def _open_directory(self) -> None:
        dir_path = filedialog.askdirectory(title="Open Road Driving Sequences Directory")
        if dir_path:
            p = Path(dir_path)
            files = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.ppm"):
                files.extend(sorted(p.rglob(ext)))
            if files:
                self.image_files = files
                self.current_frame_idx = 0
                self._process_and_display_current()
            else:
                messagebox.showwarning("No Images Found", "No image files found in selected directory.")

    def _toggle_play(self) -> None:
        if self.is_playing:
            self.is_playing = False
            self.play_btn.config(text="Play")
        else:
            if not self.image_files:
                return
            self.is_playing = True
            self.play_btn.config(text="Pause")
            self.play_thread = threading.Thread(target=self._play_loop, daemon=True)
            self.play_thread.start()

    def _play_loop(self) -> None:
        while self.is_playing and self.current_frame_idx < len(self.image_files) - 1:
            self.current_frame_idx += 1
            self.root.after(0, self._process_and_display_current)
            time.sleep(0.08)
        self.is_playing = False
        self.root.after(0, lambda: self.play_btn.config(text="Play"))

    def _step_frame(self) -> None:
        if self.image_files and self.current_frame_idx < len(self.image_files) - 1:
            self.current_frame_idx += 1
            self._process_and_display_current()

    def _reset_monitor(self) -> None:
        self.is_playing = False
        self.play_btn.config(text="Play")
        self.current_frame_idx = 0
        self.engine.reset()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self._update_hud(None, [], [])
        if self.image_files:
            self._process_and_display_current()

    def _load_default_sample(self) -> None:
        candidates = [
            Path("reports/evidence"),
            Path("reports/detection_demo/annotated_driving_frame.png"),
            Path("reports/task16_visualizations"),
            Path("data/raw/indian-road-driving-sample/inspection"),
        ]
        files = []
        for c in candidates:
            if c.is_file():
                files.append(c)
            elif c.is_dir():
                for ext in ("*.png", "*.jpg"):
                    files.extend(list(c.glob(ext)))
        if files:
            self.image_files = files[:50]
            self.current_frame_idx = 0
            self._process_and_display_current()

    def _process_and_display_current(self) -> None:
        if not self.image_files or self.current_frame_idx >= len(self.image_files):
            return

        img_path = self.image_files[self.current_frame_idx]
        frame_idx = self.current_frame_idx

        # 1. Pipeline inference
        t0 = time.time()
        raw_pil = Image.open(img_path).convert("RGB")
        result: PipelineResult = self.pipeline.process_image(raw_pil)
        annotated_pil = self.pipeline.draw_detections(raw_pil, result)

        # 2. Safety Event Engine tracking & state update
        new_events: List[SafetyEvent] = self.engine.process_frame(
            frame_index=frame_idx,
            timestamp_ms=int(time.time() * 1000),
            full_frame=raw_pil,
            detections=result.detections
        )

        # 3. Update Canvas
        cw = max(400, self.canvas.winfo_width())
        ch = max(300, self.canvas.winfo_height())
        annotated_pil.thumbnail((cw, ch), Image.Resampling.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(annotated_pil)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._tk_img, anchor=tk.CENTER)

        # 4. Update HUD
        speed = self.safety_state.get_current_posted_speed_limit()
        restrictions = self.safety_state.get_active_restrictions()
        hazards = self.safety_state.get_active_hazards()
        self._update_hud(speed, restrictions, hazards)

        # 5. Populate Event Table
        for e in new_events:
            self.tree.insert("", 0, values=(
                e.frame_index,
                e.track_id,
                e.display_name,
                str(e.category),
                str(e.severity),
                f"{e.detection_confidence * 100:.0f}%",
                f"{e.classification_confidence * 100:.0f}%",
                e.driver_advisory,
                str(e.operator_status)
            ))

        dur = int((time.time() - t0) * 1000)
        self.status_lbl.config(
            text=f"Frame [{frame_idx + 1}/{len(self.image_files)}] {img_path.name} | "
                 f"Latency: {result.total_latency_ms} ms ({result.throughput_fps:.1f} FPS) | "
                 f"Detections: {len(result.detections)}"
        )

    def _update_hud(self, speed: Optional[int], restrictions: Sequence[str], hazards: Sequence[str]) -> None:
        self._draw_speed_limit_circle(speed)

        # Advisory text & color
        if speed is not None:
            self.advisory_var.set(f"POSTED SPEED LIMIT: {speed} km/h. Maintain strict speed compliance.")
            self.advisory_lbl.config(fg="#00CC44", bg="#183622")
        elif hazards:
            self.advisory_var.set(f"CAUTION: {hazards[0]} detected ahead! Reduce speed and prepare to react.")
            self.advisory_lbl.config(fg="#FFCC00", bg="#363018")
        elif restrictions:
            self.advisory_var.set(f"PROHIBITION: {list(restrictions)[0]}. Strict compliance required.")
            self.advisory_lbl.config(fg="#FF9900", bg="#362218")
        else:
            self.advisory_var.set("Normal driving conditions. Obey prevailing traffic rules.")
            self.advisory_lbl.config(fg="#00CC44", bg="#232731")

        self.restrictions_var.set(", ".join(restrictions) if restrictions else "None")
        self.hazards_var.set(", ".join(hazards) if hazards else "None")

    def _review_selected(self, new_status: OperatorStatus) -> None:
        selected = self.tree.selection()
        if not selected:
            messagebox.showinfo("Select Event", "Please select an event row in the table first.")
            return

        for item in selected:
            vals = list(self.tree.item(item, "values"))
            vals[8] = str(new_status)
            self.tree.item(item, values=vals)

            # Update event in engine
            evt_frame = int(vals[0])
            evt_track = int(vals[1])
            for e in self.engine.event_log:
                if e.frame_index == evt_frame and e.track_id == evt_track:
                    e.operator_status = new_status

    def _export_csv(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if path:
            self.engine.export_to_csv(path)
            messagebox.showinfo("Export Successful", f"Saved {len(self.engine.event_log)} events to {path}")

    def _export_json(self) -> None:
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if path:
            self.engine.export_to_json(path)
            messagebox.showinfo("Export Successful", f"Saved {len(self.engine.event_log)} events to {path}")

    def run(self) -> None:
        self.root.mainloop()


def main():
    app = RoadSafetyMonitorApp()
    app.run()


if __name__ == "__main__":
    main()
