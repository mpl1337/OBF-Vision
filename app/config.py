import json
import logging
import os
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

CONFIG_PATH = Path("./obf_config.json")

GUI_SCHEMA: dict[str, dict[str, Any]] = {
    "pipeline_enabled": {
        "type": "bool",
        "label": "Autostart / Active",
        "desc": "Master switch for the AI engine.",
        "help": "Recommendation: ON. If OFF, only the web server runs.",
    },
    "device": {
        "type": "select",
        "label": "OpenVINO Device",
        "desc": "Hardware accelerator.",
        "options": ["CPU", "GPU", "NPU"],
        "help": "Recommendation: NPU (for Intel/Rockchip) or GPU.",
    },
    "device_yolo": {
        "type": "select",
        "label": "Hardware: Object Detection (YOLO)",
        "desc": "Empfehlung: GPU (Intel Arc) für maximale FPS.",
        "options": ["CPU", "GPU", "NPU"],
    },
    "device_face_det": {
        "type": "select",
        "label": "Hardware: Face Detection",
        "desc": "Empfehlung: GPU oder CPU zur Entlastung.",
        "options": ["CPU", "GPU", "NPU"],
    },
    "device_reid": {
        "type": "select",
        "label": "Hardware: Face Identification (ReID)",
        "desc": "Empfehlung: NPU für höchste Effizienz.",
        "options": ["CPU", "GPU", "NPU"],
    },
    "perf_hint": {
        "type": "select",
        "label": "Performance Hint",
        "desc": "Optimierungsstrategie für OpenVINO.",
        "options": ["LATENCY", "THROUGHPUT"],
        "help": "LATENCY: Minimale Verzögerung pro Bild (gut für Echtzeit/Gaming).\nTHROUGHPUT: Maximale FPS (gut für NPU/GPU & Videostreams).",
    },
    "ov_pool_cap": {
        "type": "int",
        "label": "OpenVINO: Pool Cap (max requests)",
        "desc": "Max. Anzahl InferRequests pro Modell (Obergrenze für Auto-Pool).",
        "min": 1,
        "max": 16,
        "step": 1,
        "ui": "advanced",
        "help": "THROUGHPUT nutzt sonst OPTIMAL_NUMBER_OF_INFER_REQUESTS.\n"
        "Mit Cap begrenzt du RAM/Threads und vermeidest Overcommit.\n"
        "Empfehlung: 4.",
    },
    "ov_pool_min": {
        "type": "int",
        "label": "OpenVINO: Pool Min (min requests)",
        "desc": "Min. Anzahl InferRequests pro Modell (Untergrenze für Auto-Pool).",
        "min": 1,
        "max": 16,
        "step": 1,
        "ui": "advanced",
        "help": "LATENCY bleibt typischerweise bei 1.\nFür THROUGHPUT kannst du min z.B. 2–4 setzen.",
    },
    "yolo_model": {
        "type": "select",
        "label": "YOLOv11 Model (.xml)",
        "desc": "Object detection model.",
        "options": [],
        "help": "Recommendation: 'yolo11s'.",
    },
    "yolo_conf": {
        "type": "float",
        "label": "YOLO Confidence",
        "desc": "Minimum confidence for objects.",
        "min": 0.05,
        "max": 0.95,
        "step": 0.01,
        "help": "Recommendation: 0.35 - 0.50.",
    },
    "yolo_iou": {
        "type": "float",
        "label": "YOLO NMS IoU",
        "desc": "Filter against duplicate boxes.",
        "min": 0.10,
        "max": 0.90,
        "step": 0.01,
        "help": "Recommendation: 0.45.",
    },
    "yolo_max_det": {
        "type": "int",
        "label": "YOLO Max Detections",
        "desc": "Max. Anzahl Boxen pro Bild (nach NMS).",
        "min": 1,
        "max": 1000,
        "step": 1,
        "help": "Empfehlung: 100.\nNiedriger = schneller/weniger Müll, kann aber Objekte abschneiden.\nHöher = mehr Objekte, aber mehr CPU/NPU-Last.",
    },
    "tracking_enable": {
        "type": "bool",
        "label": "Tracking / Speedup",
        "desc": "Nutzt Tracker, um Objekte zwischen Frames zu verfolgen.",
        "help": "AN: Spart massiv CPU/NPU, da YOLO nicht jedes Bild läuft.\nAUS: Analysiert JEDES Bild (maximale Genauigkeit, aber langsam).",
    },
    "keyframe_interval": {
        "type": "int",
        "label": "Keyframe Intervall",
        "desc": "Wie oft läuft YOLO? (1 = Jedes Bild, 5 = Jedes 5.)",
        "min": 1,
        "max": 30,
        "step": 1,
        "help": "Empfehlung: 1 für Tests/Snapshots.\n5 für Video-Streams (spart 80% Last).",
    },
    "track_iou_thres": {
        "type": "float",
        "label": "Tracking IoU Threshold",
        "desc": "Min. Überlappung (IoU), damit eine Box als gleiches Objekt gilt.",
        "min": 0.10,
        "max": 0.90,
        "step": 0.05,
        "help": "Empfehlung: 0.50.\nNiedriger = toleranter bei schnellen Bewegungen (mehr ID-Wechsel möglich).\nHöher = strenger (kann Tracks bei schnellen Bewegungen verlieren).",
    },
    "track_max_lost": {
        "type": "int",
        "label": "Tracking Max Lost",
        "desc": "Wie viele Frames ein Track ohne Treffer überlebt.",
        "min": 1,
        "max": 60,
        "step": 1,
        "help": "Empfehlung: 10.\nHöher = stabiler bei Aussetzern/Verdeckung.\nZu hoch kann 'Ghost'-Tracks länger halten.",
    },
    "reid_votes": {
        "type": "int",
        "label": "ReID Votes",
        "desc": "Wie viele Embeddings pro Person gesammelt werden (Stabilisierung).",
        "min": 1,
        "max": 10,
        "step": 1,
        "help": "Empfehlung: 3.\nMehr = stabilere Namen (weniger Flattern), aber etwas trägere Reaktion.",
    },
    "face_enable": {
        "type": "bool",
        "label": "Face Detection active",
        "desc": "Search for faces (YOLOv8-Face).",
        "help": "Recommendation: ON.",
    },
    "face_det_model": {
        "type": "select",
        "label": "Face Detector Model",
        "desc": "Modell für Face Detection (für Enroll/DB Build).",
        "options": [],
        "help": "Empfehlung: yolov8n-face.xml. Dieses Modell findet Gesichter (Detection), nicht die Identität (ReID).",
    },
    "face_roi": {
        "type": "bool",
        "label": "Face ROI (Person→Face)",
        "desc": "Only search inside person boxes.",
        "help": "Recommendation: ON. Saves computing power and reduces errors.",
    },
    "face_conf": {
        "type": "float",
        "label": "Face Confidence",
        "desc": "Minimum confidence for faces.",
        "min": 0.05,
        "max": 0.95,
        "step": 0.01,
        "help": "Recommendation: 0.55 (for YOLOv8-Face).",
    },
    "face_min_conf": {
        "type": "float",
        "label": "Enroll: Face Confidence",
        "desc": "Min. Confidence für Face Detection beim DB-Build (Enrollment).",
        "min": 0.05,
        "max": 0.95,
        "step": 0.01,
        "help": "Wenn leer/0: nutzt face_conf. Empfehlung: 0.30 - 0.45 (Enrollment ist toleranter als Runtime).",
        "ui": "advanced",
    },
    "face_min_px": {
        "type": "int",
        "label": "Face min size (px)",
        "desc": "Ignore small faces.",
        "min": 10,
        "max": 256,
        "step": 1,
        "help": "Recommendation: 50.",
    },
    "face_quality_enable": {
        "type": "bool",
        "label": "Filter Bad Pose",
        "desc": "Skip faces looking sideways/down.",
        "help": "Analyzes face geometry. ON=Clean DB, but fewer detections.",
    },
    "face_quality_thres": {
        "type": "float",
        "label": "Pose Score Threshold",
        "desc": "0.0 (bad) to 1.0 (perfect frontal).",
        "min": 0.1,
        "max": 0.9,
        "step": 0.05,
        "help": "Recommendation: 0.6. Filter strong profiles.",
    },
    "rec_enable": {
        "type": "bool",
        "label": "Face ReID active",
        "desc": "Assign names (Who is this?).",
        "help": "Recommendation: ON.",
    },
    "rec_align": {
        "type": "bool",
        "label": "Global Face Alignment",
        "desc": "Rotate/Align faces using landmarks.",
        "help": "ON: High accuracy, saved images are 112x112. OFF: Just crops.",
    },
    "rec_model": {
        "type": "select",
        "label": "ReID Model (.xml)",
        "desc": "Face embedding model.",
        "options": [],
        "help": "Recommendation: w600k_mbf.xml",
    },
    "rec_db": {
        "type": "select",
        "label": "Face DB (.faiss/.json)",
        "desc": "Database file to load.",
        "options": [],
        "help": "Typically: faces_db_clean.faiss",
    },
    "rec_thres": {
        "type": "float",
        "label": "ReID Threshold",
        "desc": "Higher = stricter match (less false positives).",
        "min": 0.10,
        "max": 0.95,
        "step": 0.01,
        "help": "Recommendation: 0.55",
    },
    "rec_preprocess": {
        "type": "select",
        "label": "ReID Preprocess",
        "desc": "Input normalization strategy.",
        "options": ["auto", "arcface", "0_1", "raw"],
        "help": "Recommendation: auto.\nRetail-0095: raw.\nArcFace models: arcface/auto.",
    },
    "unknown_enable": {
        "type": "bool",
        "label": "Save Unknown Faces",
        "desc": "Store unknown face crops for labeling later.",
        "help": "Recommendation: ON.\nHelps building the DB.",
    },
    "unknown_dir": {
        "type": "text",
        "label": "Unknown Dir",
        "desc": "Folder for unknown face crops.",
        "help": "Default: ./unknown_faces",
    },
    "enroll_root": {
        "type": "text",
        "label": "Enroll Root",
        "desc": "Folder with person subfolders (images).",
        "help": "Default: ./enroll",
    },
    "enroll_out_db": {
        "type": "text",
        "label": "Enroll Output DB",
        "desc": "DB file written by build job.",
        "help": "Default: ./faces_db_<tag>.faiss",
    },
    "enroll_prune_out_db": {
        "type": "text",
        "label": "Prune Output DB",
        "desc": "DB file written by prune job.",
        "help": "Default: ./faces_db_<tag>_clean.faiss",
    },
    "enroll_prune_enable": {
        "type": "bool",
        "label": "Auto Prune after Build",
        "desc": "Runs prune after build.",
        "help": "Recommendation: ON.",
    },
    "enroll_prune_min_sim": {
        "type": "float",
        "label": "Enroll: Min Similarity",
        "desc": "Minimum similarity to keep a vector.",
        "min": 0.10,
        "max": 0.80,
        "step": 0.01,
        "help": "Empfehlung: 0.30.\nHöher = strenger (sauberer, aber weniger Bilder).\nNiedriger = toleranter (mehr Varianz, evtl. mehr Müll).",
    },
    "enroll_prune_dedup_sim": {
        "type": "float",
        "label": "Enroll: Dedup Similarity",
        "desc": "Ab dieser Ähnlichkeit gelten Bilder als Duplikat.",
        "min": 0.90,
        "max": 0.999,
        "step": 0.001,
        "help": "Empfehlung: 0.985.\nSehr hoch lassen (0.98+), sonst entfernst du zu viele legitime Varianten.",
    },
    "enroll_prune_keep_top": {
        "type": "int",
        "label": "Enroll: Keep Top",
        "desc": "Max. Anzahl Bilder pro Person nach dem Prune.",
        "min": 5,
        "max": 200,
        "step": 1,
        "help": "Empfehlung: 20.\nBegrenzt die DB-Größe pro Person und entfernt redundante Bilder.",
    },
    "enroll_blur_skip_enable": {
        "type": "bool",
        "label": "Enroll: Blur Filter",
        "desc": "Ignoriert verwackelte/unscharfe Bilder beim Einlernen.",
        "help": "Empfehlung: AN.\nUnschärfe erzeugt schlechte Embeddings → False IDs.",
    },
    "enroll_blur_var_thres": {
        "type": "float",
        "label": "Enroll: Sharpness Threshold",
        "desc": "Schärfe-Schwelle für Enrollment (höher=strenger).",
        "min": 10.0,
        "max": 200.0,
        "step": 1.0,
        "help": "Empfehlung: 60.\n4K/sauberes Bild: eher 80-120.\nLow-Light/kleine Faces: eher 40-60.",
    },
    "enroll_dedup_enable": {
        "type": "bool",
        "label": "Enroll: Deduplicate",
        "desc": "Entfernt Dubletten bereits vor dem Schreiben der DB.",
        "help": "Empfehlung: AN.\nVerhindert, dass nahezu identische Serienbilder die DB aufblasen.",
    },
    "enroll_dedup_sim": {
        "type": "float",
        "label": "Enroll: Dedup Similarity (Pre)",
        "desc": "Duplikat-Schwelle für Vorfilterung.",
        "min": 0.90,
        "max": 0.999,
        "step": 0.001,
        "help": "Empfehlung: 0.995.\nSehr hoch lassen, damit nur echte Dubletten rausfliegen.",
    },
    "enroll_keep_quantile": {
        "type": "float",
        "label": "Enroll: Outlier Quantile",
        "desc": "Entfernt die schlechtesten Embeddings pro Person (niedrigste Ähnlichkeit zum Centroid).",
        "min": 0.0,
        "max": 0.45,
        "step": 0.05,
        "help": "Empfehlung: 0.25.\n0.25 = schlechteste 25% werden vor dem Speichern entfernt.",
    },
    "enroll_prototypes_k": {
        "type": "int",
        "label": "Enroll: Prototypes per Person",
        "desc": "Behält bis zu K diverse Prototyp-Embeddings pro Person (Pose/Beleuchtung).",
        "min": 1,
        "max": 20,
        "step": 1,
        "help": "Empfehlung: 5.\nErhöhen, wenn du viele unterschiedliche Bilder pro Person hast.",
    },
    "enroll_case_insensitive_folders": {
        "type": "bool",
        "label": "Enroll: Ignore Case",
        "desc": "Ordnernamen ohne Groß/Kleinschreibung behandeln.",
        "help": "Empfehlung: AN.\nVerhindert doppelte Personenordner wie 'Sarah' und 'sarah'.",
    },
    "log_rotate_enable": {
        "type": "bool",
        "label": "Log to File (Rotation)",
        "desc": "Save logs to obf_app.log",
        "help": "Writes logs to project folder. Auto-rotates when full.",
    },
    "log_rotate_mb": {
        "type": "int",
        "label": "Log File Size (MB)",
        "desc": "Max size before rotation.",
        "min": 1,
        "max": 50,
        "step": 1,
        "help": "Keep small (e.g., 5 MB) to save space.",
    },
    "host": {
        "type": "text",
        "label": "Server Host (Bind)",
        "desc": "IP/Hostname, an den der Webserver bindet.",
        "help": "0.0.0.0 = im LAN erreichbar.\n127.0.0.1 = nur lokal.\nÄnderung erfordert Neustart (Save&Apply).",
    },
    "port": {
        "type": "int",
        "label": "Server Port",
        "desc": "Port des Webservers.",
        "help": "Default: 32168.\nÄnderung erfordert Neustart (Save&Apply).",
    },
    "loglevel": {
        "type": "select",
        "label": "Log Level",
        "desc": "Logging-Detailgrad (Console + optional Logfile).",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "help": "Empfehlung: INFO.\nDEBUG nur zum Troubleshooting.\nÄnderung erfordert Neustart (Save&Apply).",
    },
    "rec_blur_enable": {
        "type": "bool",
        "label": "Filter Blurry Faces",
        "desc": "Skip ReID for blurry images.",
        "help": "Prevents false detections on motion blur.",
    },
    "rec_blur_thres": {
        "type": "float",
        "label": "Blur Threshold",
        "desc": "Higher = Stricter.",
        "min": 10.0,
        "max": 200.0,
        "step": 5.0,
        "help": "Rec: 60.0. Faces below this score are ignored.",
    },
}


_ADV_KEYS = {
    "tracking_enable",
    "keyframe_interval",
    "track_iou_thres",
    "track_max_lost",
    "reid_votes",
    "rec_preprocess",
    "rec_blur_enable",
    "rec_blur_thres",
    "face_quality_enable",
    "face_quality_thres",
    "unknown_dir",
    "rec_db",
    "yolo_iou",
    "yolo_max_det",
    "perf_hint",
    "host",
    "port",
    "loglevel",
    "log_rotate_enable",
    "log_rotate_mb",
}


_ADV_PREFIXES = (
    "enroll_",
    "track_",
)

for k, meta in GUI_SCHEMA.items():
    meta.setdefault("ui", "basic")

    if k in _ADV_KEYS or any(k.startswith(p) for p in _ADV_PREFIXES):
        meta["ui"] = "advanced"


COCO80 = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)
    device_yolo: str = "CPU"
    device_face_det: str = "CPU"
    device_reid: str = "CPU"
    perf_hint: str = "LATENCY"
    yolo_model: str = "./models/yolo11s.xml"
    yolo_conf: float = Field(default=0.35, ge=0.0, le=1.0)
    yolo_iou: float = Field(default=0.50, ge=0.0, le=1.0)
    yolo_max_det: int = Field(default=100, ge=1, le=1000)
    face_enable: bool = False
    face_roi: bool = True
    face_conf: float = Field(default=0.55, ge=0.0, le=1.0)
    face_min_conf: float = Field(default=0.30, ge=0.0, le=1.0)
    face_min_px: int = Field(default=50, ge=0, le=4096)
    face_quality_enable: bool = True
    face_quality_thres: float = Field(default=0.60, ge=0.0, le=1.0)
    face_det_model: str | None = None
    rec_enable: bool = True
    rec_align: bool = True
    rec_model: str | None = "./models/w600k_mbf.xml"
    rec_db: str | None = None
    rec_thres: float = Field(default=0.55, ge=0.0, le=1.0)
    rec_preprocess: str = "auto"
    rec_blur_enable: bool = True
    rec_blur_thres: float = Field(default=60.0, ge=0.0, le=10000.0)
    unknown_enable: bool = True
    unknown_dir: str | None = "./unknown_faces"
    tracking_enable: bool = True
    keyframe_interval: int = Field(default=5, ge=1, le=9999)
    track_iou_thres: float = Field(default=0.5, ge=0.0, le=1.0)
    track_max_lost: int = Field(default=10, ge=0, le=9999)
    reid_votes: int = Field(default=3, ge=1, le=50)

    @field_validator(
        "device_yolo",
        "device_face_det",
        "device_reid",
        "perf_hint",
        "rec_preprocess",
        mode="before",
    )
    @classmethod
    def _norm_str(cls, v):
        if v is None:
            return v
        s = str(v).strip()
        return s

    @field_validator("perf_hint", mode="after")
    @classmethod
    def _norm_hint(cls, v: str):
        if not v:
            return "LATENCY"
        u = str(v).strip().upper()
        return "THROUGHPUT" if u == "THROUGHPUT" else "LATENCY"


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="allow", validate_assignment=True)
    host: str = "0.0.0.0"
    port: int = Field(default=32168, ge=1, le=65535)
    loglevel: str = "INFO"
    pipeline_enabled: bool = True
    device: str = "NPU"
    device_yolo: str = ""
    device_face_det: str = ""
    device_reid: str = ""
    perf_hint: str = "LATENCY"
    ov_pool_cap: int = Field(default=4, ge=1, le=256)
    ov_pool_min: int = Field(default=1, ge=1, le=256)
    yolo_model: str = "./models/yolo11s.xml"
    yolo_conf: float = Field(default=0.35, ge=0.0, le=1.0)
    yolo_iou: float = Field(default=0.50, ge=0.0, le=1.0)
    yolo_max_det: int = Field(default=100, ge=1, le=1000)
    face_enable: bool = False
    face_roi: bool = True
    face_conf: float = Field(default=0.55, ge=0.0, le=1.0)
    face_min_conf: float = Field(default=0.30, ge=0.0, le=1.0)
    face_min_px: int = Field(default=50, ge=0, le=4096)
    face_quality_enable: bool = True
    face_quality_thres: float = Field(default=0.60, ge=0.0, le=1.0)
    face_det_model: str | None = None
    rec_enable: bool = True
    rec_align: bool = True
    rec_model: str | None = "./models/w600k_mbf.xml"
    rec_db: str | None = None
    rec_thres: float = Field(default=0.55, ge=0.0, le=1.0)
    rec_preprocess: str = "auto"
    rec_blur_enable: bool = True
    rec_blur_thres: float = Field(default=60.0, ge=0.0, le=10000.0)
    unknown_enable: bool = True
    unknown_dir: str = "./unknown_faces"
    log_rotate_enable: bool = True
    log_rotate_mb: int = Field(default=5, ge=1, le=500)
    enroll_root: str = "./enroll"
    enroll_out_db: str = ""
    enroll_prune_out_db: str = ""
    enroll_prune_enable: bool = True
    enroll_prune_min_sim: float = Field(default=0.30, ge=0.0, le=1.0)
    enroll_prune_dedup_sim: float = Field(default=0.985, ge=0.0, le=1.0)
    enroll_prune_keep_top: int = Field(default=20, ge=1, le=500)
    enroll_blur_skip_enable: bool = True
    enroll_blur_var_thres: float = Field(default=60.0, ge=0.0, le=10000.0)
    enroll_dedup_enable: bool = True
    enroll_dedup_sim: float = Field(default=0.995, ge=0.0, le=1.0)
    enroll_keep_quantile: float = Field(default=0.25, ge=0.0, le=0.9)
    enroll_prototypes_k: int = Field(default=5, ge=1, le=100)
    enroll_case_insensitive_folders: bool = True
    tracking_enable: bool = True
    keyframe_interval: int = Field(default=5, ge=1, le=9999)
    track_iou_thres: float = Field(default=0.5, ge=0.0, le=1.0)
    track_max_lost: int = Field(default=10, ge=0, le=9999)
    reid_votes: int = Field(default=3, ge=1, le=50)

    @field_validator(
        "host",
        "loglevel",
        "device",
        "device_yolo",
        "device_face_det",
        "device_reid",
        "perf_hint",
        "rec_preprocess",
        mode="before",
    )
    @classmethod
    def _strip_str(cls, v):
        if v is None:
            return v
        return str(v).strip()

    @model_validator(mode="after")
    def _fill_device_fallbacks(self):
        dev = (self.device or "CPU").strip()

        if not self.device_yolo:
            object.__setattr__(self, "device_yolo", dev)
        if not self.device_face_det:
            object.__setattr__(self, "device_face_det", dev)
        if not self.device_reid:
            object.__setattr__(self, "device_reid", dev)

        hint = (self.perf_hint or "LATENCY").strip().upper()
        object.__setattr__(
            self, "perf_hint", "THROUGHPUT" if hint == "THROUGHPUT" else "LATENCY"
        )
        return self

    def as_dict(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=False)


def _pick_existing(default: str, alternatives: list[str]) -> str:
    if Path(default).exists():
        return default
    for a in alternatives:
        if Path(a).exists():
            return a
    return default


def default_config_dict() -> dict[str, Any]:
    yolo_model = _pick_existing(
        "./models/yolo11s.xml",
        ["./models/yolov8s.xml", "./models/yolo11n.xml", "./models/yolov8n.xml"],
    )
    rec_db = _pick_existing(
        "./faces_db_clean.faiss",
        [
            "./faces_db.faiss",
            "./faces_db_clean.json",
            "./faces_db.json",
        ],
    )

    return {
        "host": "0.0.0.0",
        "port": 32168,
        "loglevel": "INFO",
        "pipeline_enabled": False,
        "device": "NPU",
        "device_yolo": "NPU",
        "device_face_det": "NPU",
        "device_reid": "NPU",
        "perf_hint": "LATENCY",
        "ov_pool_cap": 32,
        "ov_pool_min": 1,
        "yolo_model": yolo_model,
        "yolo_conf": 0.35,
        "yolo_iou": 0.50,
        "yolo_max_det": 100,
        "face_enable": True,
        "face_det_model": "./models/yolov8n-face.xml",
        "face_roi": True,
        "face_conf": 0.55,
        "face_min_conf": 0.30,
        "face_min_px": 50,
        "face_quality_enable": True,
        "face_quality_thres": 0.6,
        "rec_enable": True,
        "rec_align": True,
        "rec_model": "./models/w600k_mbf.xml",
        "rec_db": rec_db,
        "rec_thres": 0.55,
        "rec_preprocess": "auto",
        "rec_blur_enable": True,
        "rec_blur_thres": 60.0,
        "log_rotate_enable": True,
        "log_rotate_mb": 5,
        "unknown_enable": True,
        "unknown_dir": "./unknown_faces",
        "enroll_root": "./enroll",
        "enroll_out_db": "./faces_db.faiss",
        "enroll_prune_out_db": "./faces_db_clean.faiss",
        "enroll_prune_enable": True,
        "enroll_prune_min_sim": 0.30,
        "enroll_prune_dedup_sim": 0.985,
        "enroll_prune_keep_top": 20,
        "enroll_blur_skip_enable": True,
        "enroll_blur_var_thres": 60.0,
        "enroll_dedup_enable": True,
        "enroll_dedup_sim": 0.995,
        "enroll_keep_quantile": 0.25,
        "enroll_prototypes_k": 5,
        "enroll_case_insensitive_folders": True,
        "tracking_enable": True,
        "keyframe_interval": 5,
        "track_iou_thres": 0.5,
        "track_max_lost": 10,
        "reid_votes": 3,
    }


LOG = logging.getLogger("obf.config")


def _validate_and_normalize(cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        model = AppConfig.model_validate(cfg)
        return model.as_dict()
    except ValidationError as e:
        LOG.error("Config validation failed. Falling back to defaults. Errors: %s", e)
        d = default_config_dict()
        for k, v in (cfg or {}).items():
            if k not in d:
                d[k] = v
        try:
            return AppConfig.model_validate(d).as_dict()
        except Exception:
            return d


def load_or_create_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        cfg = _validate_and_normalize(default_config_dict())
        save_config(cfg)
        return cfg

    try:
        raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("config root must be an object/dict")
    except Exception as e:
        LOG.warning("Could not read config (%s). Recreating defaults.", e)
        raw = {}

    merged = default_config_dict()
    merged.update(raw)
    merged = _validate_and_normalize(merged)
    return merged


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(str(tmp), str(path))


def save_config(cfg: dict[str, Any] | AppConfig) -> None:
    data = cfg.as_dict() if isinstance(cfg, AppConfig) else dict(cfg or {})
    data = _validate_and_normalize(data)
    _atomic_write_text(CONFIG_PATH, json.dumps(data, indent=2, ensure_ascii=False))
