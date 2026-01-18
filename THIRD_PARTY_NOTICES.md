# Third-Party Notices (OBF Vision)

This repository ("OBF Vision") is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
This file lists third-party software and externally downloaded assets that OBF Vision can use.

Important:
- Third-party components remain under their respective licenses.
- This repository is intended to contain source code only. Model weights, biometric data, and runtime artifacts SHOULD NOT be committed.
- If you redistribute a packaged/binary distribution that bundles third-party components, you may need to include additional license texts and notices (see “Bundled binaries” below).

---

## A) Direct Python dependencies (from `requirements.txt`)

The project depends on the following Python packages (installed via pip). They are NOT vendored into this repository.

1) faiss-cpu
   - Purpose: Vector similarity search (FaceDB index).
   - Upstream: Faiss (Meta).
   - License: MIT (Faiss). Packaging metadata may reference additional permissive licenses.

2) fastapi
   - Purpose: Web API framework.
   - License: MIT.

3) numpy
   - Purpose: Numerical computing / arrays.
   - License: BSD-3-Clause (NumPy) + additional permissive licenses for bundled components.

4) opencv-python
   - Purpose: Image processing (cv2).
   - License (package build scripts): MIT.
   - OpenCV library itself: Apache-2.0 (for OpenCV >= 4.5.0).

5) openvino
   - Purpose: Inference runtime / model conversion APIs.
   - License: Apache-2.0.

6) pillow
   - Purpose: Image I/O fallback.
   - License: MIT-CMU (Pillow / PIL fork).

7) python-multipart
   - Purpose: Multipart/form-data parsing (uploads).
   - License: Apache-2.0.

8) ultralytics
   - Purpose: YOLO export/conversion to OpenVINO (used during model install jobs).
   - License: AGPL-3.0 (Ultralytics provides enterprise/commercial licensing options separately).

9) uvicorn
   - Purpose: ASGI server.
   - License: BSD-3-Clause.

10) websockets
   - Purpose: WebSocket support.
   - License: BSD-3-Clause.

---

## B) Additional runtime dependencies (imported, not pinned in `requirements.txt`)

These are imported by the codebase and typically arrive as transitive dependencies or must be installed explicitly.

- pydantic
  - Purpose: Configuration schema/validation.
  - License: MIT.

- torch (PyTorch) (OPTIONAL / required for some conversion flows)
  - Purpose: Used indirectly by Ultralytics export, and referenced in ONNX conversion workaround.
  - License: BSD-3-Clause (PyTorch).

- Open Model Zoo tools (OPTIONAL; required for `omz_downloader` / `omz_converter`)
  - Purpose: Download/convert certain Intel Open Model Zoo models (e.g., `face-reidentification-retail-0095`).
  - Typical install: via `openvino-dev` / OMZ tools.
  - License: Apache-2.0 (Open Model Zoo).

---

## C) Externally downloaded model files (NOT included in this repo)

OBF Vision can download model weights from third-party sources at runtime (Model Manager). These files are not part of this repository and must not be committed.

### C1) Ultralytics pretrained weights (.pt) downloaded from GitHub Releases
Examples configured in `MODEL_CATALOG`:
- yolo11n / yolo11s / yolo11m (.pt) from Ultralytics assets releases
- yolov8n / yolov8s (.pt) from Ultralytics assets releases

Notes:
- These are provided by Ultralytics as part of their model distribution ecosystem.
- License and usage terms depend on Ultralytics’ licensing (AGPL-3.0 vs enterprise/commercial).
- Do not redistribute these weights unless you are sure you have the rights to do so.

### C2) `yolov8n-face` face detector (Google Drive link)
Configured source:
- Google Drive direct download URL (ID: `1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb`)

Notes:
- The license/rights of this specific file are not determined by this repository.
- Treat as “downloaded third-party asset” and do not redistribute unless you have explicit permission.

### C3) InsightFace-derived ONNX ReID model (`w600k_mbf.onnx`) from Hugging Face
Configured source:
- Hugging Face: `deepghs/insightface` (`buffalo_s/w600k_mbf.onnx`)
- Hugging Face lists a custom “model-distribution-disclaimer-license”.

Important:
- InsightFace upstream notes that while the *code* is MIT, *training data and models trained with those data* may be restricted (e.g., non-commercial research only) depending on the specific model/data provenance.
- You are responsible for reviewing and complying with the model’s license and any upstream restrictions before using it, especially for commercial use.

Recommendation:
- Keep this model out of the repository.
- Consider switching to a model with a clear permissive license for your intended use.

### C4) Intel Open Model Zoo models
Example configured source:
- `face-reidentification-retail-0095` via `omz_downloader` / `omz_converter`

Notes:
- Open Model Zoo is licensed under Apache-2.0.
- Individual model directories can include additional notices; check OMZ model documentation if needed.
- Do not commit downloaded models; download on demand.

---

## D) Bundled binaries and platform-specific notes

This repository does not ship binary wheels, but users install dependencies via pip.

If you redistribute a packaged application that bundles dependencies (e.g., frozen executable / Docker image with wheels),
you may need to provide additional license texts/notices for bundled components.

Examples:
- `opencv-python` wheels may bundle third-party components (e.g., FFmpeg and, on some platforms, Qt) with their own licenses.
- Always review the specific wheels you distribute and their “3rd party licenses” files (if present).

---

## E) Trademarks

Intel, OpenVINO, and other marks are trademarks of their respective owners.
This project is not affiliated with or endorsed by the upstream vendors unless explicitly stated.

---

## F) Where to find license texts

- Python package licenses: see the package’s PyPI page and upstream repository `LICENSE` file.
- Ultralytics licensing: see Ultralytics official license information and repository license.
- OpenVINO / Open Model Zoo: see Intel’s repositories and license files.
- InsightFace: see upstream license policy and model distribution terms.


THIS FILE IS GENERATED BY AI