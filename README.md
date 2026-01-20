# OBF Vision


## Installation (Windows)

### 1) Requirements
- **Python 3.11.9 (x64) (newer not tested)**
  -> Enable "Add python.exe to PATH"
- **Intel NPU Driver (Only on CPUs with NPU (Intel® Core™ Ultra))**
- **Microsoft Visual C++ Redistributable (x64)**
- **Git for Windows** (needed to clone the repo)

<img width="654" height="458" alt="image" src="https://github.com/user-attachments/assets/17d772ab-4f56-4d15-bf34-25060b6e85ff" />


Downloads:
- Python 3.11.9: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe  
- Intel NPU Driver: https://www.intel.de/content/www/de/de/download/794734/intel-npu-driver-windows.html  
- VC++ Redist (x64): https://aka.ms/vs/17/release/vc_redist.x64.exe  
- Git for Windows: https://git-scm.com/downloads/win

---

### 2) Clone the repository (into `C:\OBF`)
Open **PowerShell (Admin)**:
- **Win + X** → **A**

Then run:

```powershell
New-Item -ItemType Directory -Force -Path "C:\OBF" | Out-Null
Set-Location "C:\OBF"
git clone https://github.com/mpl1337/OBF-Vision
Set-Location "C:\OBF\OBF-Vision"
```

---

### 3) Create a virtual environment + install dependencies
In PowerShell (inside `C:\OBF\OBF-Vision`) run:

```powershell
py -3.11 -m venv obf
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\obf\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

---

### 4) Start the app
```powershell
python start.py
```

Example output:
```
Starting main.py...
INFO obf.main: Starting server on 0.0.0.0:32168
INFO uvicorn: Started server process [1900]
INFO uvicorn: Waiting for application startup.
INFO obf.main: Startup: Pipeline disabled via config.
INFO uvicorn: Application startup complete.
INFO uvicorn: Uvicorn running on http://0.0.0.0:32168 (Press CTRL+C to quit)
INFO uvicorn: <client-ip>:<client-port> - "WebSocket /ws" [accepted]
INFO uvicorn: connection open
```

> **Do not expose this to the internet** (no port forwarding) without proper protection such as VPN / reverse proxy / authentication.

---

### 5) Open the GUI
- http://localhost:32168/gui

---

### 6) Download models and prepare the Face DB
1. **Model Manager**
   - Select models: `yolo11s`, `yolov8n-face`, `w600k_mbf`
   - Start **Download Selected**
   - Wait until the job finishes and Restart Server

2. **Unknown Faces**
   - Goto Dashboard -> Faces -> "name for new person" -> "+ New Person"

3. **Face DB Tools**
   - Run **Rebuild Database**
   - Wait until the job finishes **without errors**

---

### 7) Enable the pipeline
- **Dashboard** → enable "AI Pipeline Master Switch" & Choose Devices (CPU/GPU/NPU) → **Save & restart server**

---

### 8) Adjust settings
- Change settings in the GUI as needed
- After **Save & Apply** it will performe a restart

---

### 9) Test it
- Use **Quicktest** and run a few images through it

---

## Blue Iris Settings

To reliably identify which camera is used in OBF, a small workaround is needed:  
In Blue Iris, assign a **different AI model** to each camera.  
(OBF determines the actual model internally — this is only to distinguish cameras inside BI.)

Path:
- Camera Settings → **AI**
  - Enable **Additional models**
  - Enable **Override global list**
  - Set a unique model name per camera

---

## Privacy / Data Processing (important)
This project can process face crops and embeddings and may populate folders such as `unknown_faces/` or `enroll/`.  
Make sure you comply with applicable privacy laws (e.g., GDPR) when using real camera data, and do not share personal/biometric data.

---

## License
This project is licensed under the **GNU Affero General Public License v3 (AGPL-3.0)**.

Third-party notes:
- YOLO/Ultralytics is used as a dependency; their licensing is also AGPL-3.0 (with a commercial alternative offered by the vendor).
- OpenVINO is used as a dependency (Apache-2.0).

See `LICENSE` and any additional notes in the repository.

---

# OBF Vision (Deutsch)

## Installation (Windows)

### 1) Voraussetzungen
- **Python 3.11.9 (x64) (neuere Versionen nicht getestet)**
  -> Beim Installer "Add python.exe to PATH" aktivieren
- **Intel NPU Driver (nur bei CPUs mit NPU (Intel® Core™ Ultra))**
- **Microsoft Visual C++ Redistributable (x64)**
- **Git for Windows** (zum Klonen des Repos)

Downloads:
- Python 3.11.9: https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe  
- Intel NPU Driver: https://www.intel.de/content/www/de/de/download/794734/intel-npu-driver-windows.html  
- VC++ Redist (x64): https://aka.ms/vs/17/release/vc_redist.x64.exe  
- Git for Windows: https://git-scm.com/downloads/win

---

### 2) Repository klonen (nach `C:\OBF`)
Öffne **PowerShell (Administrator)**:
- **Win + X** → **A**

Dann ausführen:

```powershell
New-Item -ItemType Directory -Force -Path "C:\OBF" | Out-Null
Set-Location "C:\OBF"
git clone https://github.com/mpl1337/OBF-Vision
Set-Location "C:\OBF\OBF-Vision"
```

---

### 3) Virtuelle Umgebung erstellen + Dependencies installieren
In PowerShell (im Ordner `C:\OBF\OBF-Vision`) ausführen:

```powershell
py -3.11 -m venv obf
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\obf\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

---

### 4) App starten
```powershell
python start.py
```

Beispiel-Output:
```
Starte main.py...
INFO obf.main: Starting server on 0.0.0.0:32168
INFO uvicorn: Started server process [1900]
INFO uvicorn: Waiting for application startup.
INFO obf.main: Startup: Pipeline disabled via config.
INFO uvicorn: Application startup complete.
INFO uvicorn: Uvicorn running on http://0.0.0.0:32168 (Press CTRL+C to quit)
INFO uvicorn: <client-ip>:<client-port> - "WebSocket /ws" [accepted]
INFO uvicorn: connection open
```

> **Bitte nicht ohne Auth/VPN/Reverse Proxy ins Internet exponieren** (kein Port-Forwarding).

---

### 5) GUI öffnen
- http://localhost:32168/gui

---

### 6) Modelle herunterladen und Face-DB vorbereiten
1. **Model Manager**
   - Modelle auswählen: `yolo11s`, `yolov8n-face`, `w600k_mbf`
   - **Auswahl hrunterladen** starten
   - Warten, bis der Job fertig ist, dann Server neustarten

2. **Unknown Faces**
   - Dashboard -> Gesichter -> "Name für neue Person" -> "+ Neue Person"

3. **Face DB Tools**
   - **Datenbank neu berechnen** starten
   - Warten, bis der Job **ohne Fehler** abgeschlossen ist

---

### 7) Pipeline aktivieren
- **Dashboard** → "AI Pipeline Master Switch" aktivieren & Hardware wählen (CPU/GPU/NPU) → **Speichern & Server neustarten**

---

### 8) Einstellungen anpassen
- In der GUI die gewünschten Einstellungen setzen
- Nach **Save & Apply** wird ggf. ein Restart ausgeführt

---

### 9) Testen
- **Quicktest** verwenden und ein paar Testbilder durchlaufen lassen

---

## Blue Iris Settings

Um die verwendete Kamera in OBF eindeutig zuzuordnen, ist aktuell ein kleiner Workaround nötig:  
In Blue Iris muss pro Kamera ein **unterschiedliches AI-Model** zugewiesen werden.  
(OBF selbst entscheidet intern, welches Modell verwendet wird – das ist nur zur Unterscheidung in BI.)

Pfad:
- Camera Settings → **AI**
  - **Additional models** aktivieren
  - **Override global list** aktivieren
  - Ein eindeutiger Modellname pro Kamera vergeben

---

## Privacy / Datenverarbeitung (wichtig)
Dieses Projekt kann Gesichtsausschnitte und Embeddings verarbeiten und Ordner wie z. B. `unknown_faces/` oder `enroll/` befüllen.  
Bitte beachte bei realen Kameradaten die geltenden Datenschutzgesetze (z. B. DSGVO) und teile keine personenbezogenen/biometrischen Daten.

---

## Lizenz
Dieses Projekt steht unter der **GNU Affero General Public License v3 (AGPL-3.0)**.

Hinweis zu Drittkomponenten:
- YOLO/Ultralytics wird als Dependency genutzt; deren Lizenzmodell ist ebenfalls AGPL-3.0 (bzw. kommerzielle Alternative durch den Hersteller).
- OpenVINO wird als Dependency genutzt (Apache-2.0).

Siehe `LICENSE` und ggf. Hinweise im Repository.

<img width="1920" height="5628" alt="screencapture-192-168-178-70-32168-gui-2026-01-20-20_04_12" src="https://github.com/user-attachments/assets/77a1bfee-e945-4b1b-926e-5a7104ebd9dc" />

<img width="1920" height="1163" alt="screencapture-192-168-178-70-32168-gui-2026-01-20-19_52_35" src="https://github.com/user-attachments/assets/56350358-5c88-4b09-93d4-55229baddbb6" />

<img width="1920" height="963" alt="screencapture-192-168-178-70-32168-gui-2026-01-20-20_00_45" src="https://github.com/user-attachments/assets/68378b4c-aafb-495f-a96c-050007128e48" />
