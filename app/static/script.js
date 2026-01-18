

let CFG = null;
let ORIG_CFG = null;
let isStreaming = false;
let SCHEMA = null;
let MODEL_CATALOG = [];
let CURRENT_UNKNOWNS = [];
let FACE_DB_DIRTY = (localStorage.getItem("obf_db_dirty") === "1");
let ADV_PARAMS_OPEN = (localStorage.getItem("obf_adv_params") === "1");
let CUR_LANG = localStorage.getItem("obf_lang") || "de";
let _PORT_CHECK_TMR = null;
let OBF_TOOLTIP_EL = null;



const TRANS = {
  de: {
    status_label: "System Status",
    uptime_label: "Laufzeit",
    basic_cfg_label: "Basic Config",
    adv_cfg_label: "Advanced Config",
    adv_cfg_desc: "Selten nötig. Falsche Werte können Accuracy/Performance verschlechtern.",
    params_basic_label: "Basic",
    params_advanced_label: "Advanced config",
    unknown_label: "Gesichter",
    btn_adv_show: "Anzeigen",
    btn_adv_hide: "Verbergen",
    pipeline_hint: "Dieser Schalter aktiviert/deaktiviert nur die KI-Analyse. Der Webserver und die API bleiben erreichbar.",
    btn_start: "Analyse aktivieren",
    btn_stop: "Analyse pausieren",
    btn_reload: "GUI neu laden",
    btn_restart: "Server neustarten",
    req_label: "API-Aufrufe:",
    last_labels: "Letzte Erkennung:",
    models_label: "Installierte KI-Modelle",
    models_desc: "Übersicht der geladenen neuronalen Netze und der aktiven Hardware-Beschleunigung.",
    btn_refresh: "Liste aktualisieren",
    btn_dl_selected: "Auswahl herunterladen",
    params_label: "Konfiguration",
    params_desc: "Feinjustierung der Erkennungslogik. Fahre über das [?] für detaillierte Erklärungen.",
    btn_save_apply: "Speichern & Server neustarten",
    facedb_label: "Gesichts-Datenbank",
    facedb_desc: "Verwaltung der biometrischen Daten (Embeddings).",
    btn_build: "Datenbank neu berechnen",
    btn_clear_log: "Logdatei leeren",
    unknown_desc: "Training: Ziehe diese Bilder auf eine Person, um deren Erkennung zu verbessern.",
    btn_del_all: "Alle verwerfen",
    btn_add_person: "+ Neue Person",
    unknown_hint: "Wichtig: Nach Zuweisungen muss die Datenbank neu berechnet werden ('Datenbank neu berechnen').",
    manager_label: "Modell Manager",
    manager_title: "Modell Bibliothek",
    manager_intro: "Modelle von offiziellen Quellen laden und in das OpenVINO-Format (FP16) konvertieren.",
    btn_refresh_log: "Log aktualisieren",
    btn_refresh_models: "Dateisystem prüfen",
    test_label: "Diagnose & Test",
    test_desc: "Simuliere Kamera-Events mit Bildern, Videos oder Live-Streams.",
    btn_run_vis: "Bild analysieren",
    btn_upload_play: "Video hochladen",
    btn_start_stream: "RTSP Stream verbinden",
    btn_delete: "Löschen",
    confirm_del_model: "Möchtest du das Modell {name} und alle zugehörigen Quelldateien (.xml, .bin, .onnx, .pt) unwiderruflich löschen?",
    install_log_label: "Installations-Protokoll",
    storage_info: "Speicherbelegung: {size} MB ({count} Dateien)",
    modal_hint: "Klicke das rote X, um einzelne Bilder zu löschen.",

    page_title: "OBF – Vision KI Server",
    header_title: "OBF – YOLOv8+11 (OpenVINO) + Face/ReID",
    btn_theme: "🌗 Theme",
    perf_ema_title: "EMA ms (Ø, geglättet)",
    perf_last_title: "Letzte Inferenzzeit",
    perf_fps_title: "Inferences pro Sekunde",
    loading: "Lade...",
    offline: "Offline",
    state_active: "AKTIV",
    state_paused: "PAUSIERT",
    label_reqs_short: "Reqs:",
    label_last_short: "Last:",
    live_label: "Live",
    qt_image: "Bild:",
    qt_video: "Video:",
    btn_pause: "Pause",
    btn_resume: "Fortsetzen",
    btn_stop_stream: "Stop",

    qt_rtsp: "RTSP:",
    qt_min_conf: "Min Conf:",
    qt_rtsp_placeholder: "rtsp://...",
    new_person_placeholder: "Name der Person...",
    unknown_filter_all: "Alle Quellen",

    pm_person: "Person",
    uc_sim_thres: "sim_thres",
    uc_min_size: "min_size",
    uc_limit: "limit",

    models_empty: "Leer.",
    models_table_model: "Modell",
    models_table_info: "Info",
    models_table_size: "Größe",
    models_table_status: "Status",
    models_table_action: "Aktion",
    models_status_ready: "BEREIT",
    models_status_missing: "FEHLT",
    models_status_active: "AKTIV",
    models_in_use: "In Verwendung",

    runtime_offline: "Runtime: Offline",
    runtime_error: "Runtime: Fehler",
    runtime_pools_hdr: "OpenVINO Pools: min {min} • cap {cap}",
    runtime_pool_line: "pool: {sel} (opt {opt}) • frei {free}",

    bench_no_models: "Keine Modelle installiert (siehe Modell Manager)",
    bench_running: "Benchmark läuft. Bitte warten.",
    bench_done_toast: "Benchmark fertig!",
    bench_error_row: "Fehler: {err}",
    bench_hl_title: "Markiert das beste Gerät pro Zeile (OFF → L → T).",
    bench_hl_off: "Aus",

    toast_deleted: "Gelöscht",
    toast_error: "Fehler",
    toast_select_models: "Wähle Modelle.",
    toast_download_started: "Download gestartet...",
    toast_rtsp_missing: "Bitte RTSP URL eingeben!",

    toast_video_choose_first: "Bitte erst ein Video auswählen.",
    toast_video_started: "Video-Stream gestartet.",
    toast_video_stopped: "Video-Stream gestoppt.",
    toast_video_upload_failed_prefix: "Video Upload fehlgeschlagen: ",
    toast_video_paused: "Pausiert (Bild eingefroren).",
    toast_video_resumed: "Fortgesetzt.",
    toast_pause_failed: "Pause fehlgeschlagen.",

    restart_recommended: "Neustart empfohlen (Model geändert)",
    confirm_restart: "Server wirklich neustarten?",
    confirm_delete_person: "Person '{name}' löschen?",
    confirm_delete_all_unknown: "Alle Unbekannten löschen?",
    confirm_delete_image: "Bild löschen?",

    person_pics: "Bilder",
    person_hits: "Treffer",
    time_just_now: "gerade eben",
    time_ago_prefix: "vor",

    uc_assign_to: "Zuweisen zu...",
    uc_move_success: "Cluster erfolgreich zu {person} verschoben!",
    cluster_error_loading: "Fehler beim Laden der Cluster.",
    api_error_prefix: "API Fehler: ",


    tab_dash: "Dashboard",
    tab_models: "Modell Manager",
    tab_bench: "Benchmark",
    tab_clusters: "Cluster (Unbekannt)",


    bench_title: "Hardware Matrix Benchmark",
    bench_desc: "Vergleicht die Geschwindigkeit (Latenz) über alle verfügbaren Recheneinheiten.",
    btn_start_bench: "Vollständigen Test starten",
    bench_hint: "Klicke auf Start, um den Test zu beginnen...",
    col_model: "Modell",
    col_device: "Gerät",
    col_time: "Zeit (ms)",
    col_fps: "Durchsatz (FPS)",

    uc_title: "Unbekannte Cluster",
    uc_reload: "Neu laden",
    uc_hint: "Hinweis: Clustering nutzt gespeicherte .npy-Embeddings neben den Bildern. Neue Gesichter werden automatisch erfasst.",
    uc_assign: "Cluster zuweisen",
    uc_show_files: "Dateien zeigen",
    uc_hide_files: "Dateien verbergen",
    uc_choose_person: "Zuerst Person wählen.",
    uc_confirm_move: "Verschiebe {count} Dateien zu '{person}'?",


    params: {
      pipeline_enabled: { l: "KI-Pipeline Hauptschalter", d: "Legt fest, ob die Bildanalyse beim Serverstart automatisch aktiv ist.", h: "Was es macht: Startet/stoppt die komplette KI-Analyse (YOLO/Face/ReID).\nWichtig: Der Webserver/GUI bleibt erreichbar.\nWenn AUS: Es werden keine Erkennungen berechnet (Antworten leer/Unknown).\nTipp: Kann ohne Neustart umgeschaltet werden; sonst wirkt es nach \"Speichern & Neustart\"." },
      device: { l: "KI-Beschleuniger (Fallback Hardware)", d: "Wählt den Prozessor für die neuronalen Netze.", h: "Standard-Hardware für alle KI-Modelle, falls du unten nicht getrennt auswählst.\nNPU: sehr effizient (ideal 24/7).\nGPU: oft höchste FPS (gut für viele Streams).\nCPU: zuverlässiger Fallback, aber deutlich langsamer.\nWenn etwas nicht läuft: teste GPU/CPU als Gegenprobe." },
      perf_hint: { l: "OpenVINO Strategie", d: "Optimiert die Hardware-Nutzung.", h: "OpenVINO Optimierungsstrategie:\nLATENCY = jedes Bild so schnell wie möglich fertig (geringe Verzögerung).\nTHROUGHPUT = mehr Parallelität (höherer Durchsatz/FPS, etwas mehr \"Lag\").\nFür Live-Alarm/BlueIris: meist LATENCY.\nFür viele Streams/Benchmark: eher THROUGHPUT.\nWirkt nach Neustart." },
      yolo_model: { l: "YOLO Modell-Version", d: "Bestimmt die Balance zwischen Geschwindigkeit und Intelligenz.", h: "Welches YOLO-Netz für Objekte verwendet wird.\n'n' (nano) = sehr schnell, aber weniger Details.\n's/m' = genauer, braucht mehr Leistung.\nBei 4K/weit entfernten Personen: eher 's'/'m'.\nBei vielen Kameras: eher 'n' + Tracking/Keyframes nutzen." },
      yolo_conf: { l: "Objekt-Konfidenz (Schwelle)", d: "Mindestwahrscheinlichkeit (0.0-1.0), ab der ein Objekt gemeldet wird.", h: "Ab welcher Sicherheit (0.0-1.0) ein Objekt gemeldet wird.\nHöher = weniger Fehlalarme, aber evtl. verpasste Objekte.\nNiedriger = erkennt mehr, kann aber \"Geister\" erzeugen.\nTypisch: 0.35-0.50.\nWenn nichts erkannt wird: senken; wenn zu viel Müll: erhöhen." },
      yolo_iou: { l: "NMS Überlappung (IoU)", d: "Filtert doppelte Rahmen um dasselbe Objekt.", h: "IoU für Non-Maximum-Suppression (doppelte Boxen entfernen).\nNiedriger = aggressiveres Zusammenfassen (weniger doppelte Boxen).\nHöher = toleranter (besser bei Gruppen, aber mehr Doppelboxen möglich).\nTypisch: 0.40-0.55.\nViele Boxen pro Person -> senken. Personen verschmelzen -> erhöhen." },
      tracking_enable: { l: "Tracking & Speedup", d: "Aktiviert den Objekt-Verfolger und überspringt Frames.", h: "Tracker hält Boxen zwischen Keyframes stabil.\nAN = YOLO nur alle N Frames, dazwischen Tracking -> schneller.\nAUS = häufigere YOLO-Detektion -> genauer, aber langsamer.\nBei springenden Boxen: AN lassen und IOU/MaxLost anpassen." },
      keyframe_interval: { l: "Keyframe Intervall", d: "Jedes X-te Bild wird voll analysiert.", h: "Wie oft YOLO neu gerechnet wird (alle N Frames).\nKleiner = genauer/aktueller, aber mehr Last.\nGrößer = mehr FPS, kann bei schnellen Bewegungen hinterherhinken.\nTypisch: 3-10.\nBei hoher Last: erhöhen." },
      track_iou_thres: { l: "Tracking IoU Schwelle", d: "Wie stark muss eine Box überlappen, um als gleiches Objekt zu gelten?", h: "Match-Schwelle des Trackers (Box-Überlappung).\nHöher = strenger (weniger ID-Wechsel, kann Tracks verlieren).\nNiedriger = toleranter (Tracks bleiben, können aber wechseln/verschmelzen).\nTypisch: 0.3-0.6." },
      track_max_lost: { l: "Tracking Max Lost", d: "Wie viele Frames ein Track ohne Treffer überlebt.", h: "Wie viele Frames ein Track ohne neue Detektion bestehen bleibt.\nHöher = stabiler bei kurzen Aussetzern, kann aber \"Ghosts\" halten.\nNiedriger = schnelleres Aufräumen.\nTypisch: 5-20." },
      reid_votes: { l: "ReID Stabilisierung (Votes)", d: "Sammelt mehrere Embeddings pro Person für stabilere Namen.", h: "Wie viele ReID-Vektoren gesammelt werden, bevor ein Name gesetzt wird.\nHöher = stabiler (weniger Flackern), aber Name kommt später.\nNiedriger = schneller, kann aber flackern.\nTypisch: 2-5.\nName zu spät -> senken. Name springt -> erhöhen." },
      face_enable: { l: "Gesichtserkennung aktiv", d: "Schaltet das zweite neuronale Netz zur Gesichtssuche ein.", h: "Aktiviert die komplette Gesichts-Pipeline (FaceDet + ReID).\nWenn AUS: keine Gesichter/keine Namen (nur Objekte).\nEmpfehlung: AN, wenn du Namen/Unknown willst.\nHinweis: benötigt Rechenleistung (je nach Hardware/Modell)." },
      face_det_model: { l: "Face Detector Modell", d: "Netz zur Gesichtserkennung (Detection).", h: "Welches Modell Gesichter findet (Box + ggf. Landmarks).\nMuss als OpenVINO .xml/.bin vorhanden sein.\nWenn keine Gesichter: anderes FaceDet-Modell testen.\nLandmarks verbessern Alignment und damit die Erkennung." },
      face_roi: { l: "Smart ROI (Region of Interest)", d: "Performance-Boost: Sucht Gesichter nur innerhalb erkannter Personen.", h: "Wenn AN: FaceDet läuft nur innerhalb der \"Person\"-Boxen aus YOLO (ROI).\nVorteil: deutlich schneller + weniger False-Positives.\nNachteil: Wenn YOLO die Person nicht findet, wird auch kein Gesicht gesucht.\nWenn du Gesichter ohne ganze Person im Bild hast: AUS testen." },
      face_conf: { l: "Gesichts-Konfidenz", d: "Ab wann wird ein Bereich als Gesicht akzeptiert?", h: "Schwelle für FaceDet selbst (0.0-1.0).\nHöher = nur sehr sichere Gesichter (weniger Fehlboxen).\nNiedriger = findet mehr (auch teilweise verdeckte), kann aber Müll liefern.\nTypisch: 0.45-0.65.\nFehlende Boxen -> senken. Falsche Boxen -> erhöhen." },
      face_min_px: { l: "Minimale Gesichtsgröße", d: "Ignoriert Gesichter, die zu weit entfernt (zu klein) sind.", h: "Minimale Gesichtsgröße (Pixel) im Crop.\nZu kleine/farne Gesichter liefern schlechte Vektoren -> falsche Namen.\nHöher = sauberer (nur nahe Gesichter), aber mehr Misses.\nTypisch: 30-80 (abhängig von Auflösung).\nBei 4K eher höher." },
      face_quality_enable: { l: "Qualitäts-Filter (Pose)", d: "Analysiert Geometrie (Augenabstand, Nase), um die Blickrichtung zu prüfen.", h: "Zusätzlicher Pose/Qualitäts-Filter vor ReID.\nVorteil: sauberere DB + weniger falsche IDs.\nNachteil: Profile/Seitansichten werden verworfen (Unknown).\nWenn du Profile erkennen willst: AUS oder Threshold senken." },
      face_quality_thres: { l: "Blick-Score Schwelle", d: "Bewertung: 1.0 = Perfekt frontal, 0.0 = Hinterkopf.", h: "Schwelle für Pose/Qualität (0.0-1.0): 1.0 = frontal, kleiner = Profil.\nHöher = strenger (mehr wird verworfen), niedriger = toleranter.\nTypisch: 0.35-0.60.\nZu viele Unknown bei Profilen -> senken. Falsche IDs bei schrägen Gesichtern -> erhöhen." },
      rec_enable: { l: "Identifizierung (ReID)", d: "Vergleicht das gefundene Gesicht mit der Vektor-Datenbank.", h: "Aktiviert die Namens-Erkennung (ReID).\nWenn AUS: Gesichter werden ggf. erkannt, aber alle bleiben \"Unknown\".\nEmpfehlung: AN.\nZum Debuggen kannst du AUS setzen (nur FaceDet prüfen)." },
      rec_align: { l: "Gesichtsausrichtung (Alignment)", d: "Rotiert das Gesicht rechnerisch anhand der Augenlinie.", h: "Richtet Gesichter anhand der Augen (Landmarks) aus.\nFast immer AN lassen - verbessert Trefferquote deutlich.\nNur AUS, wenn dein FaceDet keine Landmarks liefert oder zum Test.\nSchlechtes Alignment = schlechte Vektoren = falsche Namen." },
      rec_model: { l: "Vergleichs-Modell (ReID)", d: "Das Netz, das das Bild in einen mathematischen Vektor (512 Zahlen) wandelt.", h: "ReID-Modell, das aus einem Gesicht einen Embedding-Vektor erzeugt.\nWenn du das Modell wechselst: Datenbank neu bauen!\nSonst passen gespeicherte Vektoren nicht -> falsche Treffer/Unknown.\nGrößere Modelle sind oft genauer, brauchen aber mehr Leistung." },
      rec_db: { l: "Vektor-Datenbank", d: "Pfad zur Datei, die die mathematischen Fingerabdrücke der Gesichter enthält.", h: "Datei mit gespeicherten Gesichts-Vektoren (FAISS + Meta-JSON).\nWird aus dem Enroll-Ordner erzeugt (\"DB bauen\").\nWenn DB fehlt/leer: du bekommst nur \"Unknown\".\nNach Modellwechsel immer neue DB erstellen." },
      rec_preprocess: { l: "Bild-Normalisierung", d: "Technische Vorverarbeitung der Farbwerte (-1..1 oder 0..255).", h: "Wie das Bild vor ReID normalisiert wird.\nEmpfehlung: \"Auto\" lassen (GUI wählt passend).\nFalsche Einstellung macht Vektoren unbrauchbar -> alles Unknown oder falsche Namen.\nNur ändern, wenn du sicher weißt, was dein Modell erwartet." },
      rec_thres: { l: "Ähnlichkeits-Limit (Threshold)", d: "Wie ähnlich müssen sich Vektoren sein (0.0 bis 1.0)?", h: "Ähnlichkeitsschwelle für einen Namens-Treffer (0.0-1.0).\nHöher = strenger (weniger False-IDs, mehr Unknown).\nNiedriger = toleranter (mehr Treffer, Risiko falscher Namen).\nTypisch: 0.50-0.65.\nFalsche Namen -> erhöhen. Nie erkannt -> senken." },
      unknown_enable: { l: "Unbekannte speichern", d: "Speichert Gesichter ohne Namenszuordnung auf der Festplatte.", h: "Speichert unbekannte Gesichter (Unknown) als Dateien.\nBasis fürs Training/Clustering: du kannst sie später Personen zuweisen.\nDeaktiviere nur aus Datenschutz- oder Speicherplatz-Gründen.\nWenn AUS: keine neuen Trainingsdaten." },
      unknown_dir: { l: "Speicherordner (Unbekannte)", d: "Pfad, wo Bilder von Fremden abgelegt werden.", h: "Ordner, in dem Unknown-Gesichter gespeichert werden.\nStandard: ./unknown_faces\nAchte auf genügend Speicher (kann schnell wachsen).\nTipp: regelmäßig aufräumen oder Clustering nutzen." },
      enroll_prune_enable: { l: "Enroll: DB Clean/Prune", d: "Räumt beim DB-Bauen Duplikate/Outlier auf.", h: "Beim DB-Build automatisch aufräumen (Ausreißer/Duplikate entfernen).\nEmpfehlung: AN (sauberere DB, weniger False-IDs).\nWenn du sehr wenige Bilder hast und alles behalten willst: AUS.\nDetails siehst du im Enroll-Log." },
      enroll_prune_min_sim: { l: "Enroll: Min Similarity", d: "Min. Ähnlichkeit, damit ein Bild behalten wird.", h: "Wie tolerant das Aufräumen gegenüber abweichenden Bildern ist.\nHöher = strenger (sauber, aber weniger Variation).\nNiedriger = toleranter (mehr Variation, aber auch mehr Müll).\nTypisch: 0.25-0.40." },
      enroll_prune_dedup_sim: { l: "Enroll: Dedup Similarity", d: "Ab dieser Ähnlichkeit gelten Bilder als Duplikat.", h: "Duplikat-Schwelle beim Prune.\nSehr hoch lassen (0.98-0.995), sonst wird zu aggressiv gelöscht.\nViele Serienbilder -> leicht senken.\nZu viel gelöscht -> erhöhen." },
      enroll_prune_keep_top: { l: "Enroll: Keep Top", d: "Max. Anzahl Bilder pro Person nach dem Prune.", h: "Maximale Anzahl Bilder/Vektoren pro Person nach dem Prune.\nBegrenzt DB-Größe und entfernt \"100 fast gleiche\" Bilder.\nTypisch: 20-50.\nWenn du viele Varianten brauchst: erhöhen." },
      enroll_blur_skip_enable: { l: "Enroll: Blur Filter", d: "Ignoriert unscharfe Bilder beim Einlernen.", h: "Beim DB-Build unscharfe Trainingsbilder überspringen.\nEmpfehlung: AN, sonst lernt die DB \"Matsch\".\nWenn du sehr wenige Bilder hast: testweise AUS.\nWirkt nur beim DB-Build, nicht live." },
      enroll_blur_var_thres: { l: "Enroll: Schärfe-Schwelle", d: "Schärfe-Grenzwert (höher=strenger).", h: "Schwelle für Unschärfe beim Enroll (größer = strenger).\nUnterhalb wird das Bild nicht in die DB aufgenommen.\nTypisch: 40-80.\nWenn zu viele Bilder rausfliegen: senken.\nWenn DB ungenau ist: erhöhen (nur scharfe Bilder lernen)." },
      enroll_dedup_enable: { l: "Enroll: Deduplicate", d: "Entfernt Duplikate schon vor dem Schreiben der DB.", h: "Entfernt sehr ähnliche Bilder pro Person (Dedup) vor dem Speichern.\nEmpfehlung: AN (mehr Vielfalt, weniger Ballast).\nAUS nur, wenn du bewusst viele ähnliche Bilder behalten willst." },
      enroll_dedup_sim: { l: "Enroll: Dedup Schwelle", d: "Duplikat-Schwelle für Vorfilterung.", h: "Ähnlichkeitsschwelle für Dedup.\nHöher = nur nahezu identische Bilder werden entfernt.\nTypisch: 0.99-0.999.\nZu niedrig = zu aggressiv." },
      enroll_case_insensitive_folders: { l: "Enroll: Ignore Case", d: "Ordnernamen ohne Groß/Kleinschreibung behandeln.", h: "Ordnernamen ohne Groß/Klein behandeln.\nVerhindert doppelte Personen wie \"Sarah\" und \"sarah\".\nEmpfehlung: AN.\nAUS nur, wenn du bewusst getrennte Identitäten willst." },
      device_yolo: { l: "Hardware: Objekt-Erkennung (YOLO)", d: "Wählt das Gerät für YOLO.", h: "Auf welcher Hardware die Objekterkennung (Person/Auto/...) läuft.\nGPU = meist höchste FPS, NPU = effizient, CPU = Fallback.\nWenn YOLO zu langsam ist: GPU wählen oder Keyframe/Tracking anpassen.\nWenn YOLO zickt/Fehler: CPU testen." },
      device_face_det: { l: "Hardware: Face Detection", d: "Wählt das Gerät für die Gesichtssuche (Detection).", h: "Hardware für das Face-Detection Modell (Gesicht finden / Box + Landmarks).\nKann getrennt von YOLO laufen, um Geräte zu entlasten.\nWenn die NPU voll ist: FaceDet auf GPU/CPU legen.\nWenn keine Gesichter erkannt werden: CPU als Test wählen." },
      device_reid: { l: "Hardware: Face Identification (ReID)", d: "Wählt das Gerät für die Identifizierung.", h: "Hardware für die Identifizierung (Name/Unknown).\nNPU ist effizient, GPU oft schnell, CPU zuverlässig.\nWenn Namen sehr spät kommen: reid_votes senken oder ReID auf GPU.\nBei falschen IDs eher Threshold/DB/Qualität prüfen." },
      ov_pool_cap: { l: "OpenVINO: Pool Cap", d: "Maximale InferRequests pro Modell (Obergrenze).", h: "Maximale Anzahl paralleler Infer-Requests pro Modell (oberes Limit).\nMehr = mehr Parallelität/Throughput, aber mehr RAM/Threads.\nZu hoch kann NPU/GPU überlasten oder instabil machen.\nTypisch: CPU 1-2 * NPU 1-4 * GPU 4-8.\nNur ändern, wenn du genau weißt warum." },
      ov_pool_min: { l: "OpenVINO: Pool Min", d: "Minimale InferRequests pro Modell (Untergrenze).", h: "Minimale Anzahl Infer-Requests pro Modell (unteres Limit).\nLATENCY: meist 1.\nTHROUGHPUT: oft 2-4.\nHöher kann \"Stottern\" reduzieren, erhöht aber Last/RAM.\nWenn du Aussetzer siehst: testweise MIN leicht erhöhen." },
      yolo_max_det: { l: "YOLO Max Detections", d: "Max. Boxen pro Bild (nach NMS).", h: "Obergrenze wie viele Boxen pro Bild nach NMS übrig bleiben.\nNiedriger = schneller, aber bei vielen Personen/Objekten kann etwas fehlen.\nHöher = vollständiger, aber mehr Last.\nTypisch: 50-150.\nBei Performance-Problemen: senken; bei \"Crowds\": erhöhen." },
      face_min_conf: { l: "Min Conf (Quick Test)", d: "Mindest-Konfidenz im Quick-Test/Stream Overlay.", h: "Nur Anzeige/Filter in der GUI (Quick-Test/Overlay).\nHat keinen Einfluss auf die eigentliche Erkennung.\nDamit blendest du sehr unsichere Boxen aus.\nTypisch: 0.40-0.55." },
      enroll_root: { l: "Enroll Ordner", d: "Wurzelordner für Personenbilder.", h: "Trainingsordner (pro Person ein Unterordner).\nStruktur: enroll/<Name>/*.jpg\nNach Änderungen: \"DB bauen\" ausführen.\nTipp: viele Varianten (Licht/Winkel/Brille)." },
      enroll_out_db: { l: "Enroll Output DB", d: "Zielpfad der FaceDB (.faiss/.json).", h: "Zielpfad der erzeugten Datenbank (FAISS).\nWird beim DB-Build geschrieben.\nWenn du den Pfad änderst, entsteht eine neue DB-Datei.\nTipp: .faiss verwenden (Meta liegt als .json daneben)." },
      enroll_prune_out_db: { l: "Enroll Prune Output DB", d: "Alternative DB für Prune/Debug.", h: "Optionaler Zielpfad für die bereinigte DB (nach Prune).\nNützlich, wenn du Original & Clean-Version behalten willst.\nWenn leer, wird ein Standardname verwendet.\nWird beim Build/Prune geschrieben." },
      enroll_keep_quantile: { l: "Enroll: Keep Quantile", d: "Behält nur die besten X% (Qualität/Score).", h: "Qualitätsfilter: behält nur die besten X% der Bilder.\nBeispiel 0.25 = nur Top 25% (sehr streng).\nTypisch: 0.20-0.60.\nZu wenige Vektoren -> erhöhen. DB zu \"matschig\" -> senken." },
      enroll_prototypes_k: { l: "Enroll: Prototypes K", d: "Prototypen pro Person (Verdichtung).", h: "Komprimiert viele Bilder auf K Prototypen pro Person.\nMehr = robuster (verschiedene Winkel), aber größere DB.\nTypisch: 1-5.\nBei hoher Varianz (Brille/Mütze): erhöhen." },



      rec_blur_enable: { l: "Unschärfe-Filter (Laplacian)", d: "Erkennt Bewegungsunschärfe und verwirft das Bild vor der Erkennung.", h: "Schärfe-Filter vor ReID: verschwommene Gesichter werden verworfen.\nEmpfehlung: AN.\nReduziert falsche IDs durch Bewegungsunschärfe/Nacht-Schmiere.\nWenn du trotz scharfer Bilder viele Unknown bekommst: Threshold/Score prüfen." },
      rec_blur_thres: { l: "Schärfe-Score Schwelle", d: "Minimaler Kantenkontrast. Höher = Strenger.", h: "Schwelle für den Schärfe-Score (größer = strenger).\nUnterhalb wird das Gesicht als zu unscharf behandelt -> Unknown.\nTypisch: 40-80.\n4K: eher 80-120. Low-Res: eher 30-50.\nWenn zu viele Bilder verworfen werden: senken." },
      log_rotate_enable: { l: "Datei-Logging aktivieren", d: "Schreibt Ereignisse und Fehler in 'obf_app.log'.", h: "Schreibt Logs zusätzlich in eine Datei (obf_app.log) und rotiert sie.\nEmpfehlung: AN, damit du Fehler/Crashes später nachvollziehen kannst.\nWenn AUS: Logs nur im Terminal/Console." },
      log_rotate_mb: { l: "Max. Log-Größe (MB)", d: "Größe, ab der eine neue Logdatei angefangen wird.", h: "Maximale Größe der Log-Datei bevor rotiert wird.\nKlein = spart Speicher, groß = mehr Historie.\nTypisch: 5-20 MB.\nBei langen Debug-Sessions: erhöhen." },
      host: { l: "Server Host", d: "IP-Adresse (0.0.0.0 für alle Netzwerke).", h: "IP/Hostname, an den der Webserver bindet.\n0.0.0.0 = im LAN erreichbar.\n127.0.0.1 = nur lokal.\nÄnderung erfordert Neustart.\nTipp: Für Zugriff im Netzwerk 0.0.0.0 lassen." },
      port: { l: "Server Port", d: "Port für das Webinterface.", h: "TCP-Port der Weboberfläche/API.\nStandard: 32168.\nWenn belegt: freien Port wählen (z.B. 32169).\nÄnderung erfordert Neustart.\nFirewall/Reverse-Proxy ggf. anpassen." },
      loglevel: { l: "Log Level", d: "Detailgrad der Ausgaben.", h: "Logging-Detailgrad.\nINFO = normaler Betrieb.\nDEBUG = sehr viele Details (Fehleranalyse), kann Performance kosten.\nWARNING/ERROR = weniger Output.\nÄnderung erfordert Neustart." }
    }
  },
  en: {
    status_label: "System Status",
    uptime_label: "Uptime",
    basic_cfg_label: "Basic Config",
    adv_cfg_label: "Advanced Config",
    adv_cfg_desc: "Rarely needed. Wrong values can reduce accuracy/performance.",
    params_basic_label: "Basic",
    params_advanced_label: "Advanced config",
    unknown_label: "Faces",
    btn_adv_show: "Show",
    btn_adv_hide: "Hide",
    pipeline_hint: "Toggle AI analysis only. Web server and API remain accessible.",
    btn_start: "Start Analysis",
    btn_stop: "Pause Analysis",
    btn_reload: "Reload GUI",
    btn_restart: "Restart Server",
    req_label: "API Requests:",
    last_labels: "Last Detection:",
    models_label: "Installed AI Models",
    models_desc: "Overview of loaded neural networks.",
    btn_refresh: "Refresh List",
    btn_dl_selected: "Download Selected",
    params_label: "Configuration",
    params_desc: "Fine-tune detection logic. Hover over [?] for detailed explanations.",
    btn_save_apply: "Save & Restart Server",
    facedb_label: "Face Database",
    facedb_desc: "Manages biometric data (embeddings).",
    btn_build: "Rebuild Database",
    btn_clear_log: "Clear Logfile",
    unknown_desc: "Training: Drag images onto a person to improve recognition.",
    btn_del_all: "Discard All",
    btn_add_person: "+ New Person",
    unknown_hint: "Important: After assigning faces, you must click 'Rebuild Database' for changes to take effect.",
    manager_label: "Model Manager",
    manager_title: "Model Library",
    manager_intro: "Downloads models from official sources and converts them to efficient OpenVINO format (FP16).",
    btn_refresh_log: "Refresh Log",
    btn_refresh_models: "Check Filesystem",
    test_label: "Diagnostics & Test",
    test_desc: "Simulate camera events with images, videos, or live streams.",
    btn_run_vis: "Analyze Image",
    btn_upload_play: "Upload Video",
    btn_start_stream: "Connect RTSP",
    btn_delete: "Delete",
    confirm_del_model: "Do you want to permanently delete model {name} and all related source files (.xml, .bin, .onnx, .pt)?",
    install_log_label: "Installation Log",
    storage_info: "Storage usage: {size} MB ({count} files)",
    modal_hint: "Click Red X to delete single image.",

    page_title: "OBF – Vision AI Server",
    header_title: "OBF – YOLOv8+11 (OpenVINO) + Face/ReID",
    btn_theme: "🌗 Theme",
    perf_ema_title: "EMA ms (avg, smoothed)",
    perf_last_title: "Last inference time",
    perf_fps_title: "Inferences per second",
    loading: "Loading...",
    offline: "Offline",
    state_active: "ACTIVE",
    state_paused: "PAUSED",
    label_reqs_short: "Reqs:",
    label_last_short: "Last:",
    live_label: "Live",
    qt_image: "Image:",
    qt_video: "Video:",
    btn_pause: "Pause",
    btn_resume: "Resume",
    btn_stop_stream: "Stop",

    qt_rtsp: "RTSP:",
    qt_min_conf: "Min Conf:",
    qt_rtsp_placeholder: "rtsp://...",
    new_person_placeholder: "Name for new person...",
    unknown_filter_all: "All Sources",

    pm_person: "Person",
    uc_sim_thres: "sim_thres",
    uc_min_size: "min_size",
    uc_limit: "limit",

    models_empty: "Empty.",
    models_table_model: "Model",
    models_table_info: "Info",
    models_table_size: "Size",
    models_table_status: "Status",
    models_table_action: "Action",
    models_status_ready: "READY",
    models_status_missing: "MISSING",
    models_status_active: "ACTIVE",
    models_in_use: "In use",

    runtime_offline: "Runtime: Offline",
    runtime_error: "Runtime: Error",
    runtime_pools_hdr: "OpenVINO Pools: min {min} • cap {cap}",
    runtime_pool_line: "pool: {sel} (opt {opt}) • free {free}",

    bench_no_models: "No models installed (see Model Manager)",
    bench_running: "Benchmark running. Please wait.",
    bench_done_toast: "Benchmark complete!",
    bench_error_row: "Error: {err}",
    bench_hl_title: "Highlights the best device per row (OFF → L → T).",
    bench_hl_off: "Off",

    toast_deleted: "Deleted",
    toast_error: "Error",
    toast_select_models: "Select models.",
    toast_download_started: "Download started...",
    toast_rtsp_missing: "Please enter an RTSP URL!",

    toast_video_choose_first: "Please choose a video first.",
    toast_video_started: "Video stream started.",
    toast_video_stopped: "Video stream stopped.",
    toast_video_upload_failed_prefix: "Video upload failed: ",
    toast_video_paused: "Paused (frame frozen).",
    toast_video_resumed: "Resumed.",
    toast_pause_failed: "Pause failed.",

    restart_recommended: "Restart recommended (model changed)",

    confirm_restart: "Restart server now?",
    confirm_delete_person: "Delete person '{name}'?",
    confirm_delete_all_unknown: "Delete all unknowns?",
    confirm_delete_image: "Delete image?",

    person_pics: "Pics",
    person_hits: "Hits",
    time_just_now: "just now",
    time_ago_suffix: "ago",

    uc_assign_to: "Assign to...",
    uc_move_success: "Moved cluster to {person}!",
    cluster_error_loading: "Error loading clusters.",
    api_error_prefix: "API error: ",

    tab_dash: "Dashboard",
    tab_models: "Model Manager",
    tab_bench: "Benchmark",
    tab_clusters: "Clusters (Unknown)",

    bench_title: "Hardware Matrix Benchmark",
    bench_desc: "Measures real-world inference time on your selected processor.",
    btn_start_bench: "Run Full Benchmark",
    bench_hint: "Click start to begin the test...",
    col_model: "Model",
    col_device: "Device",
    col_time: "Time (ms)",
    col_fps: "Throughput (FPS)",

    uc_title: "Unknown Clusters",
    uc_reload: "Reload",
    uc_hint: "Note: Clustering uses saved .npy embeddings. New faces are captured automatically.",
    uc_assign: "Assign Cluster",
    uc_show_files: "Show Files",
    uc_hide_files: "Hide Files",
    uc_choose_person: "Choose a person first.",
    uc_confirm_move: "Move {count} files to '{person}'?",

    params: {
      pipeline_enabled: { l: "AI Pipeline Master Switch", d: "Determines if image analysis starts automatically with the server.", h: "Starts/stops the full AI analysis (YOLO/Face/ReID).\nImportant: the web server/GUI stays reachable.\nIf OFF: no detections are computed (results empty/Unknown).\nTip: can be toggled live; otherwise applies after \"Save & Restart\"." },
      device: { l: "AI Accelerator (Fallback Hardware)", d: "Selects the processor for neural networks.", h: "Default hardware for all AI models unless you set per-model devices below.\nNPU: very power-efficient (great for 24/7).\nGPU: usually highest FPS (good for many streams).\nCPU: reliable fallback but much slower.\nIf something breaks: try GPU/CPU as a sanity check." },
      perf_hint: { l: "OpenVINO Strategy", d: "Optimizes hardware usage.", h: "OpenVINO strategy:\nLATENCY = finish each frame ASAP (low delay).\nTHROUGHPUT = more parallelism (higher FPS, a bit more lag).\nFor alarms/live events: usually LATENCY.\nFor many streams/benchmarks: THROUGHPUT.\nApplies after restart." },
      yolo_model: { l: "YOLO Model Version", d: "Balances speed vs. intelligence.", h: "Which YOLO network is used for objects.\n'n' (nano) = fastest, less detail.\n's/m' = more accurate, heavier.\nFor 4K / far objects: prefer 's'/'m'.\nFor many cameras: prefer 'n' + tracking/keyframes." },
      yolo_conf: { l: "Object Confidence Threshold", d: "Min probability (0.0-1.0) to report an object.", h: "Minimum confidence (0.0-1.0) to report an object.\nHigher = fewer false alarms, but may miss objects.\nLower = more detections, but more noise.\nTypical: 0.35-0.50.\nNothing detected -> lower. Too much junk -> raise." },
      yolo_iou: { l: "NMS Overlap (IoU)", d: "Non-Maximum Suppression: Filters duplicate boxes.", h: "IoU for Non-Maximum-Suppression (remove duplicate boxes).\nLower = merges duplicates more aggressively.\nHigher = keeps boxes separate (better for crowds, but more duplicates).\nTypical: 0.40-0.55.\nMany boxes per person -> lower. People merge -> raise." },
      tracking_enable: { l: "Tracking & Speedup", d: "Enables object tracker and skips frames for speed.", h: "Tracker keeps boxes stable between keyframes.\nON = YOLO runs every N frames, tracking in between -> faster.\nOFF = more YOLO work -> more accurate but slower.\nIf boxes jump: keep ON and tune IoU/MaxLost." },
      keyframe_interval: { l: "Keyframe Interval", d: "Run YOLO every X frames.", h: "How often YOLO is run (every N frames).\nLower = more accurate, higher cost.\nHigher = more FPS, but may lag on fast motion.\nTypical: 3-10.\nIf under heavy load: increase." },
      track_iou_thres: { l: "Tracking IoU Threshold", d: "How much overlap counts as the same object.", h: "Tracker match threshold (box overlap).\nHigher = stricter (fewer ID switches, can drop tracks).\nLower = more tolerant (tracks persist, may switch/merge).\nTypical: 0.3-0.6." },
      track_max_lost: { l: "Tracking Max Lost", d: "How many frames a track survives without a match.", h: "How many frames a track survives without a fresh detection.\nHigher = more stable, but can keep ghost tracks.\nLower = cleans up faster.\nTypical: 5-20." },
      reid_votes: { l: "ReID Votes", d: "How many embeddings are aggregated for stable naming.", h: "How many embeddings are collected before committing a name.\nHigher = more stable (less flicker), but slower to show a name.\nLower = faster, but can flicker.\nTypical: 2-5.\nName too late -> lower. Name flickers -> raise." },
      face_enable: { l: "Face Detection Active", d: "Enables the second neural network for face search.", h: "Enables the full face pipeline (FaceDet + ReID).\nIf OFF: no face naming (objects only).\nRecommended ON if you want Name/Unknown.\nCosts compute depending on hardware/model." },
      face_det_model: { l: "Face Detector Model", d: "Neural network for face detection (detection only).", h: "Which face detector model is used (boxes + optional landmarks).\nMust exist as OpenVINO .xml/.bin.\nIf faces are missing: try another face model.\nLandmarks improve alignment and naming quality." },
      face_roi: { l: "Smart ROI (Region of Interest)", d: "Performance: Searches faces ONLY inside detected people.", h: "If ON: run face detection only inside YOLO person boxes (ROI).\nPros: much faster + fewer false positives.\nCons: if YOLO misses the person, no face search happens.\nIf you often see faces without full bodies: try OFF." },
      face_conf: { l: "Face Confidence", d: "Threshold to accept a spot as a face.", h: "Face detector confidence threshold (0.0-1.0).\nHigher = fewer false boxes, but more missed faces.\nLower = finds more (incl. partial faces), but more noise.\nTypical: 0.45-0.65.\nMissing boxes -> lower. Wrong boxes -> raise." },
      face_min_px: { l: "Min Face Pixel Size", d: "Ignores faces that are too far away.", h: "Minimum face size in pixels (crop width).\nVery small faces produce bad embeddings -> wrong names.\nHigher = cleaner results, but more misses.\nTypical: 30-80 depending on camera/resolution.\nFor 4K: go higher." },
      face_quality_enable: { l: "Quality/Pose Filter", d: "Analyzes geometry to check gaze direction.", h: "Extra pose/quality gate before ReID.\nPros: cleaner DB + fewer false IDs.\nCons: side profiles are discarded (Unknown).\nIf you want profile naming: turn OFF or lower threshold." },
      face_quality_thres: { l: "Gaze Score Threshold", d: "Score: 1.0 = Perfect frontal, 0.0 = Back of head.", h: "Pose/quality threshold (0.0-1.0): 1.0 frontal, lower = profile.\nHigher = stricter (more discarded), lower = more tolerant.\nTypical: 0.35-0.60.\nToo many profile Unknowns -> lower. False IDs on angled faces -> raise." },
      rec_enable: { l: "Identification (ReID)", d: "Compares face against vector database.", h: "Enables identification (ReID).\nIf OFF: faces may be detected, but all stay \"Unknown\".\nRecommended ON.\nUseful OFF for debugging FaceDet only." },
      rec_align: { l: "Face Alignment", d: "Mathematically rotates face to align eyes.", h: "Aligns faces using landmarks (straightens the eye line).\nKeep ON in almost all cases.\nTurn OFF only if your face model has no landmarks (or for testing).\nBad alignment = bad embeddings = wrong names." },
      rec_model: { l: "Comparison Model (ReID)", d: "The 'Brain' deciding who is who.", h: "ReID model that turns a face into an embedding vector.\nIf you change the model: rebuild the face DB!\nOtherwise stored vectors won't match -> wrong/Unknown.\nBigger models can be more accurate but heavier." },
      rec_db: { l: "Vector Database", d: "File containing biometric fingerprints.", h: "Vector database file (FAISS + meta JSON).\nBuilt from the enroll folder (\"Rebuild DB\").\nMissing/empty DB -> only \"Unknown\".\nAlways rebuild DB after changing the ReID model." },
      rec_preprocess: { l: "Image Normalization", d: "Technical preprocessing of pixel values.", h: "Pixel normalization before ReID.\nKeep \"Auto\" (GUI can choose the right mode).\nWrong setting makes embeddings useless -> all Unknown or wrong matches.\nChange only if you know the model's expected preprocessing." },
      rec_thres: { l: "Match Threshold", d: "Similarity required (0.0 - 1.0).", h: "Similarity threshold for a name match (0.0-1.0).\nHigher = stricter (fewer false IDs, more Unknown).\nLower = more matches (risk of wrong names).\nTypical: 0.50-0.65.\nWrong names -> raise. Never recognized -> lower." },
      unknown_enable: { l: "Save Strangers", d: "Saves faces without name to disk.", h: "Saves unknown faces to disk.\nThis enables training/clustering later (assign to a person).\nDisable only for privacy or storage reasons.\nIf OFF: no new training data is collected." },
      unknown_dir: { l: "Storage Folder", d: "Path where unknown faces are saved.", h: "Folder where unknown faces are stored.\nDefault: ./unknown_faces\nMake sure you have enough disk space.\nTip: clean up regularly or use clustering tools." },
      enroll_prune_enable: { l: "Enroll: DB Clean/Prune", d: "Cleans duplicates/outliers when building DB.", h: "Automatically clean the DB build (remove outliers/duplicates).\nRecommended ON (cleaner DB, fewer false IDs).\nIf you have very few images and want to keep all: turn OFF.\nSee details in the enroll log." },
      enroll_prune_min_sim: { l: "Enroll: Min Similarity", d: "Minimum similarity to keep an image.", h: "How tolerant pruning is to ‘different' images.\nHigher = stricter (cleaner, less variation).\nLower = more tolerant (more variation, more junk).\nTypical: 0.25-0.40." },
      enroll_prune_dedup_sim: { l: "Enroll: Dedup Similarity", d: "Similarity above which images are treated as duplicates.", h: "Duplicate threshold during pruning.\nKeep very high (0.98-0.995) to avoid deleting too much.\nLots of burst shots -> slightly lower.\nToo much removed -> raise." },
      enroll_prune_keep_top: { l: "Enroll: Keep Top", d: "Max images per person after pruning.", h: "Max images/vectors per person after pruning.\nLimits DB size and removes ‘100 near-identical' shots.\nTypical: 20-50.\nIncrease if you need many variants." },
      enroll_blur_skip_enable: { l: "Enroll: Blur Filter", d: "Skips blurry images during enrollment.", h: "Skip blurry training images during DB build.\nRecommended ON, otherwise the DB learns \"mush\".\nIf you have very few images: try OFF.\nAffects DB build only, not live detection." },
      enroll_blur_var_thres: { l: "Enroll: Sharpness Threshold", d: "Sharpness cutoff (higher=stricter).", h: "Blur threshold for enroll (higher = stricter).\nBelow this, the image is not added to the DB.\nTypical: 40-80.\nToo many rejected -> lower.\nIf DB is noisy -> raise (learn only sharp images)." },
      enroll_dedup_enable: { l: "Enroll: Deduplicate", d: "Removes duplicates before writing the DB.", h: "Remove very similar images per person before saving.\nRecommended ON (more diversity, smaller DB).\nTurn OFF only if you intentionally want many near-duplicates." },
      enroll_dedup_sim: { l: "Enroll: Dedup Threshold", d: "Duplicate threshold for pre-filtering.", h: "Similarity threshold for dedup.\nHigher = only near-identical images are removed.\nTypical: 0.99-0.999.\nToo low = too aggressive." },
      enroll_case_insensitive_folders: { l: "Enroll: Ignore Case", d: "Treat folder names case-insensitively.", h: "Treat folder names case-insensitively.\nPrevents duplicates like ‘Sarah' and ‘sarah'.\nRecommended ON.\nTurn OFF only if you intentionally want separate identities." },
      device_yolo: { l: "Hardware: Object Detection (YOLO)", d: "Select device for YOLO.", h: "Hardware used for object detection (person/car/...).\nGPU = usually highest FPS, NPU = efficient, CPU = fallback.\nIf YOLO is slow: switch to GPU or increase keyframe_interval.\nIf YOLO is unstable: try CPU." },
      device_face_det: { l: "Hardware: Face Detection", d: "Select device for face detection.", h: "Hardware for face detection (find face boxes/landmarks).\nYou can move it off the NPU to reduce contention.\nIf NPU is saturated: try GPU/CPU.\nIf faces are missing: try CPU to verify the model works." },
      device_reid: { l: "Hardware: Face Identification (ReID)", d: "Select device for identification.", h: "Hardware for identification (name vs Unknown).\nNPU is efficient, GPU often fast, CPU reliable.\nIf names appear too late: lower reid_votes or use GPU.\nWrong names are usually threshold/DB/quality issues." },
      ov_pool_cap: { l: "OpenVINO: Pool Cap", d: "Max infer requests per model (upper bound).", h: "Upper bound for parallel infer requests per model.\nHigher = more parallelism/throughput, but more RAM/threads.\nToo high can overload NPU/GPU or cause instability.\nTypical: CPU 1-2 * NPU 1-4 * GPU 4-8.\nChange only if you know why." },
      ov_pool_min: { l: "OpenVINO: Pool Min", d: "Min infer requests per model (lower bound).", h: "Lower bound for parallel infer requests per model.\nLATENCY: often 1.\nTHROUGHPUT: often 2-4.\nHigher can reduce stutter but increases load/RAM.\nIf you see hiccups: try a slightly higher MIN." },
      yolo_max_det: { l: "YOLO Max Detections", d: "Max boxes per image (after NMS).", h: "Max boxes per image after NMS.\nLower = faster but may drop objects in crowded scenes.\nHigher = keeps more, but costs performance.\nTypical: 50-150.\nIf overloaded: lower. If crowds are missing: raise." },
      face_min_conf: { l: "Min Conf (Quick Test)", d: "Minimum confidence in quick-test/stream overlay.", h: "GUI/overlay filter only (quick test). No effect on real detection.\nUse it to hide very low-confidence boxes.\nTypical: 0.40-0.55." },
      enroll_root: { l: "Enroll Folder", d: "Root folder for person images.", h: "Training root folder (one subfolder per person).\nLayout: enroll/<Name>/*.jpg\nAfter changes: run \"Rebuild DB\".\nTip: use varied images (lighting/angles/accessories)." },
      enroll_out_db: { l: "Enroll Output DB", d: "Target path of FaceDB (.faiss/.json).", h: "Target path of the generated face DB (FAISS).\nWritten during DB build.\nChanging it creates a new DB file.\nTip: use .faiss (meta is stored next to it as .json)." },
      enroll_prune_out_db: { l: "Enroll Prune Output DB", d: "Alternative DB for prune/debug.", h: "Optional target for the cleaned/pruned DB.\nUseful if you want to keep both original and clean versions.\nIf empty, a default name is used.\nWritten during build/prune." },
      enroll_keep_quantile: { l: "Enroll: Keep Quantile", d: "Keep only top X% by quality/score.", h: "Quality filter: keep only the best X% of images.\nExample 0.25 = top 25% only (very strict).\nTypical: 0.20-0.60.\nToo few vectors -> increase. DB too noisy -> decrease." },
      enroll_prototypes_k: { l: "Enroll: Prototypes K", d: "Prototypes per person (compression).", h: "Compress many images into K prototypes per person.\nMore = more robust (angles), but larger DB.\nTypical: 1-5.\nIncrease if a person varies a lot (glasses/hat)." },
      rec_blur_enable: { l: "Blur Filter", d: "Ignores images with motion blur.", h: "Sharpness filter before ReID: blurred faces are discarded.\nRecommended ON.\nReduces false IDs from motion blur / night smear.\nIf many sharp faces become Unknown: adjust the threshold." },
      rec_blur_thres: { l: "Sharpness Score", d: "Min edge contrast. Higher = Stricter.", h: "Sharpness score threshold (higher = stricter).\nBelow this, the face is treated as too blurry -> Unknown.\nTypical: 40-80.\n4K: 80-120. Low-res cameras: 30-50.\nToo many rejected -> lower." },
      log_rotate_enable: { l: "File Logging", d: "Writes events/errors to 'obf_app.log'.", h: "Write logs to a file (obf_app.log) and rotate when full.\nRecommended ON for troubleshooting.\nIf OFF: logs are console only." },
      log_rotate_mb: { l: "Max Log Size", d: "Rotates log after X MB.", h: "Max size of the log file before rotation.\nSmaller = less disk, larger = more history.\nTypical: 5-20 MB.\nIncrease for long debug sessions." },
      host: { l: "Server Host (Bind)", d: "IP/hostname the server binds to.", h: "Host/IP the web server binds to.\n0.0.0.0 = reachable on your LAN.\n127.0.0.1 = local only.\nRequires restart.\nTip: keep 0.0.0.0 for network access." },
      port: { l: "Server Port", d: "TCP port of the web interface/API.", h: "TCP port of the GUI/API.\nDefault: 32168.\nIf busy: pick a free port (e.g. 32169).\nRequires restart.\nAdjust firewall/reverse-proxy if needed." },
      loglevel: { l: "Log Level", d: "Logging verbosity (console + file).", h: "How verbose logging is.\nINFO = normal.\nDEBUG = lots of details (can impact performance).\nWARNING/ERROR = quieter.\nRequires restart." }
    }
  }
};

function tr(key, vars, forceLang) {
  const lang = forceLang || CUR_LANG || "en";
  const L = TRANS[lang] || TRANS.en || {};
  const fallback = TRANS.en || {};
  let s = (L && Object.prototype.hasOwnProperty.call(L, key) ? L[key] : (fallback[key] ?? key));
  if (s == null) s = key;
  s = String(s);
  if (vars && typeof vars === "object") {
    for (const k of Object.keys(vars)) {
      const v = vars[k];
      s = s.replace(new RegExp('\\{' + k + '\\}', 'g'), String(v));
    }
  }
  return s;
}

function setLanguage(lang) {
  CUR_LANG = lang; localStorage.setItem("obf_lang", lang);
  document.querySelectorAll(".lang-opt").forEach(el => el.classList.remove("active"));
  const btn = document.getElementById("lang_" + lang); if(btn) btn.classList.add("active");


  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    const val = tr(key, null, lang);
    if (val && val !== key) el.textContent = val;
  });


  document.querySelectorAll("[data-i18n-title]").forEach(el => {
    const key = el.getAttribute("data-i18n-title");
    const val = tr(key, null, lang);
    if (val && val !== key) el.title = val;
  });


  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
    const key = el.getAttribute("data-i18n-placeholder");
    const val = tr(key, null, lang);
    if (val && val !== key) el.placeholder = val;
  });


  try { document.title = tr("page_title", null, lang); } catch(e) {}

  if(CFG && SCHEMA) renderParams();
  if(MODEL_CATALOG.length) renderModelTable();
  if(document.getElementById("benchHeaderRow")) renderBenchmarkTable([]);

  updateBenchHlBtn();
}

function toggleTheme() {
  const html = document.documentElement;

  if (html.getAttribute('data-theme') === 'light') {
    html.removeAttribute('data-theme');
    localStorage.setItem('theme', 'dark');
  } else {
    html.setAttribute('data-theme', 'light');
    localStorage.setItem('theme', 'light');
  }
}


if (localStorage.getItem('theme') === 'light') {
  document.documentElement.setAttribute('data-theme', 'light');
}

function el(id){ return document.getElementById(id); }
function esc(s){ return (s||"").toString().replace(/[&<>"']/g,m=>({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;" }[m])); }
function toast(msg, err){
  const t = el("toast"); if(!t) return;
  t.textContent = msg; t.style.display="block";
  t.style.borderColor = err ? "rgba(239,68,68,0.5)" : "rgba(34,197,94,0.5)";
  setTimeout(()=>{ t.style.display="none"; }, 4000);
}
async function apiGet(url){ const r=await fetch(url); if(!r.ok) throw new Error("API Error "+r.status); return await r.json(); }
async function apiPost(url, body){ const r=await fetch(url,{method:"POST",headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}); return await r.json(); }


let socket = null;
let wsRetryTimer = null;
let wsPingTimer = null;
const WS_PING_MS = 15000;

function connectWS() {

  if (wsPingTimer) { clearInterval(wsPingTimer); wsPingTimer = null; }

  const proto = (location.protocol === "https:") ? "wss:" : "ws:";
  const url = proto + "//" + location.host + "/ws";
  console.log("Connecting WS:", url);

  const setOffline = () => {
    const stat = el("statusText");
    const dot  = el("statusDot");
    if (stat) { stat.textContent = tr("offline"); stat.style.color = "var(--bad)"; }
    if (dot)  { dot.style.display = "none"; }


    const ids = ["p_req","p_ema","p_last","p_ips","p_labels","p_req_side","p_ema_side","p_last_side","p_ips_side","p_labels_side"];
    ids.forEach(id => { const x = el(id); if (x) x.textContent = (id.includes("labels") ? "-" : "0"); });
  };

  const setOnline = () => {
    const stat = el("statusText");
    const dot  = el("statusDot");
    if (stat) { stat.style.color = "var(--ok)"; }

    if (dot)  { dot.style.display = "none"; }
  };

  socket = new WebSocket(url);

  socket.onopen = () => {
    console.log("WS Connected");
    setOnline();
    refreshModelLog();
    pollEnroll();


    if (wsPingTimer) clearInterval(wsPingTimer);
    wsPingTimer = setInterval(() => {
      try {
        if (socket && socket.readyState === 1) {
          socket.send(JSON.stringify({ type: "ping", t: Date.now() }));
        }
      } catch (e) {  }
    }, WS_PING_MS);
  };

  socket.onmessage = (event) => {
    try { handleWSMessage(JSON.parse(event.data)); }
    catch (e) { console.error("WS Parse Error", e); }
  };

  socket.onerror = () => {
    setOffline();
  };

  socket.onclose = () => {
    console.log("WS Disconnected. Retry...");
    setOffline();

    if (wsPingTimer) { clearInterval(wsPingTimer); wsPingTimer = null; }

    socket = null;
    if (wsRetryTimer) clearTimeout(wsRetryTimer);
    wsRetryTimer = setTimeout(connectWS, 2000);
  };
}


function handleWSMessage(msg) {
  if (!msg || !msg.type) return;

  const setTxt = (id, val) => {
    const e = el(id);
    if (e) e.textContent = (val ?? "").toString();
  };

  if (msg.type === "status") {
    const d = msg.data || {};
    const stTxt = el("statusText");
    const dot = el("statusDot");
    if (stTxt) {
      stTxt.textContent = d.pipeline_enabled ? tr("state_active") : tr("state_paused");
      stTxt.style.color = d.pipeline_enabled ? "var(--ok)" : "var(--muted)";
      if (dot) dot.style.display = d.pipeline_enabled ? "inline-block" : "none";
    }
    setTxt("uptimeText", d.uptime ?? "-");
    setTxt("p_req", d.req_count ?? 0);
    setTxt("p_req_side", d.req_count ?? 0);
    setTxt("p_ema", d.ema_ms ?? 0);
    setTxt("p_ema_side", d.ema_ms ?? 0);
    setTxt("p_last", d.last_ms ?? 0);
    setTxt("p_last_side", d.last_ms ?? 0);
    setTxt("p_ips", d.fps ?? 0);
    setTxt("p_ips_side", d.fps ?? 0);
    const labels = (d.last_labels || []).join(", ") || "-";
    setTxt("p_labels", labels);
    setTxt("p_labels_side", labels);


    if ((d.last_labels || []).some(x => String(x).trim().toLowerCase() === "unknown")) {
      scheduleUnknownRefresh();
    }

    scheduleRuntimeModelsRefresh(150);
    return;
  }

  if (msg.type === "model_log") {
    const lines = msg.lines || [];
    const logBox = el("modelLog");
    if (logBox) {
      logBox.innerHTML = lines.map(l => {
        const s = esc(l);
        if (l.includes("Error") || l.includes("Fehler")) return `<span style="color:#ef4444">${s}</span>`;
        if (l.includes("Download")) return `<span style="color:#3b82f6">${s}</span>`;
        if (l.includes("Fertig") || l.includes("Success") || l.includes("Installiert")) return `<span style="color:#22c55e">${s}</span>`;
        return `<span>${s}</span>`;
      }).join("\n");
      logBox.scrollTop = logBox.scrollHeight;
    }
    if (modelLooksDone(lines)) {
      finalizeModelInstallWatch();
    }
    return;
  }

  if (msg.type === "enroll_log") {
    const lines = msg.lines || [];
    const enBox = el("enrollLog");
    if (enBox) {
      enBox.innerHTML = lines.map(l => {
        const s = esc(l);
        if (l.includes("Error") || l.includes("Fehler")) return `<span style="color:#ef4444">${s}</span>`;
        if (l.includes("Fertig") || l.includes("Success") || l.includes("gespeichert") || l.includes("Job erfolgreich"))
          return `<span style="color:#22c55e">${s}</span>`;
        return `<span>${s}</span>`;
      }).join("\n");
      enBox.scrollTop = enBox.scrollHeight;
    }
    if (enrollLooksDone(lines)) {
    setFaceDbDirty(false);
    refreshFaceDbStats();
  }
    return;
  }
}


function showTab(id) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
    const pane = document.getElementById('tab-' + id);
    if (pane) pane.classList.add('active');
    const btn = document.querySelector(`.tab-btn[onclick="showTab('${id}')"]`);
    if (btn) btn.classList.add('active');
    if (id === 'models') reloadModels().catch(console.error);
    if (id === 'dashboard') startRuntimeModelsAutoRefresh(2000);
    else stopRuntimeModelsAutoRefresh();

    if (id === 'benchmark') refreshBenchmarkOnce().catch(console.error);
    if (id === 'benchmark') updateBenchHlBtn();
    if (id === 'clusters' && typeof loadUnknownClusters === 'function') loadUnknownClusters().catch(console.error);
}
window.showTab = showTab;

let BENCH_HIGHLIGHT_MODE = (localStorage.getItem("bench_hl_mode") || "L");
let LAST_BENCH_RESULTS_UI = [];

function benchHlLabel(mode) {
  const m = (mode || BENCH_HIGHLIGHT_MODE);
  const off = tr("bench_hl_off");
  const label = (m === "OFF") ? off : m;
  return `Highlight: ${label}`;
}

function updateBenchHlBtn() {
  const b = el("benchHighlightToggle");
  if (!b) return;
  b.textContent = benchHlLabel();
  b.title = tr("bench_hl_title");
}

function toggleBenchHighlight() {
  BENCH_HIGHLIGHT_MODE = (BENCH_HIGHLIGHT_MODE === "OFF") ? "L" : (BENCH_HIGHLIGHT_MODE === "L" ? "T" : "OFF");
  localStorage.setItem("bench_hl_mode", BENCH_HIGHLIGHT_MODE);
  updateBenchHlBtn();

  if (Array.isArray(LAST_BENCH_RESULTS_UI) && LAST_BENCH_RESULTS_UI.length) {
    renderBenchmarkTable(LAST_BENCH_RESULTS_UI);
  } else {
    refreshBenchmarkOnce().catch(console.error);
  }
}
window.toggleBenchHighlight = toggleBenchHighlight;


function updateBenchDevices() {
    const sel = el("benchDevice");
    if(!sel || !SCHEMA || !SCHEMA.device) return;
    sel.innerHTML = "";
    (SCHEMA.device.options || ["CPU"]).forEach(d => {
        const opt = document.createElement("option"); opt.value = d; opt.textContent = d;
        if(CFG && d === CFG.device) opt.selected = true;
        sel.appendChild(opt);
    });
}

async function refreshBenchmarkOnce() {
    const data = await apiGet("/gui/benchmark/results");
    if (data && Array.isArray(data.results)) renderBenchmarkTable(data.results);
}

async function runBenchmark() {
  console.log("Starting Benchmark.");
  const ids = MODEL_CATALOG.filter(m => m.exists).map(m => m.id);

  if (!ids.length) {
    console.warn("No models found via CATALOG:", MODEL_CATALOG);
    return toast(tr("bench_no_models"), true);
  }

  const pContainer = el("benchProgressContainer");
  if (pContainer) pContainer.style.display = "block";

  const pBar = el("benchProgressBar");
  if (pBar) {
    pBar.style.width = "1%";
    pBar.innerText = "0%";
    pBar.style.background = "#0078d4";
  }

  el("benchResultsBody").innerHTML =
    `<tr><td colspan="10" style="padding:40px; text-align:center;">${esc(tr("bench_running"))}</td></tr>`;

  try {
    await apiPost("/gui/benchmark/run", { ids });

    let lastRenderKey = "";

    const poll = async () => {
      const data = await apiGet("/gui/benchmark/results");


      if (pBar && typeof data.progress === "number" && data.progress >= 0) {
        pBar.style.width = data.progress + "%";
        pBar.innerText = data.progress + "%";
      }


      const results = Array.isArray(data.results) ? data.results : [];
      const lastId = results.length ? (results[results.length - 1]?.id ?? "") : "";
      const renderKey = `${results.length}:${lastId}:${data.progress ?? ""}`;

      if (results.length > 0 && renderKey !== lastRenderKey) {
        lastRenderKey = renderKey;
        renderBenchmarkTable(results);
      }


      const done = (typeof data.progress === "number" && data.progress >= 100) || data.running === false;
      if (done) {
        if (results.length > 0) renderBenchmarkTable(results);
        if (pBar) pBar.style.background = "var(--ok)";
        toast(tr("bench_done_toast"));
        if (pContainer) setTimeout(() => { pContainer.style.display = "none"; }, 5000);
        return true;
      }
      return false;
    };


    if (await poll()) return;

    const pollInterval = setInterval(async () => {
      try {
        if (await poll()) clearInterval(pollInterval);
      } catch (e) {
        console.warn("Benchmark poll failed:", e);
      }
    }, 1000);

  } catch (e) {
    console.error("Benchmark Error:", e);
    el("benchResultsBody").innerHTML =
      `<tr><td colspan="4" class="badtxt">${esc(tr("bench_error_row", {err: (e?.message || e)}))}</td></tr>`;
  }
}


function renderBenchmarkTable(results) {
  const body = el("benchResultsBody");
  const head = el("benchHeaderRow");
  if (!body || !head) return;


  LAST_BENCH_RESULTS_UI = Array.isArray(results) ? results : [];

  updateBenchHlBtn();

  const T = TRANS[CUR_LANG] || {};
  const colModel = T.col_model || "Model";


  const devices = ["CPU", "GPU", "NPU"];
  head.innerHTML = `<th>${colModel}</th>` + devices.map(d => `<th>${d}</th>`).join("");

  body.innerHTML = "";
  if (!results || !results.length) return;

  const mode = (BENCH_HIGHLIGHT_MODE || "L");

  results.forEach(row => {
    const tr = document.createElement("tr");
    let html = `<td><b class="mono">${row.id}</b></td>`;


    const getLatMs = (d) => {
      if (!d || d === "N/A") return Infinity;
      const v = (typeof d.lat_ms === "number") ? d.lat_ms : ((typeof d.ms === "number") ? d.ms : Infinity);
      return (v > 0) ? v : Infinity;
    };
    const getLatFps = (d) => {
      if (!d || d === "N/A") return 0;
      const v = (typeof d.lat_fps === "number") ? d.lat_fps : ((typeof d.fps === "number") ? d.fps : 0);
      return (v > 0) ? v : 0;
    };
    const getThrFps = (d) => {
      if (!d || d === "N/A") return 0;
      const v = (typeof d.thr_fps === "number") ? d.thr_fps : 0;
      return (v > 0) ? v : 0;
    };
    const getThrReq = (d) => {
      if (!d || d === "N/A") return null;
      return (typeof d.thr_nireq === "number") ? d.thr_nireq : null;
    };


    let bestDev = null;
    if (mode === "L") {
      let best = Infinity;
      devices.forEach(dev => {
        const d = row.benchmarks ? row.benchmarks[dev] : null;
        const ms = getLatMs(d);
        if (ms < best) { best = ms; bestDev = dev; }
      });
      if (!isFinite(best)) bestDev = null;
    } else if (mode === "T") {
      let best = -Infinity;
      devices.forEach(dev => {
        const d = row.benchmarks ? row.benchmarks[dev] : null;
        const fps = getThrFps(d) || getLatFps(d);
        if (fps > best) { best = fps; bestDev = dev; }
      });
      if (!(best > 0)) bestDev = null;
    } else {
      bestDev = null;
    }

    devices.forEach(dev => {
      const data = row.benchmarks ? row.benchmarks[dev] : null;

      if (!data || data === "N/A") {
        html += `<td class="muted">-</td>`;
        return;
      }

      const latMs = getLatMs(data);
      const latFps = getLatFps(data);
      const thrFps = getThrFps(data);
      const thrReq = getThrReq(data);

      const isBest = (bestDev && dev === bestDev);
      const style = isBest
        ? 'background: rgba(34, 197, 94, 0.15); border: 1px solid var(--ok);'
        : '';

      if (thrFps > 0) {
        html += `
          <td style="${style}">
            <div class="mono"><b>L:</b> ${Number.isFinite(latMs) ? latMs.toFixed(2) : "-"} ms <span class="muted small">(${latFps.toFixed(1)} FPS)</span></div>
            <div class="mono"><b>T:</b> ${thrFps.toFixed(1)} FPS ${thrReq != null ? `<span class="muted small">(req ${thrReq})</span>` : ""}</div>
          </td>`;
      } else {
        // fallback (falls nur alt-format vorhanden)
        html += `<td style="${style}"><div class="mono"><b>${Number.isFinite(latMs) ? latMs.toFixed(2) : "-"}</b> ms</div><div class="muted small">${latFps.toFixed(1)} FPS</div></td>`;
      }
    });

    tr.innerHTML = html;
    body.appendChild(tr);
  });
}

function renderModelTable(activePaths = []) {
  const box = el("modelListContainer");
  if (!box) return;

  const T = TRANS[CUR_LANG] || {};
  if (!MODEL_CATALOG || !MODEL_CATALOG.length) {
    box.innerHTML = `<div style="padding:20px;" class="muted">${esc(T.models_empty || "Empty")}</div>`;
    return;
  }

  const thModel  = T.models_table_model  || "Model";
  const thInfo   = T.models_table_info   || "Info";
  const thSize   = T.models_table_size   || "Size";
  const thStatus = T.models_table_status || "Status";
  const thAction = T.models_table_action || "Action";

  let html = `<table class="mod-table" style="width:100%">
    <thead>
      <tr>
        <th width="30">#</th>
        <th>${esc(thModel)}</th>
        <th>${esc(thInfo)}</th>
        <th>${esc(thSize)}</th>
        <th>${esc(thStatus)}</th>
        <th>${esc(thAction)}</th>
      </tr>
    </thead>
    <tbody>`;

  MODEL_CATALOG.forEach(m => {
    const exists = m.exists === true;
    const isActive = exists && activePaths.some(p => p && p.includes(m.filename));

    const chkTitle = T.models_in_use || (CUR_LANG === "de" ? "In Verwendung" : "In use");
    const chkHtml = isActive
      ? `<input type="checkbox" disabled title="${esc(chkTitle)}" style="opacity:0.5">`
      : `<input type="checkbox" class="mod-chk" value="${esc(m.id)}">`;

    let btnHtml = `<span class="muted">-</span>`;
    if (exists) {
      if (isActive) {
        const activeLbl = T.models_status_active || (CUR_LANG === "de" ? "AKTIV" : "ACTIVE");
        btnHtml = `<span class="pill" style="background:rgba(59,130,246,0.1); border:1px solid var(--acc); color:var(--acc); font-size:10px; cursor:default;">${esc(activeLbl)}</span>`;
      } else {
        const delLbl = T.btn_delete || "Delete";
        btnHtml = `<button class="btn bad" style="padding:2px 8px; font-size:11px;" onclick="deleteModel('${esc(m.id)}')">${esc(delLbl)}</button>`;
      }
    }

    const st = exists ? (T.models_status_ready || "READY") : (T.models_status_missing || "MISSING");

    html += `<tr>
      <td>${chkHtml}</td>
      <td><div class="mod-name" ${isActive ? 'style="color:var(--acc)"' : ''}>${esc(m.name || m.id)}</div><div class="mod-file">${esc(m.filename || "")}</div></td>
      <td><span class="mod-type">${esc(String(m.type || "").toUpperCase())}</span><div class="mod-desc">${esc(m.desc || "")}</div></td>
      <td class="mono small">${m.size_mb ? esc(m.size_mb + " MB") : "-"}</td>
      <td><b style="${exists ? "color:var(--ok)" : "color:var(--bad)"}; font-size:11px;">${esc(st)}</b></td>
      <td>${btnHtml}</td>
    </tr>`;
  });

  box.innerHTML = html + `</tbody></table>`;
}

function renderModelSummary(data) {
  const box = el("modelSummaryBox");
  if (!box) return;
  box.innerHTML = "";

  [
    ["YOLO",    data.yolo_model,     data.yolo_exists],
    ["FaceDet", data.face_det_model, data.face_det_exists],
    ["ReID",    data.reid_model,     data.reid_exists],
    ["DB",      data.db_path,        data.db_exists],
  ].forEach(([k, v, ok]) => {
    const r = document.createElement("div");
    r.className = "row";
    r.style.marginBottom = "4px";

    r.innerHTML =
      `<div class="mono" style="min-width:70px;color:#9aa4b2">${esc(k)}</div>` +
      `<div class="mono" style="flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(v || "-")}</div>` +
      `<div class="${ok ? "oktxt" : "badtxt"}">${ok ? "OK" : esc(tr("models_status_missing"))}</div>`;

    box.appendChild(r);
  });
}

let _lastRuntimeFetch = 0;

function renderRuntimeModels(data) {
  const box = el("modelRuntimeBox"); if (!box) return;
  box.innerHTML = "";

  if (!data || data.loaded !== true) {
    box.innerHTML = `<div class="muted small">(${esc(tr("runtime_offline"))})</div>`;
    return;
  }

  const mn  = (data.ov_pool_min != null) ? data.ov_pool_min : "-";
  const cap = (data.ov_pool_cap != null) ? data.ov_pool_cap : "-";

  const hdr = document.createElement("div");
  hdr.className = "muted small";
  hdr.style.borderTop = "1px dashed var(--border)";
  hdr.style.paddingTop = "8px";
  hdr.style.marginTop = "4px";
  hdr.textContent = tr("runtime_pools_hdr", {min: mn, cap: cap});
  box.appendChild(hdr);

  (data.models || []).forEach(m => {
    const role = (m.role || "?");
    const model = (m.model || "-");
    const dev = (m.device || "-");
    const hint = (m.hint || "-");

    const sel  = (m.pool_selected != null) ? String(m.pool_selected) : "-";
    const opt  = (m.pool_optimal != null) ? String(m.pool_optimal) : "-";
    const free = (m.pool_free != null) ? String(m.pool_free) : "-";

    const wrap = document.createElement("div");
    wrap.style.display = "grid";
    wrap.style.gridTemplateColumns = "70px 1fr auto";
    wrap.style.columnGap = "10px";
    wrap.style.rowGap = "2px";
    wrap.style.alignItems = "center";
    wrap.style.marginTop = "8px";

    wrap.innerHTML = `
      <div class="mono" style="color:var(--muted)">${esc(role)}</div>
      <div class="mono" style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">${esc(model)}</div>
      <div class="mono small" style="text-align:right;white-space:nowrap;">${esc(dev)} • ${esc(hint)}</div>

      <div></div>
      <div class="muted small" style="grid-column:2 / span 2;">
        ${esc(tr("runtime_pool_line", {sel: sel, opt: opt, free: free}))}
      </div>
    `;
    box.appendChild(wrap);
  });
}


let _RUNTIME_LAST_RENDER_KEY = null;

function _runtimeRenderKey(data) {

  const lang = (typeof CUR_LANG === "string" ? CUR_LANG : "");

  if (!data || data.loaded !== true) return `${lang}|offline`;

  const mn  = (data.ov_pool_min != null) ? data.ov_pool_min : "-";
  const cap = (data.ov_pool_cap != null) ? data.ov_pool_cap : "-";

  const models = (data.models || []).map(m => ([
    m.role || "",
    m.model || "",
    m.device || "",
    m.hint || "",
    (m.pool_selected != null ? String(m.pool_selected) : "-"),
    (m.pool_optimal  != null ? String(m.pool_optimal)  : "-"),
    (m.pool_free     != null ? String(m.pool_free)     : "-"),
  ].join("|"))).join(";");

  return `${lang}|on|min=${mn}|cap=${cap}|${models}`;
}

function renderRuntimeModelsIfChanged(data, force = false) {
  const key = _runtimeRenderKey(data);
  if (!force && key === _RUNTIME_LAST_RENDER_KEY) return false;
  _RUNTIME_LAST_RENDER_KEY = key;
  renderRuntimeModels(data);
  return true;
}


async function loadRuntimeModels(force=false) {
  const now = Date.now();
  if (!force && (now - _lastRuntimeFetch) < 10000) return;
  _lastRuntimeFetch = now;

  try {
    const data = await apiGet("/gui/runtime/models");
    renderRuntimeModelsIfChanged(data, force);
  } catch (e) {
    const box = el("modelRuntimeBox");
    if (box) box.innerHTML = `<div class="muted small">(${esc(tr("runtime_error"))})</div>`;
  }
}


let RUNTIME_AUTO_TIMER = null;
let _lastRuntimeLiveFetch = 0;
let _runtimeLivePending = false;

function _runtimeStatsVisible() {
  if (document.hidden) return false;


  const dash = document.getElementById("tab-dashboard");
  if (dash && !dash.classList.contains("active")) return false;


  return !!el("modelRuntimeBox");
}

async function loadRuntimeModelsLive() {
  const now = Date.now();

  if ((now - _lastRuntimeLiveFetch) < 1500) return;
  _lastRuntimeLiveFetch = now;

  if (!_runtimeStatsVisible()) return;

  try {
    const data = await apiGet("/gui/runtime/models");
    renderRuntimeModelsIfChanged(data, false);
  } catch (e) {
    const box = el("modelRuntimeBox");
    if (box) box.innerHTML = `<div class="muted small">(Runtime: Error)</div>`;
  }
}

function scheduleRuntimeModelsRefresh(delayMs = 200) {
  if (_runtimeLivePending) return;
  if (!_runtimeStatsVisible()) return;

  _runtimeLivePending = true;
  setTimeout(async () => {
    _runtimeLivePending = false;
    await loadRuntimeModelsLive();
  }, delayMs);
}

function startRuntimeModelsAutoRefresh(ms = 2000) {
  stopRuntimeModelsAutoRefresh();

  loadRuntimeModelsLive();

  RUNTIME_AUTO_TIMER = setInterval(() => {
    loadRuntimeModelsLive();
  }, ms);
}

function stopRuntimeModelsAutoRefresh() {
  if (RUNTIME_AUTO_TIMER) {
    clearInterval(RUNTIME_AUTO_TIMER);
    RUNTIME_AUTO_TIMER = null;
  }
}


document.addEventListener("visibilitychange", () => {
  if (!document.hidden) scheduleRuntimeModelsRefresh(50);
});


async function downloadSelectedModels() {
  const selected = [];
  document.querySelectorAll('.mod-chk:checked').forEach(c => selected.push(c.value));

  if (!selected.length) {
    return toast(tr("toast_select_models"), true);
  }


  try { await reloadModels(); } catch (e) {}
  armModelInstallWatch(selected);

  try {
    await apiPost('/gui/model/start', { kind: 'download_selection', selection: selected });
    toast(tr("toast_download_started"), false);
    refreshModelLog();
  } catch (e) {
    toast(tr("toast_error"), true);
  }
}



async function deleteModel(id) {
    const T = TRANS[CUR_LANG] || {};
    const msg = (T.confirm_del_model || "Delete {name}?").replace("{name}", id);
    if (!confirm(msg)) return;
    try { const res = await apiPost("/gui/model/delete", { id: id }); if (res.ok) { toast(tr("toast_deleted"), false); await reloadModels(); } } catch (e) { toast(tr("toast_error"), true); }
}

function paramLevel(k, meta) {

  const lvl = meta && (meta.ui || meta.ui_level || meta.level) ? String(meta.ui || meta.ui_level || meta.level).toLowerCase() : "";
  if (lvl.includes("adv")) return "advanced";
  return "basic";
}

function syncAdvancedParamsUI() {
  const advBox = el("paramBoxAdvanced");
  if (advBox) advBox.style.display = ADV_PARAMS_OPEN ? "" : "none";

  const btn = el("btnAdvParams");
  if (btn) {
    const T = TRANS[CUR_LANG] || {};
    btn.textContent = ADV_PARAMS_OPEN ? (T.btn_adv_hide || "Hide") : (T.btn_adv_show || "Show");
  }
}

function toggleAdvancedParams(force) {
  if (typeof force === "boolean") ADV_PARAMS_OPEN = force;
  else ADV_PARAMS_OPEN = !ADV_PARAMS_OPEN;

  localStorage.setItem("obf_adv_params", ADV_PARAMS_OPEN ? "1" : "0");
  syncAdvancedParamsUI();
}
window.toggleAdvancedParams = toggleAdvancedParams;


function buildParamControl(k, meta, val, txt) {
  const label = (txt && txt.l) || meta.label || k;
  const desc  = (txt && txt.d) || meta.desc || "";
  const help  = (txt && txt.h) || meta.help || "";

  const wrap = document.createElement("div"); wrap.className = "ctrl";
  const left = document.createElement("div"); left.className = "left";
  const right = document.createElement("div"); right.className = "right";

  const helpHtml = help ? `<div class="tooltip">?<span class="tiptext">${esc(help)}</span></div>` : "";
  left.innerHTML = `<div><b>${esc(label)}</b>${helpHtml} <span class="small mono">${esc(k)}</span></div><div class="muted">${esc(desc)}</div>`;


  if (k === "rec_preprocess") {
    const isAuto = (val === "auto" || val === "");

    const div = document.createElement("div");
    div.style.display = "flex";
    div.style.alignItems = "center";
    div.style.gap = "10px";

    const lbl = document.createElement("label");
    lbl.style.display = "flex";
    lbl.style.alignItems = "center";
    lbl.style.gap = "4px";
    lbl.style.cursor = "pointer";

    const chk = document.createElement("input");
    chk.type = "checkbox";
    chk.id = "chk_auto_preprocess";
    chk.checked = isAuto;

    const sp = document.createElement("span");
    sp.className = "muted";
    sp.textContent = (CUR_LANG === "de") ? "Auto" : "Auto";

    lbl.appendChild(chk);
    lbl.appendChild(sp);

    const sel = document.createElement("select");
    sel.id = "cfg_" + k;
    (meta.options || ["auto"]).forEach(o => {
      const op = document.createElement("option");
      op.value = o; op.textContent = o;
      sel.appendChild(op);
    });


    sel.value = (val ?? (meta.options || [])[0] ?? "");
    sel.disabled = chk.checked;

    const hint = document.createElement("span");
    hint.id = "rec_preprocess_hint";
    hint.className = "muted";
    hint.style.fontSize = "11px";

    chk.onchange = async () => {
      if (chk.checked) {
        sel.disabled = true;
        CFG[k] = "auto";
        sel.value = "auto";
        await reidCompatUpdate(true);
      } else {
        sel.disabled = false;
        CFG[k] = sel.value || "auto";
        await reidCompatUpdate(true);
      }
    };

    sel.onchange = async () => {
      if (chk.checked) return;
      CFG[k] = sel.value;
      await reidCompatUpdate(true);
    };

    div.appendChild(lbl);
    div.appendChild(sel);
    div.appendChild(hint);
    right.appendChild(div);

    wrap.appendChild(left);
    wrap.appendChild(right);
    return wrap;
  }


  if (meta.type === "bool") {
    const inp = document.createElement("input");
    inp.type = "checkbox";
    inp.id = "cfg_" + k;
    inp.checked = !!val;
    inp.onchange = () => { CFG[k] = inp.checked; };
    right.appendChild(inp);

  } else if (meta.type === "int" || meta.type === "float") {
    const rng = document.createElement("input");
    rng.type = "range";
    rng.id = "cfg_" + k;
    rng.min = (meta.min ?? 0);
    rng.max = (meta.max ?? 1);
    rng.step = (meta.step ?? (meta.type === "int" ? 1 : 0.01));
    rng.value = (val ?? meta.min ?? 0);

    const out = document.createElement("span");
    out.className = "mono";
    out.style.minWidth = "70px";
    out.textContent = parseFloat(rng.value).toFixed(meta.type === "int" ? 0 : 2);

    rng.oninput = () => {
      out.textContent = parseFloat(rng.value).toFixed(meta.type === "int" ? 0 : 2);
      CFG[k] = (meta.type === "int") ? parseInt(rng.value) : parseFloat(rng.value);
    };

    right.appendChild(rng);
    right.appendChild(out);

  } else if (meta.type === "select") {
    const sel = document.createElement("select");
    sel.id = "cfg_" + k;

    (meta.options || []).forEach(o => {
      const op = document.createElement("option");
      op.value = o; op.textContent = o;
      sel.appendChild(op);
    });

    sel.value = (val ?? (meta.options || [])[0] ?? "");

    sel.onchange = async () => {
      CFG[k] = sel.value;

      if (k === "rec_model") {

        const ac = el("chk_auto_preprocess");
        const ps = el("cfg_rec_preprocess");
        if (ac && ps) {
          ac.checked = true;
          ps.disabled = true;
          CFG["rec_preprocess"] = "auto";
          ps.value = "auto";
        }
        await reidCompatUpdate(true);
      } else if (k === "rec_db") {
        await checkReidCompat();
      }
    };

    right.appendChild(sel);

  } else if (meta.type === "textarea") {
    const ta = document.createElement("textarea");
    ta.id = "cfg_" + k;
    ta.value = (val ?? "");
    ta.rows = 5;
    ta.style.width = "100%";
    ta.oninput = () => { CFG[k] = ta.value; };
    right.appendChild(ta);

  } else {
    const inp = document.createElement("input");
    inp.id = "cfg_" + k;
    inp.type = "text";
    inp.value = (val ?? "");
    inp.oninput = () => { CFG[k] = inp.value; };
    right.appendChild(inp);
  }

  wrap.appendChild(left);
  wrap.appendChild(right);
  return wrap;
}


async function checkPortNow() {

  const box = el("port_status_box");
  if (!box || !CFG) return true;

  const port = Number(CFG.port);
  const host = (CFG.host || "0.0.0.0").toString();

  if (!Number.isFinite(port) || port < 1 || port > 65535) {
    box.style.display = "inline-flex";
    box.style.borderColor = "rgba(239,68,68,0.35)";
    box.textContent = "❌ Port ungültig";
    return false;
  }

  box.style.display = "inline-flex";
  box.style.borderColor = "var(--border)";
  box.textContent = "⏳ Port check...";

  try {
    const url = `/gui/port/check?host=${encodeURIComponent(host)}&port=${encodeURIComponent(port)}`;
    const res = await apiGet(url);

    const ok = res && (res.status === "ok" || res.status === "warn");
    box.style.borderColor = ok ? "rgba(34,197,94,0.35)" : "rgba(239,68,68,0.35)";
    box.textContent = (ok ? "✅ " : "❌ ") + (res.message || (ok ? "OK" : "Belegt"));

    return ok;
  } catch (e) {

    box.style.borderColor = "rgba(239,68,68,0.25)";
    box.textContent = "Port check failed";
    return true;
  }
}

function schedulePortCheck(delayMs = 350) {
  if (_PORT_CHECK_TMR) clearTimeout(_PORT_CHECK_TMR);
  _PORT_CHECK_TMR = setTimeout(() => { checkPortNow(); }, delayMs);
}



function renderParams(){
  const basicBox = el("paramBasicBox") || el("paramBox");
  const advBox = el("paramAdvancedBox");

  if(!basicBox) return;
  basicBox.innerHTML = "";
  if(advBox) advBox.innerHTML = "";
  if(!SCHEMA || !CFG) return;

  const tParams = (TRANS[CUR_LANG] && TRANS[CUR_LANG].params) ? TRANS[CUR_LANG].params : {};

  const keys = Object.keys(SCHEMA);
  keys.forEach((k) => {
    const meta = SCHEMA[k] || {};
    const val = CFG[k];

    const txt = tParams[k] || {};
    const label = txt.l || meta.label || k;
    const desc  = txt.d || meta.desc  || "";
    const help  = txt.h || meta.help  || "";

    const wrap = document.createElement("div"); wrap.className = "ctrl";
    const left = document.createElement("div"); left.className = "left";
    const right = document.createElement("div"); right.className = "right";

    const helpHtml = help
      ? `<div class="tooltip">?<span class="tiptext">${esc(help)}</span></div>`
      : "";

    left.innerHTML = `
      <div><b>${esc(label)}</b>${helpHtml} <span class="mono muted">${esc(k)}</span></div>
      <div class="muted">${esc(desc)}</div>
    `;


    const type = (meta.type || "text").toLowerCase();

    const setVal = (v) => {
      CFG[k] = v;


      if (k === "host" || k === "port") schedulePortCheck(250);
    };

    if (type === "bool") {
      const inp = document.createElement("input");
      inp.type = "checkbox";
      inp.checked = !!val;
      inp.onchange = () => setVal(!!inp.checked);
      right.appendChild(inp);

    } else if (type === "select") {


      if (k === "rec_preprocess") {
        const isAuto = (val === "auto" || val === "" || val == null);

        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.gap = "10px";

        const lbl = document.createElement("label");
        lbl.style.display="flex";
        lbl.style.alignItems="center";
        lbl.style.gap="6px";
        lbl.style.cursor="pointer";

        const chk = document.createElement("input");
        chk.type = "checkbox";
        chk.checked = isAuto;

        const sel = document.createElement("select");
        (meta.options || []).forEach(o => {
          if (o === "auto") return;
          const op = document.createElement("option");
          op.value = o;
          op.textContent = o;
          sel.appendChild(op);
        });
        sel.value = isAuto ? (sel.options[0]?.value || "arcface") : val;
        sel.disabled = isAuto;

        chk.onchange = async () => {
          const autoOn = !!chk.checked;
          sel.disabled = autoOn;
          setVal(autoOn ? "auto" : sel.value);
          await reidCompatUpdate(true);
        };
        sel.onchange = async () => {
          setVal(sel.value);
          await reidCompatUpdate(true);
        };

        lbl.appendChild(chk);
        lbl.appendChild(document.createTextNode("Auto"));
        row.appendChild(lbl);
        row.appendChild(sel);
        right.appendChild(row);

      } else {
        const sel = document.createElement("select");
        (meta.options || []).forEach(o => {
          const op = document.createElement("option");
          op.value = o;
          op.textContent = o;
          sel.appendChild(op);
        });
        if (val != null) sel.value = val;

        sel.onchange = async () => {
          setVal(sel.value);
          if (k === "rec_model") await reidCompatUpdate(true);
          else if (k === "rec_db") await checkReidCompat();
        };

        right.appendChild(sel);
      }

    } else if (type === "int" || type === "float") {
      const min = (meta.min != null) ? Number(meta.min) : null;
      const max = (meta.max != null) ? Number(meta.max) : null;
      const step = (meta.step != null) ? Number(meta.step) : (type === "int" ? 1 : 0.01);


      if (min != null && max != null) {
        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.alignItems = "center";
        row.style.gap = "10px";
        row.style.minWidth = "260px";

        const range = document.createElement("input");
        range.type = "range";
        range.min = String(min);
        range.max = String(max);
        range.step = String(step);
        range.value = (val != null) ? String(val) : String(min);

        const num = document.createElement("input");
        num.type = "number";
        num.min = String(min);
        num.max = String(max);
        num.step = String(step);
        num.value = (val != null) ? String(val) : String(min);
        num.style.width = "90px";

        const parse = (x) => {
          const n = Number(x);
          if (Number.isNaN(n)) return (val != null ? Number(val) : min);
          return (type === "int") ? Math.round(n) : n;
        };

        range.oninput = () => {
          const v = parse(range.value);
          num.value = String(v);
          setVal(v);
        };
        num.oninput = () => {
          const v = parse(num.value);
          range.value = String(v);
          setVal(v);
        };

        row.appendChild(range);
        row.appendChild(num);
        right.appendChild(row);

      } else {
        const num = document.createElement("input");
        num.type = "number";
        num.step = String(step);
        num.value = (val != null) ? String(val) : "";
        num.oninput = () => {
          const n = Number(num.value);
          if (Number.isNaN(n)) return;
          setVal((type === "int") ? Math.round(n) : n);
        };
        right.appendChild(num);
      }

    } else if (type === "textarea") {
      const ta = document.createElement("textarea");
      ta.value = (val ?? "");
      ta.rows = 5;
      ta.style.width = "100%";
      ta.oninput = () => setVal(ta.value);
      right.appendChild(ta);

    } else {
      const inp = document.createElement("input");
      inp.type = "text";
      inp.value = (val ?? "");
      inp.oninput = () => setVal(inp.value);
      right.appendChild(inp);
    }

    wrap.appendChild(left);
    wrap.appendChild(right);


    const ui = String(meta.ui || meta.ui_level || meta.level || "basic").toLowerCase();
    const isAdv = ui.includes("adv");

    const target = (advBox && isAdv) ? advBox : basicBox;


    if (k === "port") {
    const b = document.createElement("div");
    b.id = "port_status_box";
    b.className = "pill";
    b.style.display = "inline-flex";
    b.style.marginTop = "8px";
    b.style.borderColor = "var(--border)";
    b.textContent = "…";
    right.appendChild(b);

    schedulePortCheck(0);
    }

    target.appendChild(wrap);
  });
}

async function checkReidCompat() {
    try {
        const res = await apiGet(`/gui/reid/check_compat`);
        const box = el("reid_status_box");
        if(box) box.innerHTML = res.status==="ok" ? "✅ OK" : "⚠️ Check";
    } catch(e){}
}

async function reidCompatUpdate(showToast=true){
  if(!CFG) return;
  let recPre = CFG.rec_preprocess || "auto";
  const autoChk = el("chk_auto_preprocess");
  if (autoChk && autoChk.checked) recPre = "auto";
  try{
    const modelVal = CFG.rec_model || "";
    const url = `/gui/reid/compat?rec_model=${encodeURIComponent(modelVal)}&rec_preprocess=${recPre}`;
    const info = await apiGet(url);
    const h = el("rec_preprocess_hint");
    if(h && autoChk && autoChk.checked){
        h.textContent=`(Sys: ${info.recommended_preprocess})`;
        h.style.color="var(--ok)";
    }
    const compat = info.compatible_db_paths || [];
    const dbSel = el("cfg_rec_db");
    if(dbSel && compat.length){
      const curr = dbSel.value;
      dbSel.innerHTML = "";
      compat.forEach(o=>{
          const op = document.createElement("option");
          op.value = o; op.textContent = o; dbSel.appendChild(op);
      });
      if(!compat.includes(curr)){
          CFG.rec_db = compat[0]; dbSel.value = CFG.rec_db;
          if(showToast) toast("DB auto-selected", false);
      } else { dbSel.value = curr; }
    }
    await checkReidCompat();
  }catch(e){ console.warn("Compat check failed", e); }
}

async function refreshFaceDbStats() {
    const elStats = el("faceDbStats");
    if (!elStats) return;

    try {
        const s = await apiGet("/gui/facedb/stats");
        if (!s.exists) {
            elStats.innerHTML = `<span style="color:var(--bad)">Keine Datenbank gefunden.</span>`;
            return;
        }


        const lblSize = CUR_LANG === 'de' ? "Größe:" : "Size:";
        const lblDate = CUR_LANG === 'de' ? "Update:" : "Last mod:";
        const lblVec  = CUR_LANG === 'de' ? "Vektoren (Live):" : "Vectors (Live):";

        let html = `<span><b>${lblDate}</b> ${s.mtime}</span> <span class="sep">•</span> <span><b>${lblSize}</b> ${s.size_mb} MB</span>`;

        if (s.loaded) {
            html += ` <span class="sep">•</span> <span style="color:var(--ok)"><b>${lblVec}</b> ${s.count}</span>`;
        } else {

            html += ` <span class="sep">•</span> <span class="muted">(Offline)</span>`;
        }

        elStats.innerHTML = html;
    } catch(e) {
        console.warn("DB Stats failed", e);
    }
}

async function loadConfig(){
  try {
      const data = await apiGet("/gui/state"); CFG = data.config;
      SCHEMA = data.schema; ORIG_CFG = JSON.parse(JSON.stringify(CFG));
      renderParams();
      await reidCompatUpdate(false);
  } catch(e){}
}

async function saveAndApply() {

  const ok = await checkPortNow();
  if (!ok) {
    toast("Port ist belegt/ungültig – bitte anderen Port wählen.", true);
    return;
  }

  try {
    await apiPost("/gui/state", { config: CFG, action: "apply" });
    setTimeout(() => location.reload(), 5000);
  } catch (e) {}
}

async function startPipeline(){ await apiPost("/gui/pipeline", {enabled:true}); }
async function stopPipeline(){ await apiPost("/gui/pipeline", {enabled:false}); }

async function reloadModels(){
    try {
        const data = await apiGet("/gui/models");
        MODEL_CATALOG = data.catalog || [];
        const activeFiles = [];
        if (data.yolo_model && data.yolo_exists) activeFiles.push(data.yolo_model);
        if (data.face_det_model && data.face_det_exists) activeFiles.push(data.face_det_model);
        if (data.reid_model && data.reid_exists) activeFiles.push(data.reid_model);
        renderModelTable(activeFiles);
        renderModelSummary(data);
        loadRuntimeModels();
        if(data.storage) {
            const s = data.storage;
            const infoText = tr("storage_info", {size: s.size_mb, count: s.file_count});
            const infoEl = el("modelStorageInfo");
            if(infoEl) infoEl.textContent = infoText;
        }
    } catch(e){ console.error("Update Models failed", e); }
}

async function refreshModelLog() {
  try {
    const data = await apiGet("/gui/model/log");
    const lines = (data && data.lines) ? data.lines : [];
    handleWSMessage({ type: "model_log", lines });
  } catch (e) {}
}

async function clearModelLog(){ await apiPost("/gui/model/clear", {}); refreshModelLog(); }

function timeAgo(ts) {
  if (!ts || ts === 0) return "-";
  const now = Date.now() / 1000;
  const diff = now - ts;
  if (diff < 60) return tr("time_just_now");
  if (diff < 3600) return Math.floor(diff / 60) + " min";
  if (diff < 86400) return Math.floor(diff / 3600) + " h";
  return Math.floor(diff / 86400) + " d";
}

async function loadPeople(){
  try {
      const d = await apiGet("/gui/people/list");
      const zone = el("peopleDrops");
      if(!zone) return; zone.innerHTML = "";

      (d.people||[]).forEach(item => {
        const div = document.createElement("div"); div.className = "person-card";
        let imgHtml = `<div class="person-avatar" style="display:flex;align-items:center;justify-content:center;">?</div>`;
        if(item.thumb) imgHtml = `<img src="/gui/enroll/img/${encodeURIComponent(item.name)}/${encodeURIComponent(item.thumb)}" class="person-avatar">`;
        const lastSeenStr = item.last_seen ? timeAgo(item.last_seen) : "";
        const hitsLbl = tr("person_hits");
        const detInfo = item.detections > 0 ? `<span style="color:var(--acc)">${item.detections} ${esc(hitsLbl)}</span>` : `<span class="muted">0 ${esc(hitsLbl)}</span>`;
        let timeInfo = "";
        if (lastSeenStr) {
          if (CUR_LANG === "de") timeInfo = `<span class="muted" style="margin-left:6px; font-size:10px;">(${esc(tr("time_ago_prefix"))} ${esc(lastSeenStr)})</span>`;
          else timeInfo = `<span class="muted" style="margin-left:6px; font-size:10px;">(${esc(lastSeenStr)} ${esc(tr("time_ago_suffix"))})</span>`;
        }

        div.innerHTML = `
            ${imgHtml}
            <div class="person-info">
                <span class="person-name">${esc(item.name)}</span>
                <div style="font-size:11px; color:#9aa4b2; margin-top:2px;">
                    ${item.count} ${esc(tr("person_pics"))} • ${detInfo} ${timeInfo}
                </div>
            </div>
            <span class="del-btn" onclick="deletePerson(event, '${esc(item.name)}')">×</span>
        `;
        div.ondragover = (e) => { e.preventDefault(); div.classList.add("drag-hover"); };
        div.ondragleave = () => div.classList.remove("drag-hover");
        div.ondrop = (e) => handleDrop(e, item.name, div);
        div.onclick = (e) => { if (!e.target.classList.contains("del-btn")) openPersonModal(item.name); };
        zone.appendChild(div);
      });
  } catch(e) {}
}
async function deletePerson(e, name){ if(e)e.stopPropagation(); if(confirm(tr("confirm_delete_person", {name}))) { await apiPost("/gui/people/delete", {name}); loadPeople(); } }


async function addPerson(){
  const name = el("newPersonName").value;
  if(!name) return;

  const res = await apiPost("/gui/people/add", {name});
  if (res && res.ok) {
    el("newPersonName").value = "";
    setFaceDbDirty(true);
    loadPeople();
  }
}


async function loadUnknowns(){
    try {
        const d = await apiGet("/gui/unknown/list");
        CURRENT_UNKNOWNS = d.files || [];


        const sources = new Set();
        CURRENT_UNKNOWNS.forEach(item => {
            if(item.source) sources.add(item.source);
        });


        const sel = el("unknownFilter");
        if(sel) {
            const oldVal = sel.value;
            sel.innerHTML = "";
            const optAll = document.createElement("option");
            optAll.value = "ALL";
            optAll.textContent = tr("unknown_filter_all");
            sel.appendChild(optAll);

            if (sources.size > 0) {
                sel.style.display = "block";
                Array.from(sources).sort().forEach(s => {
                    const opt = document.createElement("option");
                    opt.value = s;
                    opt.textContent = s;
                    sel.appendChild(opt);
                });
                sel.value = oldVal;
            } else {
                sel.style.display = "none";
            }
        }


        filterUnknowns();

    } catch(e){ console.error(e); }
}

let UNKNOWN_DOM = new Map();
let UNKNOWN_LAST_FILTER = "ALL";
let UNKNOWN_IS_DRAGGING = false;

function _unknownFilterVal() {
  const sel = el("unknownFilter");
  return sel ? (sel.value || "ALL") : "ALL";
}

function _ufname(item) {
  return (item && item.file) ? item.file : item;
}
function _usrc(item) {
  return (item && item.source) ? item.source : "Unknown";
}

function _passesUnknownFilter(item, filterSrc) {
  if (!filterSrc || filterSrc === "ALL") return true;
  return _usrc(item) === filterSrc;
}

function _ensureTooltip() {
  if (OBF_TOOLTIP_EL) return OBF_TOOLTIP_EL;
  const t = document.createElement("div");
  t.id = "obfTooltip";
  t.style.display = "none";
  document.body.appendChild(t);
  OBF_TOOLTIP_EL = t;
  return t;
}

function _posTooltip(clientX, clientY) {
  const t = _ensureTooltip();
  const pad = 14;
  const margin = 8;


  let x = clientX + pad;
  let y = clientY + pad;


  const rect = t.getBoundingClientRect();


  if (x + rect.width > window.innerWidth - margin) {
    x = clientX - pad - rect.width;
  }

  if (y + rect.height > window.innerHeight - margin) {
    y = clientY - pad - rect.height;
  }

  x = Math.max(margin, Math.min(x, window.innerWidth - margin - rect.width));
  y = Math.max(margin, Math.min(y, window.innerHeight - margin - rect.height));
  t.style.left = `${Math.round(x)}px`;
  t.style.top = `${Math.round(y)}px`;
}

function tooltipShow(text, clientX, clientY) {
  const t = _ensureTooltip();
  t.textContent = text ?? "";
  t.style.display = "block";

  t.classList.add("show");
  _posTooltip(clientX, clientY);
}

function tooltipMove(clientX, clientY) {
  if (!OBF_TOOLTIP_EL || OBF_TOOLTIP_EL.style.display === "none") return;
  _posTooltip(clientX, clientY);
}

function tooltipHide() {
  if (!OBF_TOOLTIP_EL) return;
  OBF_TOOLTIP_EL.classList.remove("show");
  OBF_TOOLTIP_EL.style.display = "none";
}
function _prettyUnknownSuffix(suf) {
  if (!suf) return "—";
  const s = String(suf);

  if (s === "raw") return "Raw";
  if (s === "aligned" || s === "ali") return "Aligned";
  if (s === "norec") return "No ReID";

  if (s.startsWith("blurry_")) return `Blur: ${s.slice("blurry_".length)}`;
  if (s.startsWith("badqual_")) return `Quality: ${s.slice("badqual_".length)}`;

  return s.replaceAll("_", " ");
}

function _prettyUnknownSource(src) {
  if (!src) return "Unknown";
  let s = String(src).replaceAll("_", " ").trim();


  if (s.toLowerCase().startsWith("bi ")) s = "Blue Iris " + s.slice(3);

  return s || "Unknown";
}

function prettyUnknownTooltip(fname, srcFromApi) {
  const p = _parseUnknownFilename(fname);
  const srcRaw = (srcFromApi && String(srcFromApi).trim()) || p.sourceFromName || "Unknown";
  const src = _prettyUnknownSource(srcRaw);
  const typ = _prettyUnknownSuffix(p.suffix);
  const simStr = (typeof p.sim === "number") ? p.sim.toFixed(3) : "—";
  const ts = p.dtStr || "—";

  return `Zeit: ${ts}\nQuelle: ${src}\nTyp: ${typ}\nSim: ${simStr}\n\n${p.raw}`;
}


function _parseUnknownFilename(fname) {

  const out = { raw: String(fname || "") };
  let base = out.raw;
  base = base.replace(/\.(jpg|jpeg|png)$/i, "");
  let sim = null;
  const simIdx = base.lastIndexOf("_sim");
  if (simIdx >= 0) {
    sim = base.slice(simIdx + 4);
    base = base.slice(0, simIdx);
    const v = Number(sim);
    if (Number.isFinite(v)) out.sim = v;
  }

  const parts = base.split("_");

  if (parts.length >= 5 && parts[0].length === 8 && parts[1].length === 6 && parts[2].length >= 3) {
    const date = parts[0];
    const time = parts[1];
    const micro = parts[2];
    const y = date.slice(0, 4), mo = date.slice(4, 6), d = date.slice(6, 8);
    const hh = time.slice(0, 2), mm = time.slice(2, 4), ss = time.slice(4, 6);
    const ms = micro.slice(0, 3);
    out.dtStr = `${d}.${mo}.${y} ${hh}:${mm}:${ss}.${ms}`;
    out.suffix = parts[parts.length - 1];
    const srcTokens = parts.slice(3, -1);
    out.sourceFromName = srcTokens.join("_");
  }

  return out;
}



function _makeUnknownThumb(fname, src, filterSrc) {
  const div = document.createElement("div");
  div.className = "thumb";
  div.dataset.fname = fname;
  div.removeAttribute("title");
  const img = document.createElement("img");
  img.removeAttribute("title");
  img.alt = fname;


  const sanitizedUrl = `/gui/unknown/img/${encodeURIComponent(fname)}`;
  try {
    const url = new URL(sanitizedUrl, window.location.origin);
    if (url.origin === window.location.origin && url.pathname.startsWith("/gui/unknown/img/")) {
      img.src = url.href;
    } else {
      console.error("Invalid image URL blocked:", sanitizedUrl);
      img.src = "";
    }
  } catch (e) {
    console.error("URL validation failed:", e);
    img.src = "";
  }


  img.onclick = (e) => {
    e.preventDefault();
    const safeUrl = new URL(img.src, window.location.origin);
    if (safeUrl.origin === window.location.origin) {
      window.open(safeUrl.href, "_blank", "noopener,noreferrer");
    }
  };

  div.appendChild(img);


  if (filterSrc === "ALL") {
    const lbl = document.createElement("div");
    lbl.textContent = src;
    lbl.style.position = "absolute";
    lbl.style.bottom = "0";
    lbl.style.left = "0";
    lbl.style.right = "0";
    lbl.style.background = "rgba(0,0,0,0.6)";
    lbl.style.color = "#fff";
    lbl.style.fontSize = "9px";
    lbl.style.padding = "2px";
    lbl.style.textAlign = "center";
    lbl.style.whiteSpace = "nowrap";
    lbl.style.overflow = "hidden";
    lbl.style.textOverflow = "ellipsis";
    lbl.style.pointerEvents = "none";
    div.appendChild(lbl);
    div.style.position = "relative";
  }


div.addEventListener("mouseenter", (e) => {
  if (typeof UNKNOWN_IS_DRAGGING !== "undefined" && UNKNOWN_IS_DRAGGING) return;
  tooltipShow(prettyUnknownTooltip(fname, src), e.clientX, e.clientY);

});
div.addEventListener("mousemove", (e) => tooltipMove(e.clientX, e.clientY));
div.addEventListener("mouseleave", () => tooltipHide());



  div.draggable = true;
  div.ondragstart = (e) => {
    tooltipHide();
    UNKNOWN_IS_DRAGGING = true;
    e.dataTransfer.setData("text", fname);
  };
  div.ondragend = () => {
    UNKNOWN_IS_DRAGGING = false;
  };

  return div;
}



function rebuildUnknownZone() {
  const zone = el("unknownZone");
  if (!zone) return;

  const filterSrc = _unknownFilterVal();
  UNKNOWN_LAST_FILTER = filterSrc;
  zone.innerHTML = "";
  UNKNOWN_DOM.clear();

  (CURRENT_UNKNOWNS || []).forEach((it) => {
    if (!_passesUnknownFilter(it, filterSrc)) return;
    const fname = _ufname(it);
    const src = _usrc(it);
    const node = _makeUnknownThumb(fname, src, filterSrc);
    UNKNOWN_DOM.set(fname, node);
    zone.appendChild(node);
  });
}

function applyUnknownDelta(files) {
  const zone = el("unknownZone");
  if (!zone) return;
  if (UNKNOWN_IS_DRAGGING) return;
  const filterSrc = _unknownFilterVal();
  if (filterSrc !== UNKNOWN_LAST_FILTER) {
    rebuildUnknownZone();
    return;
  }

  const list = Array.isArray(files) ? files : [];
  const filtered = list.filter((it) => _passesUnknownFilter(it, filterSrc));
  const want = new Set(filtered.map(_ufname));


  for (const [fname, node] of UNKNOWN_DOM.entries()) {
    if (!want.has(fname)) {
      try { node.remove(); } catch (e) {}
      UNKNOWN_DOM.delete(fname);
    }
  }


  let firstExistingIdx = -1;
  for (let i = 0; i < filtered.length; i++) {
    const fname = _ufname(filtered[i]);
    if (UNKNOWN_DOM.has(fname)) { firstExistingIdx = i; break; }
  }
  if (firstExistingIdx === -1) firstExistingIdx = filtered.length;

  const newOnes = filtered.slice(0, firstExistingIdx);
  for (let i = newOnes.length - 1; i >= 0; i--) {
    const it = newOnes[i];
    const fname = _ufname(it);
    if (UNKNOWN_DOM.has(fname)) continue;

    const src = _usrc(it);
    const node = _makeUnknownThumb(fname, src, filterSrc);
    UNKNOWN_DOM.set(fname, node);
    zone.insertBefore(node, zone.firstChild);
  }
}

function filterUnknowns() {

  applyUnknownDelta(CURRENT_UNKNOWNS);
}


let UNKNOWN_AUTO_TIMER = null;
let UNKNOWN_LAST_KEY = "";
let UNKNOWN_LAST_AT = 0;
let UNKNOWN_WS_THROTTLE = 0;

function _dashActive() {
  const t = document.getElementById("tab-dash");

  return !t || t.classList.contains("active");
}

function _unknownKey(files) {
  if (!Array.isArray(files) || files.length === 0) return "0";
  const first = (files[0] && (files[0].file || files[0])) || "";
  return `${files.length}:${first}`;
}

async function refreshUnknownsIfChanged(force = false) {
  try {
    if (document.hidden) return;
    if (!_dashActive()) return;

    const d = await apiGet("/gui/unknown/list");
    const files = d.files || [];
    const key = _unknownKey(files);
    const now = Date.now();
    if (!force && key === UNKNOWN_LAST_KEY) return;

    UNKNOWN_LAST_KEY = key;
    UNKNOWN_LAST_AT = now;
    CURRENT_UNKNOWNS = files;

    const sources = new Set();
    CURRENT_UNKNOWNS.forEach((it) => {
      const src = (it && it.source) ? it.source : null;
      if (src) sources.add(src);
    });

    const sel = el("unknownFilter");
    if (sel) {
      const oldVal = sel.value || "ALL";
      sel.innerHTML = "";

      const optAll = document.createElement("option");
      optAll.value = "ALL";
      optAll.textContent = (typeof tr === "function") ? tr("unknown_filter_all") : "All Sources";
      sel.appendChild(optAll);

      if (sources.size > 0) {
        sel.style.display = "block";
        Array.from(sources).sort().forEach((s) => {
          const opt = document.createElement("option");
          opt.value = s;
          opt.textContent = s;
          sel.appendChild(opt);
        });


        const hasOld = Array.from(sel.options).some(o => o.value === oldVal);
        sel.value = hasOld ? oldVal : "ALL";
      } else {
        sel.style.display = "none";
      }
    }

    filterUnknowns();
  } catch (e) {

  }
}

function scheduleUnknownRefresh() {
  const now = Date.now();

  if (now - UNKNOWN_WS_THROTTLE < 1200) return;
  UNKNOWN_WS_THROTTLE = now;


  setTimeout(() => refreshUnknownsIfChanged(false), 200);
}

function startUnknownAutoRefresh(ms = 8000) {
  stopUnknownAutoRefresh();
  UNKNOWN_AUTO_TIMER = setInterval(() => refreshUnknownsIfChanged(false), ms);
  refreshUnknownsIfChanged(true);
}

function stopUnknownAutoRefresh() {
  if (UNKNOWN_AUTO_TIMER) {
    clearInterval(UNKNOWN_AUTO_TIMER);
    UNKNOWN_AUTO_TIMER = null;
  }
}


document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refreshUnknownsIfChanged(true);
});


function updateBuildDbButton() {
  const btn = document.getElementById("btnBuildDb")
           || document.querySelector('button[onclick="startEnrollBuild()"]');
  if (!btn) return;

  if (FACE_DB_DIRTY) btn.classList.add("build-dirty");
  else btn.classList.remove("build-dirty");
}

function setFaceDbDirty(on = true) {
  FACE_DB_DIRTY = !!on;
  localStorage.setItem("obf_db_dirty", FACE_DB_DIRTY ? "1" : "0");
  updateBuildDbButton();
}

async function reloadUnknown(){ await loadUnknowns(); }
async function clearUnknowns(){ if(confirm(tr("confirm_delete_all_unknown"))) { await apiPost("/gui/unknown/clear", {}); reloadUnknown(); } }


async function handleDrop(e, person, div){
  e.preventDefault();
  div.classList.remove("drag-hover");
  const f = e.dataTransfer.getData("text");
  if(!f) return;

  const res = await apiPost("/gui/unknown/assign", {file:f, person});
  if (res && res.ok) {
    setFaceDbDirty(true);
  }

  reloadUnknown();
  loadPeople();
}

let CUR_MODAL = null;

async function openPersonModal(name) { CUR_MODAL = name; el("pmTitle").textContent = name; el("pmGrid").innerHTML = esc(tr("loading")); el("personModal").style.display = "flex"; await loadPersonImages(name); }
function closePersonModal() { el("personModal").style.display = "none"; CUR_MODAL = null; loadPeople(); }
async function loadPersonImages(name) {
    try {
        const res = await apiGet(`/gui/people/files/${encodeURIComponent(name)}`);
        const grid = el("pmGrid"); grid.innerHTML = "";
        (res.files||[]).forEach(f => {
            const div = document.createElement("div"); div.className = "p-img-card";
            div.innerHTML = `<img src="/gui/enroll/img/${encodeURIComponent(name)}/${encodeURIComponent(f)}"><div class="p-img-del" onclick="delSingle('${esc(name)}','${esc(f)}')">×</div>`;
            grid.appendChild(div);
        });
    } catch(e){ el("pmGrid").textContent = tr("toast_error"); }
}
async function delSingle(person, file) { if(confirm(tr("confirm_delete_image"))) { await apiPost("/gui/people/file/delete", {person, file}); loadPersonImages(person); } }

function enrollLooksDone(lines) {
  if (!Array.isArray(lines) || !lines.length) return false;
  const last = String(lines[lines.length - 1] || "");
  return /FERTIG|SUCCESS|Job erfolgreich|Job.*beendet|DONE|COMPLETED/i.test(last);
}


async function pollEnroll(){
  try{
    const log = await apiGet("/gui/enroll/log");
    const lines = (log.lines || []);


    const box = el("enrollLog");
    if (box && box.textContent !== lines.join("\n")) {
      box.textContent = lines.join("\n");
      box.scrollTop = box.scrollHeight;
    }


if (enrollLooksDone(log.lines)) {
  setFaceDbDirty(false);
  refreshFaceDbStats();
}
  } catch(e) {}
}


async function startEnrollBuild(){
  if(!CFG) await loadConfig();
  await apiPost("/gui/enroll/start", { kind:"build", config: CFG });

  pollEnroll();
  setTimeout(pollEnroll, 1500);
  setTimeout(pollEnroll, 3500);
}

async function clearEnrollLog(){ await apiPost("/gui/enroll/clear", {}); pollEnroll(); }

function fitText(ctx, text, maxW) {
  if (ctx.measureText(text).width <= maxW) return text;
  if (maxW <= ctx.measureText("…").width) return "…";
  let t = text;
  while (t.length > 0 && ctx.measureText(t + "…").width > maxW) t = t.slice(0, -1);
  return t.length ? (t + "…") : "…";
}

function rectsOverlap(a, b) {
  return !(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y);
}

function placeNonOverlappingRect(x, y, w, h, occupied, canvasW, canvasH) {
  x = Math.max(0, Math.min(x, canvasW - w));
  y = Math.max(0, Math.min(y, canvasH - h));

  for (let tries = 0; tries < 80; tries++) {
    const r = { x, y, w, h };
    let hit = false;
    for (const o of occupied) {
      if (rectsOverlap(r, o)) { hit = true; break; }
    }
    if (!hit) return { x, y };
    y += h + 2;
    if (y + h > canvasH) break;
  }
  return null;
}

function drawLabelBadgeSmart(ctx, x1, y1, x2, y2, col, shown, conf, occupied, canvasW, canvasH) {
  ctx.save();

  const fontPx = 11;
  const lineH  = 12;
  const pad    = 3;

  ctx.font = `${fontPx}px system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif`;
  ctx.textBaseline = "top";
  ctx.textAlign = "left";

  const confTxt = `${Math.round((conf || 0) * 100)}%`;
  const oneLine = `${shown} ${confTxt}`;
  const maxBadgeW = Math.min(260, canvasW - 4);
  const maxTextW  = Math.max(1, maxBadgeW - pad * 2);

  let lines = [];
  if (ctx.measureText(oneLine).width <= maxTextW) {
    lines = [oneLine];
  } else {
    const nameLine = fitText(ctx, String(shown || ""), maxTextW);
    const confLine = fitText(ctx, confTxt, maxTextW);
    lines = [nameLine, confLine];
  }

  const textWidths = lines.map(t => ctx.measureText(t).width);
  const badgeW = Math.min(maxBadgeW, Math.max(...textWidths) + pad * 2);
  const badgeH = lines.length * lineH + pad * 2;


  const candidates = [
    { x: x1,           y: y1 - badgeH - 2 },
    { x: x2 - badgeW,  y: y1 - badgeH - 2 },
    { x: x1,           y: y1 + 2 },
    { x: x2 - badgeW,  y: y1 + 2 },
    { x: x1,           y: y2 + 2 },
    { x: x2 - badgeW,  y: y2 + 2 },
  ];

  let placed = null;
  for (const c of candidates) {
    if (c.y < 0 || c.y + badgeH > canvasH) continue;
    placed = placeNonOverlappingRect(c.x, c.y, badgeW, badgeH, occupied, canvasW, canvasH);
    if (placed) break;
  }

  if (!placed) {
    placed = {
      x: Math.max(0, Math.min(x1, canvasW - badgeW)),
      y: Math.max(0, Math.min(y1 + 2, canvasH - badgeH)),
    };
  }


  ctx.fillStyle = col;
  ctx.fillRect(placed.x, placed.y, badgeW, badgeH);


  ctx.fillStyle = "#000";
  for (let i = 0; i < lines.length; i++) {
    ctx.fillText(lines[i], placed.x + pad, placed.y + pad + i * lineH);
  }

  occupied.push({ x: placed.x, y: placed.y, w: badgeW, h: badgeH });

  ctx.restore();
}



async function runTest(){
  const f = el("testFile").files[0]; if(!f)return;
  const cvs = el("testCanvas");
  const ctx = cvs.getContext("2d");

  cvs.style.display="none";
  el("testOut").textContent="Processing...";
  const img = new Image();
  img.onload = async () => {

  const parentW = (cvs.parentElement && cvs.parentElement.clientWidth) ? cvs.parentElement.clientWidth : img.width;
  const cssW = Math.min(img.width, parentW);
  const cssH = Math.round(cssW * (img.height / img.width));

  cvs.style.width = cssW + "px";
  cvs.style.height = cssH + "px";
  cvs.style.display = "block";

  const MAX_CANVAS_DIM = 4096;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const canvasW = Math.min(MAX_CANVAS_DIM, Math.round(cssW * dpr));
  const canvasH = Math.min(MAX_CANVAS_DIM, Math.round(cssH * dpr));
  cvs.width = canvasW;
  cvs.height = canvasH;


  if (canvasW === MAX_CANVAS_DIM || canvasH === MAX_CANVAS_DIM) {
      console.warn(`Canvas size clamped to ${MAX_CANVAS_DIM}px (original: ${Math.round(cssW * dpr)}x${Math.round(cssH * dpr)})`);
  }

  const ctx = cvs.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.drawImage(img, 0, 0, cssW, cssH);
  const sx = cssW / img.width;
  const sy = cssH / img.height;
  const fd = new FormData();
  fd.append("image", f);
  fd.append("min_confidence", el("testMinConf").value);

  const j = await (await fetch("/v1/vision/detection", { method: "POST", body: fd })).json();
  el("testOut").textContent = JSON.stringify(j, null, 2);

  const occupied = [];

  const preds = (j.predictions || []).slice().sort((a, b) =>
    (a.y_min - b.y_min) || (a.x_min - b.x_min)
  );


  const items = preds.map(p => {
    const lblRaw = String(p.label ?? "");
    const lbl = lblRaw.trim().toLowerCase();
    const uid = String(p.userid ?? "").trim();
    const shown = uid ? uid : lblRaw;

    let col = "#3b82f6";
    if (lbl === "person" || lbl.startsWith("person")) col = "#facc15";
    else if (lbl === "face" || lbl.startsWith("face") || lbl === "unknown" || lbl.startsWith("unknown") || uid) col = "#ef4444";

    return { p, col, shown, conf: (p.confidence || 0) };
  });


  items.forEach(({ p, col }) => {
    const x1 = p.x_min * sx, y1 = p.y_min * sy, x2 = p.x_max * sx, y2 = p.y_max * sy;
    const w = x2 - x1, h = y2 - y1;

    ctx.save();
    ctx.strokeStyle = col;
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, w, h);
    ctx.restore();
  });


  items.forEach(({ p, col, shown, conf }) => {
    const x1 = p.x_min * sx, y1 = p.y_min * sy, x2 = p.x_max * sx, y2 = p.y_max * sy;

    drawLabelBadgeSmart(
      ctx,
      x1, y1, x2, y2,
      col,
      shown,
      conf,
      occupied,
      cssW,
      cssH
    );
  });
};
img.src = URL.createObjectURL(f);

}


function toggleStream() {
    const urlInput = el("rtspUrl");
    const minConf = el("testMinConf").value || 0.40;
    const img = el("streamDisplay");
    const canvas = el("testCanvas");
    const btn = el("btnStream");

    if (!urlInput || !img || !btn) return;

    if (!isStreaming) {

        const rtspUrl = urlInput.value.trim();
        if (!rtspUrl) {
            toast(tr("toast_rtsp_missing"), true);
            return;
        }

        canvas.style.display = "none";
        img.style.display = "block";
        img.src = `/v1/vision/stream_test?url=${encodeURIComponent(rtspUrl)}&min_conf=${minConf}&t=${Date.now()}`;
        btn.textContent = tr("btn_stop_stream");
        btn.classList.remove("ok");
        btn.classList.add("bad");
        isStreaming = true;
        urlInput.disabled = true;

    } else {

        img.src = "";
        img.style.display = "none";
        btn.textContent = tr("btn_start_stream");
        btn.classList.remove("bad");
        btn.classList.add("ok");
        isStreaming = false;
        urlInput.disabled = false;
    }
}

async function runVideoTest(){
  const f = el("videoFile")?.files?.[0];
  if(!f){
    toast(tr("toast_video_choose_first"), true);
    return;
  }

  const btn = el("btnVideo");
  const btnPause = el("btnVideoPause");
  const img = el("streamDisplay");
  const cvs = el("testCanvas");


  if (btn && btn.dataset && btn.dataset.mode === "playing") {
    stopVideoTest();
    return;
  }

  if(btn){ btn.disabled = true; btn.textContent = "⏳"; }
  if(btnPause){ btnPause.disabled = true; btnPause.textContent = tr("btn_pause"); btnPause.dataset.mode = ""; }

  try{
    const fd = new FormData();
    fd.append("file", f, f.name);

    const r = await fetch("/v1/vision/video/upload", { method:"POST", body: fd });
    const j = await r.json().catch(()=> ({}));

    if(!r.ok || !j.success){
      const err = j.error || (CUR_LANG === "de" ? `HTTP Fehler ${r.status}` : `HTTP error ${r.status}`);
      throw new Error(err);
    }

    const token = j.token;
    const minConf = parseFloat(el("testMinConf")?.value || "0.40");

    window.__videoTest = { token, minConf };


    if(cvs) cvs.style.display = "none";
    if(img){
      img.style.display = "block";
      img.style.visibility = "visible";
      img.src = `/v1/vision/video/stream?token=${encodeURIComponent(token)}&min_conf=${encodeURIComponent(minConf)}&t=${Date.now()}`;
    }

    if(btn){
      btn.dataset.mode = "playing";
      btn.textContent = tr("btn_stop_stream");
    }

    if(btnPause){
      btnPause.disabled = false;
      btnPause.textContent = tr("btn_pause");
      btnPause.dataset.mode = "live";
    }

    toast(tr("toast_video_started"), false);
  }catch(e){
    console.error("Video upload failed:", e);
    toast(tr("toast_video_upload_failed_prefix") + (e?.message || e), true);

    if(btn){ btn.dataset.mode = ""; btn.textContent = tr("btn_upload_play"); }
    if(btnPause){ btnPause.disabled = true; btnPause.textContent = tr("btn_pause"); btnPause.dataset.mode = ""; }
    if(img){ img.src = ""; }
  }finally{
    if(btn) btn.disabled = false;
  }
}


function stopVideoTest(){
  const btn = el("btnVideo");
  const btnPause = el("btnVideoPause");
  const img = el("streamDisplay");
  const cvs = el("testCanvas");

  if(img){
    img.src = "";
    img.style.display = "none";
    img.style.visibility = "visible";
  }


  if(cvs){
    cvs.style.display = "none";
    const ctx = cvs.getContext("2d");
    if(ctx) ctx.clearRect(0, 0, cvs.width, cvs.height);
  }


  window.__videoTest = null;

  if(btn){
    btn.dataset.mode = "";
    btn.textContent = tr("btn_upload_play");
    btn.disabled = false;
  }

  if(btnPause){
    btnPause.disabled = true;
    btnPause.textContent = tr("btn_pause");
    btnPause.dataset.mode = "";
  }

  toast(tr("toast_video_stopped"), false);
}

function pauseVideoTest(){
  const btn = el("btnVideo");
  const btnPause = el("btnVideoPause");
  const img = el("streamDisplay");
  const cvs = el("testCanvas");

  if(!btn || !btn.dataset || btn.dataset.mode !== "playing"){
    return;
  }
  if(!img || !cvs) return;

  const mode = btnPause?.dataset?.mode || "live";


  if(mode === "live"){
    try{

      const w = img.naturalWidth || img.width || 1280;
      const h = img.naturalHeight || img.height || 720;
      cvs.width = w;
      cvs.height = h;

      const ctx = cvs.getContext("2d");
      if(ctx){
        ctx.drawImage(img, 0, 0, w, h);
      }
      cvs.style.display = "block";


      img.style.visibility = "hidden";

      if(btnPause){
        btnPause.textContent = tr("btn_resume");
        btnPause.dataset.mode = "paused";
      }
      toast(tr("toast_video_paused"), false);
    }catch(e){
      console.error("Pause snapshot failed:", e);
      toast(tr("toast_pause_failed"), true);
    }
    return;
  }


  if(mode === "paused"){
    cvs.style.display = "none";
    img.style.visibility = "visible";

    if(btnPause){
      btnPause.textContent = tr("btn_pause");
      btnPause.dataset.mode = "live";
    }
    toast(tr("toast_video_resumed"), false);
  }
}


async function loadUnknownClusters() {

    const sim = parseFloat((el("uc_sim")?.value) || "0.75");
    const minsz = parseInt((el("uc_minsz")?.value) || "2", 10);
    const limit = parseInt((el("uc_limit")?.value) || "2000", 10);
    const grid = el("uc_grid");
    const stats = el("uc_stats");

    if (!grid || !stats) return;

    grid.innerHTML = "";
    stats.textContent = tr("loading");


    let people = [];
    try {
        const ppl = await apiGet("/gui/people/list");
        people = (ppl.people || []).map(p => p.name).filter(Boolean);
    } catch (e) {
        console.warn("Cluster: Konnte Personenliste nicht laden", e);
    }


    try {
        const d = await apiGet(`/gui/unknown/cluster?sim_thres=${sim}&min_cluster_size=${minsz}&limit=${limit}`);
        const clusters = d.clusters || [];
        const miss = (d.missing_embeddings || []).length;

        stats.textContent = `clusters=${clusters.length} items_with_emb=${d.count_items || 0} missing_emb=${miss}`;


        for (const c of clusters) {
            const card = document.createElement("div");
            card.className = "card";
            card.style.padding = "10px";


            const img = document.createElement("img");
            img.src = "/gui/unknown/img/" + encodeURIComponent(c.rep_file);
            img.style.width = "100%";
            img.style.aspectRatio = "1/1";
            img.style.objectFit = "cover";
            img.style.borderRadius = "10px";
            img.loading = "lazy";
            card.appendChild(img);


            const meta = document.createElement("div");
            meta.className = "row";
            meta.style.justifyContent = "space-between";
            meta.style.marginTop = "8px";

            meta.innerHTML = `<span class="pill">src: <b>${esc(c.source || "")}</b></span><span class="pill">count: <b>${c.count}</b></span>`;
            card.appendChild(meta);


            const actions = document.createElement("div");
            actions.className = "row";
            actions.style.gap = "8px";
            actions.style.marginTop = "10px";
            actions.style.flexWrap = "wrap";


            const sel = document.createElement("select");
            sel.style.minWidth = "140px";
            const o0 = document.createElement("option");
            o0.value = "";
            o0.textContent = tr("uc_assign_to");
            sel.appendChild(o0);

            for (const p of people) {
                const o = document.createElement("option");
                o.value = p;
                o.textContent = p;
                sel.appendChild(o);
            }
            actions.appendChild(sel);


            const btn = document.createElement("button");
            btn.className = "btn ok";
            btn.textContent = tr("uc_assign");
            btn.onclick = async () => {
                const person = sel.value;
                if (!person) return toast(tr("uc_choose_person"), true);
                if (!confirm(tr("uc_confirm_move", {count: c.files.length, person}))) return;

                btn.disabled = true;
                try {
                    await apiPost("/gui/unknown/assign_many", { person, files: c.files });
                    toast(tr("uc_move_success", {person}), false);
                    setFaceDbDirty(true);
                    await loadUnknownClusters();
                } catch (e) {
                    toast(tr("toast_error") + ": " + (e.message || e), true);
                } finally {
                    btn.disabled = false;
                }
            };
            actions.appendChild(btn);


            const toggle = document.createElement("button");
            toggle.className = "btn";
            toggle.textContent = tr("uc_show_files");


            const filesWrap = document.createElement("div");
            filesWrap.style.display = "none";
            filesWrap.style.gap = "6px";
            filesWrap.style.flexWrap = "wrap";
            filesWrap.style.marginTop = "10px";


            for (const f of (c.files || []).slice(0, 36)) {
                const mi = document.createElement("img");
                mi.src = "/gui/unknown/img/" + encodeURIComponent(f);
                mi.style.width = "56px";
                mi.style.height = "56px";
                mi.style.objectFit = "cover";
                mi.style.borderRadius = "8px";
                mi.style.border = "1px solid rgba(255,255,255,0.10)";
                mi.loading = "lazy";
                mi.title = prettyUnknownTooltip(f, null);
                mi.onclick = () => {
                    const url = new URL(mi.src, window.location.origin);
                    if (url.origin === window.location.origin) {
                        window.open(url.href, "_blank");
                    }
                };
                filesWrap.appendChild(mi);
            }

            toggle.onclick = () => {
                const open = filesWrap.style.display === "flex";
                filesWrap.style.display = open ? "none" : "flex";
                toggle.textContent = open ? tr("uc_show_files") : tr("uc_hide_files");
            };
            actions.appendChild(toggle);

            card.appendChild(actions);
            card.appendChild(filesWrap);
            grid.appendChild(card);
        }
    } catch (e) {
        console.error(e);
        stats.textContent = tr("cluster_error_loading");
        toast(tr("api_error_prefix") + e.message, true);
    }
}

function modelLooksDone(lines) {
  if (!Array.isArray(lines) || !lines.length) return false;

  const tail = lines.slice(Math.max(0, lines.length - 10)).join("\n");

  if (/CRITICAL ERROR|Traceback|Exception/i.test(tail)) return false;

  return /Setup\s*beendet|Download\s*finished|Please\s*Restart|Konvertierung\s*erfolgreich|export\s*success|Conversion\s*successful|DONE|COMPLETED/i.test(tail);
}

function setRestartNeeded(on) {
  const btn = el("btnRestartModels");
  if (btn) {
    btn.classList.toggle("restart-blink", !!on);
    btn.title = !!on ? tr("restart_recommended") : tr("btn_restart");
  }
  try {
    if (on) localStorage.setItem("obf_restart_needed", "1");
    else localStorage.removeItem("obf_restart_needed");
  } catch (e) {}
}

function restoreRestartNeeded() {
  try {
    const v = localStorage.getItem("obf_restart_needed");
    if (v === "1") setRestartNeeded(true);
  } catch (e) {}
}

window._MODEL_INSTALL_WATCH = null;

function armModelInstallWatch(selectedIds) {
  const before = new Set();
  (MODEL_CATALOG || []).forEach(m => { if (m && m.exists) before.add(m.id); });

  window._MODEL_INSTALL_WATCH = {
    armed: true,
    before,
    wanted: Array.isArray(selectedIds) ? selectedIds.slice() : [],
    startedAt: Date.now(),
  };
}

function finalizeModelInstallWatch() {
  const w = window._MODEL_INSTALL_WATCH;
  if (!w || !w.armed) return;
  w.armed = false;

  reloadModels().then(() => {
    const after = new Set();
    (MODEL_CATALOG || []).forEach(m => { if (m && m.exists) after.add(m.id); });

    const newlyInstalled = [];
    after.forEach(id => { if (!w.before.has(id)) newlyInstalled.push(id); });

    const wanted = new Set(w.wanted || []);
    const hit = newlyInstalled.some(id => wanted.size ? wanted.has(id) : true);
    if (hit) setRestartNeeded(true);
  }).catch(() => {});
}


window.hardRestart = async function(btnEl) {
  if(!confirm(tr("confirm_restart"))) return;

  setRestartNeeded(false);

  let btn = null;
  if (btnEl && btnEl.tagName) btn = btnEl;
  if (!btn) btn = document.getElementById("btnRestartModels") || document.getElementById("btnRestartServer");

  if (btn) { btn.textContent = "⏳"; btn.disabled = true; }
  try { await fetch("/gui/restart", { method: "POST" }); } catch (e) {}
  setTimeout(() => window.location.reload(), 5000);
};


(async function init(){
    setLanguage(CUR_LANG);
    restoreRestartNeeded();
    try { await loadConfig(); updateBenchDevices(); } catch(e){}
    try { await reloadModels(); } catch(e){}
    try { await refreshFaceDbStats(); } catch(e){}
    try { await loadPeople(); } catch(e){}
    updateBuildDbButton();
    try { await loadUnknowns(); } catch(e){}
    startUnknownAutoRefresh(8000);
    try { await refreshModelLog(); } catch(e){}


document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;

  const tag = (e.target && e.target.tagName) ? e.target.tagName.toLowerCase() : "";
  if (tag === "input" || tag === "textarea" || e.target?.isContentEditable) return;

  const modal = el("personModal");
  if (modal && modal.style.display !== "none") {
    closePersonModal();
  }
});

connectWS();
startRuntimeModelsAutoRefresh(2000);

const rbtn = document.getElementById("btnRestartModels") || document.getElementById("btnRestartServer");
if (rbtn) rbtn.onclick = (e) => window.hardRestart(e.currentTarget);
})();
