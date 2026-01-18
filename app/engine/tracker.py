from dataclasses import dataclass


@dataclass
class Track:
    id: int
    bbox: tuple[int, int, int, int]
    label: str
    conf: float
    lost: int = 0


class SimpleTracker:
    def __init__(self, iou_th: float = 0.5, max_lost: int = 10):
        self.iou_th = float(iou_th)
        self.max_lost = int(max_lost)
        self.tracks: dict[int, Track] = {}
        self.next_id = 1

    @staticmethod
    def iou(b1, b2) -> float:
        xA = max(b1[0], b2[0])
        yA = max(b1[1], b2[1])
        xB = min(b1[2], b2[2])
        yB = min(b1[3], b2[3])
        inter = max(0, xB - xA) * max(0, yB - yA)
        a1 = max(0, b1[2] - b1[0]) * max(0, b1[3] - b1[1])
        a2 = max(0, b2[2] - b2[0]) * max(0, b2[3] - b2[1])
        return inter / (a1 + a2 - inter + 1e-6)

    def update(self, dets: list[tuple[int, int, int, int, float, str]]) -> list[Track]:
        updated = set()

        for x1, y1, x2, y2, conf, label in dets:
            best_id, best_iou = None, 0.0
            for tid, trk in self.tracks.items():
                i = self.iou(trk.bbox, (x1, y1, x2, y2))
                if i > best_iou:
                    best_iou, best_id = i, tid

            if best_id is not None and best_iou >= self.iou_th:
                t = self.tracks[best_id]
                t.bbox = (int(x1), int(y1), int(x2), int(y2))
                t.conf = float(conf)
                t.label = str(label)
                t.lost = 0
                updated.add(best_id)
            else:
                tid = self.next_id
                self.tracks[tid] = Track(
                    tid,
                    (int(x1), int(y1), int(x2), int(y2)),
                    str(label),
                    float(conf),
                    0,
                )
                updated.add(tid)
                self.next_id += 1

        for tid in list(self.tracks.keys()):
            if tid not in updated:
                self.tracks[tid].lost += 1
                if self.tracks[tid].lost > self.max_lost:
                    del self.tracks[tid]

        return list(self.tracks.values())
