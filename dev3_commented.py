import sys, json, math, random
from pathlib import Path
import pygame
import numpy as np
from scipy.ndimage import uniform_filter1d

# ═══════════════════════════════════════════════════════════════
#  GLOBALE KONSTANTEN
# ═══════════════════════════════════════════════════════════════

# Fenstergröße in Pixeln und Ziel-Framerate
W, H, FPS = 1500, 1000, 60

# TB = Toolbar-Höhe (untere Leiste mit Buttons); das Spielfeld
# ist damit H-TB Pixel hoch (also 940 px)
TB = 60

# Wie viele vollständige Runden ein Spieler braucht, um zu gewinnen
RUNDEN_MAX = 3

# Pfad zur JSON-Datei, in der die gezeichnete Strecke gespeichert wird
SAVE = Path(__file__).parent / "strecke.json"

# ── Farben (RGB-Tupel) ────────────────────────────────────────
BG    = (22, 26, 35)    # Hintergrundfarbe (wird kaum direkt genutzt)
TB_C  = (18, 21, 30)    # Toolbar-Hintergrund
PANEL = (32, 37, 50)    # HUD-Panels / Knopf-Ruhezustand
RAND  = (55, 62, 80)    # Rahmen / Hover-Hintergrund
TEXT  = (210, 215, 228) # Normaler Fließtext
DIM   = (110, 118, 140) # Gedimmter/grauer Text (Hinweise)
WEISS   = (255, 255, 255)
SCHWARZ = (0, 0, 0)
ROT     = (220, 55, 55)
BLAU    = (50, 100, 220)
GRUEN   = (45, 185, 100)
ORANGE  = (230, 145, 35)
LILA    = (155, 65, 215)   # Checkpoint-Hauptfarbe
LILA2   = (100, 40, 145)   # Checkpoint-Streifenfarbe (dunkleres Lila)
GELB    = (255, 220, 40)
CYAN    = (45, 195, 210)
GRAS    = (68, 122, 50)    # Gras-Grundfarbe
GRAS_H  = (82, 145, 62)    # Hellere Gras-Flecken
GRAS_D  = (52, 98, 40)     # Dunklere Gras-Flecken

# ── Physik-Parameter ─────────────────────────────────────────
MAX_V   = 5.0   # Maximale Geschwindigkeit auf der Strecke (Pixel/Frame)
MAX_OFF = 1.8   # Maximale Geschwindigkeit im Offroad-Bereich
ACCEL   = 0.15  # Beschleunigung pro Frame beim Gasgeben
BREMSE  = 0.25  # Verzögerung pro Frame beim Bremsen/Rückwärtsfahren
REIB    = 0.04  # Reibung auf der Strecke (natürliches Abbremsen)
REIB_OFF= 0.12  # Reibung im Offroad-Bereich (bremst stärker)
LENKUNG = 2.2   # Maximaler Lenkwinkel pro Frame in Grad
SCHWELLE= 0.30  # Anteil der Auto-Punkte, die auf schwarzem Asphalt liegen
                # müssen, damit das Auto als "auf der Strecke" gilt (0–1)

# ── Maße (Pixel) ─────────────────────────────────────────────
AUTO_B, AUTO_H = 22, 38    # Breite und Höhe des Auto-Sprites
CP_B,   CP_H   = 120, 14   # Breite und Höhe eines Checkpoints
SZ_B,   SZ_H   = 140, 14   # Breite und Höhe der Start/Ziel-Linie
HANDLE_R        = 8         # Radius des Dreh-Handles (kleiner Kreis)
DICKE_MIN, DICKE_MAX, DICKE_S = 4, 60, 4  # Min/Max/Schrittgröße für Streckenbreite


# ── Spielmodus-Enum (als einfache Klasse mit Klassenvariablen) ─
class Modus:
    ZEICHNEN = 0  # Strecke mit Maus zeichnen
    START    = 1  # Start/Ziel-Linie platzieren und drehen
    CP       = 2  # Checkpoints setzen, verschieben, löschen
    AUTOS    = 3  # Autos auf der Strecke platzieren und ausrichten
    FAHREN   = 4  # Rennen läuft
    GEWONNEN = 5  # Jemand hat gewonnen – Overlay wird angezeigt


# ═══════════════════════════════════════════════════════════════
#  HILFSFUNKTIONEN
# ═══════════════════════════════════════════════════════════════

def glaette(pts):
    """
    Glättet eine Liste von (x,y)-Punkten mit einem gleitenden Durchschnitt.
    Wird genutzt, um die vom Spieler gezeichnete Strecke weicher zu machen.
    Die Endpunkte bleiben unverändert (mode='nearest' + manuelle Fixierung).
    Gibt die Punkte unverändert zurück, wenn es weniger als 4 sind.
    """
    if len(pts) < 4:
        return pts
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    # uniform_filter1d = gleitender Mittelwert mit Fenstergröße 25
    xg = uniform_filter1d(xs, 25, mode="nearest")
    yg = uniform_filter1d(ys, 25, mode="nearest")
    # Ersten und letzten Punkt einfrieren, damit die Linie nicht "wegdriftet"
    xg[0], yg[0] = xs[0], ys[0]
    xg[-1], yg[-1] = xs[-1], ys[-1]
    return [(int(x), int(y)) for x, y in zip(xg, yg)]


def rot_pt(px, py, cx, cy, deg):
    """
    Rotiert den Punkt (px, py) um den Mittelpunkt (cx, cy) um 'deg' Grad.
    Gibt den neuen (x, y) zurück. Wird genutzt, um die Ecken von
    rotierten Rechtecken (OBBs) zu berechnen.
    """
    r = math.radians(deg)
    dx, dy = px - cx, py - cy
    return (cx + dx * math.cos(r) - dy * math.sin(r),
            cy + dx * math.sin(r) + dy * math.cos(r))


def obb_ecken(cx, cy, b, h, deg):
    """
    Berechnet die 4 Ecken eines orientierten Rechtecks (OBB = Oriented Bounding Box).
    cx, cy = Mittelpunkt | b = Breite | h = Höhe | deg = Rotation in Grad.
    Rückgabe: Liste von 4 (x,y)-Tupeln in der Reihenfolge oben-links, oben-rechts,
    unten-rechts, unten-links.
    """
    hw, hh = b / 2, h / 2
    # Relative Ecken-Offsets (unrotiert), dann rotieren
    return [rot_pt(cx + ex, cy + ey, cx, cy, deg)
            for ex, ey in [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]]


def in_obb(px, py, cx, cy, b, h, deg):
    """
    Prüft, ob der Punkt (px, py) innerhalb des OBB liegt.
    Trick: Den Punkt in das lokale Koordinatensystem des Rechtecks
    transformieren (Rotation rückgängig machen) und dann gegen
    halbe Breite/Höhe testen – das ist schneller als Polygon-Tests.
    """
    r = math.radians(-deg)  # Gegenrotation
    dx, dy = px - cx, py - cy
    # Lokale x- und y-Koordinate im rotierten System
    lx = dx * math.cos(r) - dy * math.sin(r)
    ly = dx * math.sin(r) + dy * math.cos(r)
    return abs(lx) <= b / 2 and abs(ly) <= h / 2


def rect_trifft_obb(rect, cx, cy, b, h, deg):
    """
    Kollisionstest zwischen einem Pygame-Rect und einem OBB.
    Statt eines exakten Tests werden 9 charakteristische Punkte des
    Rects (Ecken, Kantenmittelpunkte, Zentrum) gegen das OBB geprüft.
    Gibt True zurück, sobald ein Punkt im OBB liegt.
    """
    pts = [rect.topleft, rect.topright, rect.bottomleft, rect.bottomright,
           rect.center, rect.midtop, rect.midbottom, rect.midleft, rect.midright]
    return any(in_obb(px, py, cx, cy, b, h, deg) for px, py in pts)


def streifen(surf, ecken, n, f1, f2):
    """
    Zeichnet abwechselnd farbige Querstreifen in ein rotiertes Rechteck.
    Wird für das Schachbrettmuster der Start/Ziel-Linie und die
    Streifen der Checkpoints verwendet.

    surf  = Ziel-Surface
    ecken = 4 Eckpunkte des Rechtecks (von obb_ecken)
    n     = Anzahl Streifen
    f1/f2 = abwechselnde Farben
    """
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n

        def ip(t, e=ecken):
            # Interpoliert linear zwischen den Kanten des Rechtecks,
            # um die linke und rechte Kante des i-ten Streifens zu finden
            p0 = (e[0][0] + t * (e[1][0] - e[0][0]),
                  e[0][1] + t * (e[1][1] - e[0][1]))
            p1 = (e[3][0] + t * (e[2][0] - e[3][0]),
                  e[3][1] + t * (e[2][1] - e[3][1]))
            return p0, p1

        a, b_ = ip(t0)
        c, d  = ip(t1)
        pygame.draw.polygon(surf, f1 if i % 2 == 0 else f2,
            [(int(a[0]), int(a[1])), (int(c[0]), int(c[1])),
             (int(d[0]), int(d[1])), (int(b_[0]), int(b_[1]))])


def gras_textur(w, h):
    """
    Erzeugt einmalig eine prozedurale Gras-Textur als pygame.Surface.
    Wird beim Start generiert und dann als Hintergrund verwendet.
    Seed 42 sorgt dafür, dass die Textur bei jedem Start identisch aussieht.
    """
    s = pygame.Surface((w, h))
    s.fill(GRAS)
    rng = random.Random(42)  # Fester Seed für reproduzierbare Textur

    # Runde Flecken in hell/dunkel für Tiefe
    for _ in range(3500):
        x = rng.randint(0, w - 1)
        y = rng.randint(0, h - 1)
        r = rng.randint(6, 45)
        farbe = GRAS_H if rng.random() > 0.5 else GRAS_D
        pygame.draw.circle(s, farbe, (x, y), r)

    # Kurze diagonale Linien als Grashalme
    for _ in range(7000):
        x, y = rng.randint(0, w - 1), rng.randint(0, h - 1)
        l   = rng.randint(3, 10)       # Länge des Halms
        ang = rng.uniform(-0.45, 0.45) # Leichte Neigung
        g   = rng.randint(115, 180)    # Zufälliger Grünton
        pygame.draw.line(s,
            (rng.randint(28, 55), g, rng.randint(28, 55)),
            (x, y),
            (x + int(math.sin(ang) * l), y - l),
            1)
    return s


# ═══════════════════════════════════════════════════════════════
#  KLASSE: Linie  (Basisklasse für Checkpoint und Start/Ziel)
# ═══════════════════════════════════════════════════════════════

class Linie:
    """
    Abstrakte Basisklasse für alle platzierbaren Linien auf der Strecke.
    Eine Linie ist ein rotiertes Rechteck, das der Spieler mit der Maus
    verschieben und drehen kann. Checkpoint und StartZiel erben von ihr.
    """

    def __init__(self, cx, cy, b, h, farbe, rand_f):
        """
        cx, cy  = Mittelpunkt der Linie (float, für Sub-Pixel-Genauigkeit)
        b, h    = Breite und Höhe des Rechtecks in Pixeln
        farbe   = Füllfarbe des Rechtecks
        rand_f  = Randfarbe wenn nicht ausgewählt
        """
        self.cx, self.cy   = float(cx), float(cy)
        self.b, self.h     = b, h
        self.farbe         = farbe
        self.rand_f        = rand_f
        self.winkel        = 0.0    # Aktuelle Rotation in Grad

        # Drag/Dreh-Zustand
        self.gedreht       = False  # True: Linie wird gerade gedreht
        self.gezogen       = False  # True: Linie wird gerade verschoben
        self.drag_off      = (0., 0.)  # Offset Maus→Mittelpunkt beim Ziehen
        self.dreh_maus0    = 0.     # Mauswinkel zu Beginn des Drehens
        self.dreh_obj0     = 0.     # Objektwinkel zu Beginn des Drehens

    def handle_pos(self):
        """
        Gibt die Position des Dreh-Handles zurück (kleiner Kreis rechts
        außen, mit dem man die Linie rotieren kann).
        """
        return rot_pt(self.cx + self.b / 2 + HANDLE_R + 4,
                      self.cy, self.cx, self.cy, self.winkel)

    def body_hit(self, mx, my):
        """True, wenn (mx, my) das Rechteck (leicht vergrößert) trifft."""
        return in_obb(mx, my, self.cx, self.cy, self.b, self.h + 14, self.winkel)

    def handle_hit(self, mx, my):
        """True, wenn (mx, my) in der Nähe des Dreh-Handles liegt."""
        hx, hy = self.handle_pos()
        return math.hypot(mx - hx, my - hy) <= HANDLE_R + 3

    def start_drehen(self, mx, my):
        """
        Beginnt den Dreh-Vorgang. Speichert den Start-Mauswinkel und
        den aktuellen Objekt-Winkel, um Delta-Rotation berechnen zu können.
        """
        self.gedreht    = True
        self.dreh_maus0 = math.degrees(math.atan2(my - self.cy, mx - self.cx))
        self.dreh_obj0  = self.winkel

    def start_ziehen(self, mx, my):
        """
        Beginnt den Verschiebe-Vorgang. Speichert den Offset zwischen
        Mausposition und Mittelpunkt, damit das Objekt nicht "springt".
        """
        self.gezogen  = True
        self.drag_off = (self.cx - mx, self.cy - my)

    def update(self, mx, my):
        """
        Aktualisiert Position/Rotation während Mausbewegung.
        Wird jeden Frame aufgerufen, solange gedreht oder gezogen wird.
        """
        if self.gedreht:
            # Neue Rotation = Differenz zwischen aktuellem und Start-Mauswinkel
            delta = math.degrees(math.atan2(my - self.cy, mx - self.cx)) - self.dreh_maus0
            self.winkel = (self.dreh_obj0 + delta) % 360
        elif self.gezogen:
            self.cx = mx + self.drag_off[0]
            self.cy = my + self.drag_off[1]

    def stop(self):
        """Beendet Drehen und Ziehen (beim Loslassen der Maustaste)."""
        self.gedreht = self.gezogen = False

    def kollidiert(self, rect):
        """True, wenn das Auto-Rect das Liniensrechteck trifft."""
        return rect_trifft_obb(rect, self.cx, self.cy, self.b, self.h, self.winkel)

    def zeichne(self, surf, schrift, sel=False, edit=False):
        """
        Zeichnet die Linie auf 'surf'.
        sel  = True: weißer Auswahlrahmen anzeigen
        edit = True: Dreh-Handle und Handle-Linie einzeichnen
        """
        ecken = obb_ecken(self.cx, self.cy, self.b, self.h, self.winkel)
        pts   = [(int(x), int(y)) for x, y in ecken]

        # Gefülltes Rechteck
        pygame.draw.polygon(surf, self.farbe, pts)
        # Rahmen: weiß wenn ausgewählt, sonst Standardrandfarbe
        pygame.draw.polygon(surf, WEISS if sel else self.rand_f, pts, 2 if sel else 1)

        # Unterklassen-spezifische Streifenmuster
        self._streifen(surf, ecken)

        # Unterklassen-spezifisches Label (z.B. Checkpoint-Nummer)
        lbl = self._label()
        if schrift and lbl:
            t = pygame.transform.rotate(schrift.render(lbl, True, WEISS), -self.winkel)
            surf.blit(t, t.get_rect(center=(int(self.cx), int(self.cy))))

        # Dreh-Handle einzeichnen wenn im Edit-Modus
        if edit:
            hx, hy = (int(v) for v in self.handle_pos())
            # Punkt am rechten Rand des Rechtecks (Verbindung zum Handle)
            ex = int(self.cx + self.b / 2 * math.cos(math.radians(self.winkel)))
            ey = int(self.cy + self.b / 2 * math.sin(math.radians(self.winkel)))
            pygame.draw.line(surf, DIM, (ex, ey), (hx, hy), 1)
            # Handle-Kreis: gelb wenn gerade gedreht, sonst cyan
            pygame.draw.circle(surf, GELB if self.gedreht else CYAN, (hx, hy), HANDLE_R)
            pygame.draw.circle(surf, WEISS, (hx, hy), HANDLE_R, 1)

    # Methoden, die Unterklassen überschreiben können:
    def _streifen(self, surf, ecken): pass  # Keine Streifen in der Basis
    def _label(self): return ""             # Kein Label in der Basis

    def als_dict(self):
        """Serialisiert die Linie für die JSON-Speicherung."""
        return {"cx": self.cx, "cy": self.cy, "winkel": self.winkel}


# ── Unterklasse: Start/Ziel-Linie ────────────────────────────
class StartZiel(Linie):
    """
    Die Start/Ziel-Linie. Erbt Linie mit oranger Farbe und
    Schachbrettmuster aus 8 schwarz-weißen Querstreifen.
    """
    def __init__(self, cx, cy):
        super().__init__(cx, cy, SZ_B, SZ_H, ORANGE, (255, 175, 65))

    def _streifen(self, s, e):
        streifen(s, e, 8, SCHWARZ, WEISS)  # 8 Streifen, schwarz-weiß

    @classmethod
    def von_dict(cls, d):
        """Erstellt ein StartZiel-Objekt aus einem gespeicherten Dict."""
        o = cls(d["cx"], d["cy"])
        o.winkel = d.get("winkel", 0.)
        return o


# ── Unterklasse: Checkpoint ───────────────────────────────────
class Checkpoint(Linie):
    """
    Ein Checkpoint, den Autos passieren müssen, bevor die Ziellinie
    als gültig gewertet wird. Jeder hat einen Index (1, 2, 3, …),
    der als Label angezeigt wird.
    """
    def __init__(self, cx, cy, idx=0):
        super().__init__(cx, cy, CP_B, CP_H, LILA, (190, 100, 250))
        self.idx = idx  # Position in der Checkpoint-Reihenfolge (0-basiert)

    def _streifen(self, s, e):
        streifen(s, e, 6, (190, 100, 250), LILA2)  # 6 lila Streifen

    def _label(self):
        return str(self.idx + 1)  # Zeigt "1", "2", "3", … an

    def als_dict(self):
        return {**super().als_dict(), "idx": self.idx}

    @classmethod
    def von_dict(cls, d):
        o = cls(d["cx"], d["cy"], d.get("idx", 0))
        o.winkel = d.get("winkel", 0.)
        return o


# ═══════════════════════════════════════════════════════════════
#  KLASSE: Strecke
# ═══════════════════════════════════════════════════════════════

class Strecke:
    """
    Verwaltet alle Daten der gezeichneten Strecke:
    - Die Pixel-Surface (schwarz = Asphalt, grün = Gras)
    - Die Rohdaten der gezeichneten Linien
    - Die Start/Ziel-Linie und alle Checkpoints
    """

    def __init__(self, gras):
        """
        gras = vorgerenderte Gras-Textur (pygame.Surface)
        """
        self.gras  = gras
        # surf = die eigentliche Spielfeld-Surface; schwarze Pixel = Asphalt
        self.surf  = pygame.Surface((W, H - TB))
        self.linien: list        = []    # Liste von Punktlisten [[p1,p2,...], ...]
        self.start: StartZiel | None = None
        self.cps: list[Checkpoint]   = []
        self.dicke = 20  # Strichbreite der gezeichneten Strecke in Pixeln

    def leere(self):
        """Löscht die gesamte Strecke (Linien, CPs, Start/Ziel)."""
        self.linien.clear()
        self.cps.clear()
        self.start = None
        self.rendere()

    def rendere(self, dicke=None):
        """
        Zeichnet die Strecke neu auf self.surf.
        Zuerst Gras-Hintergrund, dann jede gezeichnete Linie als
        Folge von gefüllten Kreisen (Radius = self.dicke).
        Optional kann eine neue Strichdicke übergeben werden.
        """
        if dicke is not None:
            self.dicke = dicke
        self.surf.blit(self.gras, (0, 0))  # Hintergrund zurücksetzen
        for l in self.linien:
            if l:
                # Geglättete Punkte als schwarze Kreise zeichnen → Asphalt
                for p in glaette(l):
                    pygame.draw.circle(self.surf, SCHWARZ, p, self.dicke)

    def pixel_ok(self, x, y):
        """
        Gibt True zurück, wenn der Pixel an (x, y) schwarz ist,
        also auf dem Asphalt liegt. Klemmt Koordinaten ans Spielfeld.
        """
        x = max(0, min(int(x), W - 1))
        y = max(0, min(int(y), H - TB - 1))
        return self.surf.get_at((x, y))[:3] == SCHWARZ

    def auf_strecke(self, rect):
        """
        Bestimmt, ob ein Auto (gegeben durch sein Rect) auf der Strecke ist.
        Prüft 9 charakteristische Punkte des Rects auf schwarze Pixel.
        Gibt (anteil, bool) zurück:
          anteil = Bruchteil der Punkte auf Asphalt (0.0–1.0)
          bool   = True wenn anteil >= SCHWELLE (0.30)
        """
        pts = [rect.topleft, rect.topright, rect.bottomleft, rect.bottomright,
               rect.center, rect.midtop, rect.midbottom, rect.midleft, rect.midright]
        anteil = sum(1 for x, y in pts if self.pixel_ok(x, y)) / len(pts)
        return anteil, anteil >= SCHWELLE


# ═══════════════════════════════════════════════════════════════
#  AUTO-SPRITE
# ═══════════════════════════════════════════════════════════════

def baue_auto_image(farbe):
    """
    Erzeugt das Basis-Sprite eines Autos als Surface mit Transparenz.
    Zeichnet: Karosserie, Windschutzscheibe, Scheinwerfer (vorne)
    und Rücklichter (hinten).
    farbe = Hauptfarbe des Autos (RGB-Tupel)
    """
    s = pygame.Surface((AUTO_B, AUTO_H), pygame.SRCALPHA)

    # Karosserie
    pygame.draw.rect(s, farbe, (0, 0, AUTO_B, AUTO_H), border_radius=5)

    # Windschutzscheibe (aufgehellte Karosseriefarbe)
    wf = tuple(min(255, c + 60) for c in farbe)
    pygame.draw.rect(s, wf, (3, 8, AUTO_B - 6, 12), border_radius=3)

    # Scheinwerfer vorne (gelblich-weiß) und Rücklichter (rot)
    for x in (4, AUTO_B - 4):
        pygame.draw.circle(s, (255, 255, 200), (x, 3), 2)       # Scheinwerfer
        pygame.draw.circle(s, ROT, (x, AUTO_H - 4), 2)          # Rücklicht

    return s


class Auto(pygame.sprite.Sprite):
    """
    Repräsentiert ein fahrbares Auto. Erbt von pygame.sprite.Sprite,
    damit es in einer Sprite-Group verwaltet werden kann.
    """

    def __init__(self, x, y, farbe, keys, name):
        """
        x, y  = Startposition (Pixel)
        farbe = RGB-Farbe des Autos
        keys  = Dict mit Pygame-Keycodes:
                {"vor": K_w, "rück": K_s, "li": K_a, "re": K_d}
        name  = Anzeigename im HUD ("ROT" / "BLAU")
        """
        super().__init__()
        self.name  = name
        self.farbe = farbe
        self.keys  = keys

        self.base  = baue_auto_image(farbe)   # Unverändertes Basis-Sprite
        self.image = self.base.copy()          # Aktuell rotiertes Sprite
        self.pos   = pygame.Vector2(x, y)     # Sub-Pixel-genaue Position
        self.angle = 0.     # Blickrichtung in Grad (0 = nach oben)
        self.speed = 0.     # Aktuelle Geschwindigkeit (+ = vorwärts)

        # Pygame-Rect für Kollision und Rendering (wird jedes Frame aktualisiert)
        self.rect  = self.image.get_rect(center=(x, y))

        # Rennfortschritt
        self.runden  = 0           # Abgeschlossene Runden
        self.im_ziel = False       # True wenn Auto gerade die Ziellinie berührt
                                   # (verhindert mehrfaches Zählen)
        self.cps_ok: set[int] = set()  # Indices der bereits passierten Checkpoints

        # Platzierungs-Modus (Modus.AUTOS)
        self.gezogen   = False     # True wenn Spieler das Auto gerade zieht
        self.offset    = (0, 0)    # Maus-Offset beim Ziehen
        self.platziert = False     # True wenn Auto gültig auf Asphalt steht

        self._shake = 0  # Zähler für den Offroad-Shake-Effekt (0 = kein Shake)

    def update(self, strecke):
        """
        Wird jeden Frame aufgerufen wenn das Rennen läuft.
        Liest Tastatureingaben, berechnet neue Geschwindigkeit,
        Lenkung, Position und prüft Checkpoint-/Zielkollisionen.
        """
        k  = pygame.key.get_pressed()
        _, auf = strecke.auf_strecke(self.rect)  # auf = True wenn auf Asphalt
        vm = MAX_V if auf else MAX_OFF            # Erlaubte Höchstgeschwindigkeit

        # ── Geschwindigkeit ──────────────────────────────────
        if k[self.keys["vor"]]:
            self.speed = min(self.speed + ACCEL, vm)    # Gas geben
        elif k[self.keys["rück"]]:
            self.speed = max(self.speed - BREMSE, -vm / 2)  # Bremsen/Rückwärts
        else:
            # Natürliche Reibung: Geschwindigkeit nähert sich 0 an
            r = REIB if auf else REIB_OFF
            self.speed = (max(0., self.speed - r) if self.speed > 0
                          else min(0., self.speed + r))

        # ── Lenkung ──────────────────────────────────────────
        if abs(self.speed) > 0.1:
            # Lenkwinkel nimmt bei hoher Geschwindigkeit etwas ab (Realismus)
            # Beim Rückwärtsfahren dreht das Steuer invertiert
            lw = LENKUNG * (1. - abs(self.speed) / MAX_V * 0.35) * (1 if self.speed > 0 else -1)
            if k[self.keys["li"]]: self.angle += lw
            if k[self.keys["re"]]: self.angle -= lw
        self.angle %= 360

        # ── Bewegung ─────────────────────────────────────────
        # Rotiertes Sprite neu zeichnen
        self.image = pygame.transform.rotate(self.base, self.angle)
        # Richtungsvektor: (0,-1) = nach oben, dann um angle drehen
        richt = pygame.Vector2(0, -1).rotate(-self.angle)
        self.pos  += richt * self.speed
        self.rect  = self.image.get_rect(center=(int(self.pos.x), int(self.pos.y)))

        # ── Spielfeldrand-Kollision ───────────────────────────
        if not pygame.Rect(0, 0, W, H - TB).contains(self.rect):
            # Bewegung rückgängig machen und Geschwindigkeit umkehren
            self.pos  -= richt * self.speed
            self.rect  = self.image.get_rect(center=(int(self.pos.x), int(self.pos.y)))
            self.speed *= -0.3  # Gedämpftes Abprallen

        # ── Offroad-Shake-Effekt ──────────────────────────────
        # Zähler läuft in 6er-Schritten, solange das Auto offroad und schnell ist
        self._shake = (self._shake + 1) % 6 if not auf and abs(self.speed) > 0.2 else 0

        # ── Checkpoint-Erkennung ──────────────────────────────
        for cp in strecke.cps:
            if cp.kollidiert(self.rect):
                self.cps_ok.add(cp.idx)  # CP als "passiert" markieren

        # ── Ziel-Erkennung ────────────────────────────────────
        if strecke.start:
            # Runde gilt nur wenn alle Checkpoints bereits passiert wurden
            alle = len(self.cps_ok) >= len(strecke.cps)
            if strecke.start.kollidiert(self.rect):
                if not self.im_ziel and self.speed > 0 and alle:
                    self.runden += 1        # Runde zählen
                    self.cps_ok.clear()     # Checkpoints zurücksetzen
                    self.im_ziel = True     # Verhindern, dass es doppelt zählt
            else:
                self.im_ziel = False        # Auto hat Ziellinie wieder verlassen

    def shake(self):
        """
        Gibt einen zufälligen (dx, dy)-Offset für den Offroad-Vibrationseffekt zurück.
        Gibt (0,0) zurück wenn kein Shake aktiv ist.
        """
        return (random.randint(-1, 1), random.randint(-1, 1)) if self._shake else (0, 0)

    @property
    def hat_gewonnen(self):
        """True wenn das Auto RUNDEN_MAX Runden abgeschlossen hat."""
        return self.runden >= RUNDEN_MAX


# ═══════════════════════════════════════════════════════════════
#  UI-WIDGETS
# ═══════════════════════════════════════════════════════════════

class Knopf:
    """
    Ein einfacher klickbarer Button für die Toolbar.
    Unterstützt Hover-Hervorhebung und einen "aktiv"-Zustand
    (z.B. wenn der zugehörige Modus gerade aktiv ist).
    """

    def __init__(self, rect, text, aktiv_f=GRUEN):
        """
        rect    = pygame.Rect mit Position und Größe
        text    = Beschriftung
        aktiv_f = Hintergrundfarbe im aktiven Zustand
        """
        self.rect    = rect
        self.text    = text
        self.aktiv_f = aktiv_f
        self.aktiv   = False  # Wird von Spiel._sync_knoepfe() gesetzt

    def zeichne(self, surf, font):
        """Zeichnet den Button mit Hover- und Aktiv-Effekt."""
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        # Hintergrundfarbe: aktiv > hover > normal
        bg   = self.aktiv_f if self.aktiv else (RAND if hover else PANEL)
        # Rahmen etwas heller als der Hintergrund wenn aktiv
        rand = tuple(min(255, c + 40) for c in bg) if self.aktiv else RAND
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, rand, self.rect, 1, border_radius=6)
        t = font.render(self.text, True, WEISS if (self.aktiv or hover) else TEXT)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def hit(self, ev):
        """True wenn der Button mit der linken Maustaste geklickt wurde."""
        return (ev.type == pygame.MOUSEBUTTONDOWN
                and ev.button == 1
                and self.rect.collidepoint(ev.pos))


class Regler:
    """
    Ein einfacher +/−-Regler zur Einstellung der Strichdicke.
    Besteht aus drei Rects: Minus-Button, Wert-Anzeige, Plus-Button.
    """

    def __init__(self, x, y):
        """x, y = linke obere Ecke des Reglers in der Toolbar."""
        self.rm   = pygame.Rect(x,      y, 26, 26)  # "−"-Button
        self.rp   = pygame.Rect(x + 66, y, 26, 26)  # "+"-Button
        self.rw   = pygame.Rect(x + 28, y, 36, 26)  # Wert-Anzeige (nicht klickbar)
        self.wert = 20  # Aktueller Wert (= Strecken-Strichdicke in Pixeln)

    def zeichne(self, surf, font):
        """Zeichnet Minus-Button, Wertfeld und Plus-Button."""
        for r, t in [(self.rm, "−"), (self.rp, "+")]:
            h = r.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(surf, RAND if h else PANEL, r, border_radius=4)
            pygame.draw.rect(surf, DIM, r, 1, border_radius=4)
            lbl = font.render(t, True, TEXT)
            surf.blit(lbl, lbl.get_rect(center=r.center))
        pygame.draw.rect(surf, SCHWARZ, self.rw, border_radius=3)
        v = font.render(str(self.wert), True, GELB)
        surf.blit(v, v.get_rect(center=self.rw.center))
        surf.blit(font.render("Dicke", True, DIM), (self.rm.x, self.rm.y - 17))

    def click(self, ev):
        """
        Verarbeitet einen Mausklick. Erhöht oder verringert self.wert
        um DICKE_S innerhalb [DICKE_MIN, DICKE_MAX].
        Gibt True zurück wenn ein Button getroffen wurde.
        """
        if ev.type != pygame.MOUSEBUTTONDOWN or ev.button != 1:
            return False
        if self.rm.collidepoint(ev.pos):
            self.wert = max(DICKE_MIN, self.wert - DICKE_S)
            return True
        if self.rp.collidepoint(ev.pos):
            self.wert = min(DICKE_MAX, self.wert + DICKE_S)
            return True
        return False


# ═══════════════════════════════════════════════════════════════
#  HAUPTKLASSE: Spiel
# ═══════════════════════════════════════════════════════════════

class Spiel:
    """
    Hauptklasse des Spiels. Verwaltet alle Objekte, den Spielzustand,
    die Event-Verarbeitung und das Rendering.
    """

    # Hilfstexte, die je nach Modus in der Toolbar angezeigt werden
    HINTS = {
        Modus.ZEICHNEN: "Maus halten: Strecke zeichnen",
        Modus.START:    "Klick: setzen | Ziehen: verschieben | Handle: drehen",
        Modus.CP:       "Klick: neu | Ziehen: verschieben | Handle: drehen | Rechtsklick: loeschen",
        Modus.AUTOS:    "Auto auf Strecke ziehen (gruen = OK) | Scroll: drehen",
        Modus.FAHREN:   "ROT: WASD   |   BLAU: Pfeiltasten",
    }

    def __init__(self):
        pygame.init()
        self.win  = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Rennspiel")
        self.uhr  = pygame.time.Clock()

        # Schriftarten in verschiedenen Größen
        self.font  = pygame.font.SysFont("segoeui", 20)           # Standard
        self.fontB = pygame.font.SysFont("segoeui", 22, bold=True) # Fett (HUD)
        self.fontS = pygame.font.SysFont("segoeui", 16)            # Klein (Hints, CP-Label)
        self.fontG = pygame.font.SysFont("segoeui", 72, bold=True) # Groß (Gewinner-Overlay)

        # Strecke initialisieren
        gras = gras_textur(W, H - TB)
        self.strecke = Strecke(gras)

        # Aktueller Spielmodus
        self.modus = Modus.ZEICHNEN

        # Aktuell bearbeitete Linie (CP oder StartZiel) beim Verschieben/Drehen
        self.edit = None

        # Aktuell gezogenes Auto im Autos-Modus
        self.edit_auto = None

        # True solange Maus gedrückt gehalten wird (Strecke zeichnen)
        self.zeichnet = False

        # Feedback-Nachricht: (text, verbleibende_frames)
        self.feedback = ("", 0)

        # ── Toolbar-Buttons ───────────────────────────────────
        ty = H - TB + TB // 2 - 13  # Vertikale Mitte der Toolbar
        bh = 28                      # Button-Höhe
        self.knoepfe = {
            "zeichnen":  Knopf(pygame.Rect(  8, ty, 100, bh), "Strecke"),
            "start":     Knopf(pygame.Rect(116, ty, 110, bh), "Start/Ziel",  ORANGE),
            "cp":        Knopf(pygame.Rect(234, ty, 120, bh), "Checkpoints", LILA),
            "autos":     Knopf(pygame.Rect(362, ty,  80, bh), "Autos",       CYAN),
            "fahren":    Knopf(pygame.Rect(450, ty,  90, bh), "Fahren",      GRUEN),
            "reset":     Knopf(pygame.Rect(W - 230, ty, 80, bh), "Reset",    ROT),
            "speichern": Knopf(pygame.Rect(W - 142, ty,130, bh), "Speichern",BLAU),
        }

        # Strichdicken-Regler (nur sichtbar im Zeichnen-Modus)
        self.regler = Regler(590, H - TB + 16)

        # ── Autos erstellen ───────────────────────────────────
        # Auto 1 (ROT) = WASD-Steuerung
        self.auto1 = Auto(120, 100, ROT,
            {"vor": pygame.K_w, "rück": pygame.K_s,
             "li":  pygame.K_a, "re":   pygame.K_d}, "ROT")
        # Auto 2 (BLAU) = Pfeiltasten
        self.auto2 = Auto(180, 100, BLAU,
            {"vor": pygame.K_UP,   "rück": pygame.K_DOWN,
             "li":  pygame.K_LEFT, "re":   pygame.K_RIGHT}, "BLAU")
        # Sprite-Group für gleichzeitiges update() aller Autos
        self.autos = pygame.sprite.Group(self.auto1, self.auto2)

        self._lade()          # Gespeicherte Strecke laden (falls vorhanden)
        self._sync_knoepfe()  # Button-Aktiv-Zustand initialisieren

    # ── Modus-Verwaltung ──────────────────────────────────────

    def _modus(self, m):
        """Wechselt den Spielmodus und setzt die Auswahl zurück."""
        self.modus = m
        self.edit  = None
        self._sync_knoepfe()

    def _sync_knoepfe(self):
        """Setzt den aktiv-Flag aller Modus-Buttons passend zum aktuellen Modus."""
        for key, mod in [("zeichnen", Modus.ZEICHNEN), ("start", Modus.START),
                         ("cp", Modus.CP), ("autos", Modus.AUTOS), ("fahren", Modus.FAHREN)]:
            self.knoepfe[key].aktiv = (self.modus == mod)

    def _info(self, t, d=110):
        """
        Setzt eine Feedback-Nachricht (z.B. "Gespeichert").
        d = Anzahl Frames, die sie sichtbar bleibt (Standard: ~1.8 Sekunden bei 60 FPS).
        """
        self.feedback = (t, d)

    # ── Speichern / Laden ─────────────────────────────────────

    def _speichere(self):
        """Speichert die aktuelle Strecke als JSON-Datei."""
        d = {
            "linien": self.strecke.linien,
            "dicke":  self.regler.wert,
            "start":  self.strecke.start.als_dict() if self.strecke.start else None,
            "cps":    [c.als_dict() for c in self.strecke.cps],
        }
        SAVE.write_text(json.dumps(d), encoding="utf-8")
        self._info("Gespeichert")

    def _lade(self):
        """Lädt eine gespeicherte Strecke aus der JSON-Datei, falls vorhanden."""
        if not SAVE.exists():
            return
        d = json.loads(SAVE.read_text(encoding="utf-8"))
        # Punktlisten rekonstruieren (JSON speichert Listen, kein Tupel)
        self.strecke.linien = [[tuple(p) for p in l] for l in d.get("linien", [])]
        self.regler.wert    = d.get("dicke", 20)
        if d.get("start"):
            self.strecke.start = StartZiel.von_dict(d["start"])
        for c in d.get("cps", []):
            self.strecke.cps.append(Checkpoint.von_dict(c))
        self.strecke.rendere(self.regler.wert)

    # ── Event-Handling ────────────────────────────────────────

    def events(self):
        """Verarbeitet alle pygame-Events des aktuellen Frames."""
        for ev in pygame.event.get():

            # Fenster schließen oder ESC: Programm beenden
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                # Ausnahme: Im Gewonnen-Bildschirm → Neustart statt Beenden
                if self.modus == Modus.GEWONNEN and ev.type == pygame.KEYDOWN:
                    self._neustart()
                    return
                pygame.quit()
                sys.exit()

            # Beliebige Taste im Gewonnen-Bildschirm → Neustart
            if ev.type == pygame.KEYDOWN and self.modus == Modus.GEWONNEN:
                self._neustart()
                return

            # Mausrad: Auto drehen (Autos-Modus) oder Linie drehen (Edit)
            if ev.type == pygame.MOUSEWHEEL:
                if self.modus == Modus.AUTOS:
                    mx, my = pygame.mouse.get_pos()
                    for a in self.autos:
                        if a.rect.collidepoint(mx, my):
                            # ev.y = Scroll-Richtung (+1 oder -1), × 5 Grad pro Schritt
                            a.angle = (a.angle + ev.y * 5) % 360
                            a.image = pygame.transform.rotate(a.base, a.angle)
                            a.rect  = a.image.get_rect(center=(int(a.pos.x), int(a.pos.y)))
                elif self.edit and isinstance(self.edit, Linie):
                    self.edit.winkel = (self.edit.winkel + ev.y * 5) % 360
                    self.strecke.rendere()

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if   ev.button == 1: self._down(ev)   # Linksklick
                elif ev.button == 3: self._rechts(ev)  # Rechtsklick

            elif ev.type == pygame.MOUSEMOTION:
                self._move(ev)

            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self._up(ev)

    def _up(self, ev):
        """Linke Maustaste losgelassen: Zeichnen/Ziehen/Drehen beenden."""
        self.zeichnet = False

        # Linie loslassen
        if self.edit and isinstance(self.edit, Linie):
            self.edit.stop()
            self.strecke.rendere()

        # Auto loslassen: Prüfen ob es auf der Strecke steht
        if self.edit_auto:
            a = self.edit_auto
            a.gezogen  = False
            _, ok      = self.strecke.auf_strecke(a.rect)
            a.platziert = ok                              # Grüner/roter Rahmen
            a.pos       = pygame.Vector2(a.rect.center)  # Pos mit Rect synchronisieren
            self.edit_auto = None

    def _rechts(self, ev):
        """Rechtsklick im CP-Modus: Checkpoint unter der Maus löschen."""
        if self.modus != Modus.CP:
            return
        mx, my = ev.pos
        for i, cp in enumerate(self.strecke.cps):
            if cp.body_hit(mx, my):
                self.strecke.cps.pop(i)
                # Indices aller verbleibenden CPs neu nummerieren
                for j, c in enumerate(self.strecke.cps):
                    c.idx = j
                if self.edit is cp:
                    self.edit = None
                self.strecke.rendere()
                self._info("Checkpoint entfernt")
                break

    def _down(self, ev):
        """Linke Maustaste gedrückt: Buttons, Regler oder Spielfeld-Interaktion."""

        # ① Button-Klick prüfen (hat Vorrang)
        for n, k in self.knoepfe.items():
            if k.hit(ev):
                self._knopf(n)
                return

        # ② Dicken-Regler klicken (nur im Zeichnen-Modus sichtbar)
        if self.modus == Modus.ZEICHNEN and self.regler.click(ev):
            self.strecke.rendere(self.regler.wert)
            return

        # ③ Klick in der Toolbar-Zone ignorieren
        if ev.pos[1] >= H - TB:
            return

        mx, my = ev.pos

        if self.modus == Modus.ZEICHNEN:
            # Neuen Strichzug beginnen
            self.zeichnet = True
            self.strecke.linien.append([ev.pos])

        elif self.modus == Modus.START:
            sz = self.strecke.start
            if sz and sz.handle_hit(mx, my):
                sz.start_drehen(mx, my); self.edit = sz       # Dreh-Handle getroffen
            elif sz and sz.body_hit(mx, my):
                sz.start_ziehen(mx, my); self.edit = sz       # Körper getroffen → ziehen
            else:
                # Neue Start/Ziel-Linie an Klickposition erstellen
                self.strecke.start = StartZiel(mx, my)
                self.edit = self.strecke.start
                self.strecke.rendere()

        elif self.modus == Modus.CP:
            hit = None
            for cp in self.strecke.cps:
                if cp.handle_hit(mx, my):
                    cp.start_drehen(mx, my); self.edit = cp; hit = cp; break
                elif cp.body_hit(mx, my):
                    cp.start_ziehen(mx, my); self.edit = cp; hit = cp; break
            if not hit:
                # Keinen CP getroffen → neuen Checkpoint an Klickposition
                cp = Checkpoint(mx, my, len(self.strecke.cps))
                self.strecke.cps.append(cp)
                self.edit = cp
                self.strecke.rendere()

        elif self.modus == Modus.AUTOS:
            for a in self.autos:
                if a.rect.collidepoint(ev.pos):
                    a.gezogen    = True
                    a.offset     = (a.rect.x - mx, a.rect.y - my)
                    self.edit_auto = a
                    break

    def _knopf(self, n):
        """Verarbeitet einen Button-Klick anhand des Button-Namens 'n'."""
        if n == "fahren":
            # Fahren nur starten wenn beide Autos platziert und Start/Ziel gesetzt
            if self.auto1.platziert and self.auto2.platziert and self.strecke.start:
                self._modus(Modus.FAHREN)
            else:
                self._info("Autos platzieren + Start/Ziel setzen")
        elif n == "reset":
            self.strecke.leere()
            self.edit = None
            self._info("Strecke geleert")
        elif n == "speichern":
            self._speichere()
        else:
            # Alle Modus-Buttons direkt mappen
            modus_map = {
                "zeichnen": Modus.ZEICHNEN,
                "start":    Modus.START,
                "cp":       Modus.CP,
                "autos":    Modus.AUTOS,
            }
            if n in modus_map:
                self._modus(modus_map[n])

    def _move(self, ev):
        """Maus bewegt: Strecke zeichnen oder Objekt verschieben/drehen."""
        mx, my = ev.pos

        if self.modus == Modus.ZEICHNEN and self.zeichnet and my < H - TB:
            # Aktuellen Maus-Punkt dem laufenden Strichzug anhängen
            self.strecke.linien[-1].append(ev.pos)
            self.strecke.rendere()

        elif (self.edit and isinstance(self.edit, Linie)
              and (self.edit.gedreht or self.edit.gezogen)):
            # Aktive Linie bewegen/drehen
            self.edit.update(mx, my)
            self.strecke.rendere()

        elif self.edit_auto:
            # Auto verschieben (offset verhindert Sprung zum Mauszeiger)
            a = self.edit_auto
            a.rect.x = mx + a.offset[0]
            a.rect.y = my + a.offset[1]

    # ── Neustart ──────────────────────────────────────────────

    def _neustart(self):
        """
        Setzt den Rennfortschritt aller Autos zurück und wechselt
        in den Autos-Modus, damit die Startpositionen neu gewählt werden können.
        """
        for a in self.autos:
            a.runden   = 0
            a.im_ziel  = False
            a.cps_ok.clear()
            a.speed    = 0
        self._modus(Modus.AUTOS)

    # ── Rendering ────────────────────────────────────────────

    def _draw_feld(self):
        """Zeichnet das Spielfeld: Strecke, Linien, Autos und modusspezifische Overlays."""

        # Strecken-Surface als Hintergrund
        self.win.blit(self.strecke.surf, (0, 0))

        # Checkpoints zeichnen (ausgewählt = weißer Rahmen, im CP-Modus = mit Handle)
        for cp in self.strecke.cps:
            cp.zeichne(self.win, self.fontS,
                       sel=cp is self.edit, edit=self.modus == Modus.CP)

        # Start/Ziel-Linie zeichnen
        if self.strecke.start:
            self.strecke.start.zeichne(self.win, self.fontS,
                sel=self.strecke.start is self.edit,
                edit=self.modus == Modus.START)

        # Autos zeichnen (mit optionalem Shake-Offset bei Offroad)
        for a in self.autos:
            ox, oy = a.shake()
            self.win.blit(a.image, (a.rect.x + ox, a.rect.y + oy))

        # Im Autos-Modus: farbigen Rahmen um jedes Auto (grün = OK, rot = nicht auf Strecke)
        if self.modus == Modus.AUTOS:
            for a in self.autos:
                c = GRUEN if a.platziert else ROT
                pygame.draw.rect(self.win, c, a.rect.inflate(6, 6), 2, border_radius=4)

        # Im CP-Modus: halbtransparente Vorschau eines neuen CPs an der Mausposition
        if self.modus == Modus.CP:
            mx, my = pygame.mouse.get_pos()
            if my < H - TB and not any(cp.body_hit(mx, my) for cp in self.strecke.cps):
                s = pygame.Surface((CP_B, CP_H), pygame.SRCALPHA)
                s.fill((*LILA, 90))
                self.win.blit(s, (mx - CP_B // 2, my - CP_H // 2))

        # Im Fahren-Modus: "OFFROAD"-Label über Autos die daneben sind
        if self.modus == Modus.FAHREN:
            for a in [self.auto1, self.auto2]:
                _, auf = self.strecke.auf_strecke(a.rect)
                if not auf and abs(a.speed) > 0.3:
                    t = self.fontS.render("OFFROAD", True, a.farbe)
                    self.win.blit(t, (a.rect.centerx - t.get_width() // 2, a.rect.top - 20))

    def _draw_toolbar(self):
        """Zeichnet die untere Toolbar mit Buttons, Regler und Hinweistext."""
        pygame.draw.rect(self.win, TB_C, (0, H - TB, W, TB))
        pygame.draw.line(self.win, RAND, (0, H - TB), (W, H - TB), 1)

        for k in self.knoepfe.values():
            k.zeichne(self.win, self.font)

        if self.modus == Modus.ZEICHNEN:
            self.regler.zeichne(self.win, self.fontS)

        # Hinweistext zentriert in der Toolbar
        t = self.fontS.render(self.HINTS.get(self.modus, ""), True, DIM)
        self.win.blit(t, t.get_rect(center=(W // 2, H - 12)))

    def _draw_hud(self):
        """
        Zeichnet das Head-Up-Display (HUD) oben links:
        Name, Rundenanzahl und Checkpoint-Fortschrittsbalken für jedes Auto.
        Außerdem die Feedback-Nachricht in der Mitte.
        """
        ncps = len(self.strecke.cps)  # Gesamtanzahl Checkpoints

        for i, a in enumerate([self.auto1, self.auto2]):
            px, py = 12, 12 + i * 58  # Versatz: Auto 2 ist 58 px tiefer

            # Halbtransparentes Hintergrund-Panel
            s = pygame.Surface((190, 46), pygame.SRCALPHA)
            s.fill((*PANEL, 200))
            self.win.blit(s, (px, py))
            pygame.draw.rect(self.win, a.farbe, (px, py, 190, 46), 1, border_radius=5)

            # Name und Rundenanzeige
            n = self.fontB.render(a.name, True, a.farbe)
            r = self.fontB.render(f"Runde {a.runden}/{RUNDEN_MAX}", True, TEXT)
            self.win.blit(n, (px + 6, py + 4))
            self.win.blit(r, (px + 6 + n.get_width() + 8, py + 4))

            # Checkpoint-Fortschrittsbalken (nur wenn CPs vorhanden und Rennen läuft)
            if ncps > 0 and self.modus == Modus.FAHREN:
                ok = len(a.cps_ok)  # Bereits passierte CPs in dieser Runde
                bw = 178            # Balkenbreite in Pixeln
                pygame.draw.rect(self.win, RAND,    (px + 6, py + 30, bw, 8), border_radius=3)
                if ok:
                    # Gefüllter Anteil proportional zu passierten CPs
                    pygame.draw.rect(self.win, a.farbe,
                        (px + 6, py + 30, int(bw * ok / ncps), 8), border_radius=3)
                ct = self.fontS.render(f"CP {ok}/{ncps}", True, DIM)
                self.win.blit(ct, (px + bw - ct.get_width() + 6, py + 30))

        # Feedback-Nachricht (z.B. "Gespeichert") zentriert über der Toolbar
        txt, t = self.feedback
        if t > 0:
            ft = self.font.render(txt, True, GELB)
            bg = pygame.Surface((ft.get_width() + 16, ft.get_height() + 8), pygame.SRCALPHA)
            bg.fill((0, 0, 0, 180))
            cx, cy = W // 2, H - TB - 36
            self.win.blit(bg, (cx - bg.get_width() // 2, cy))
            self.win.blit(ft, ft.get_rect(center=(cx, cy + bg.get_height() // 2)))
            self.feedback = (txt, t - 1)  # Countdown um 1 Frame reduzieren

    def _draw_gewonnen(self, gewinner: Auto):
        """
        Zeichnet das Sieger-Overlay über das gesamte Fenster.
        Halbtransparente schwarze Fläche + Rahmen in Siegerfarbe + Texte.
        """
        # Schwarzes Overlay über alles
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 170))
        self.win.blit(overlay, (0, 0))

        titel = self.fontG.render(f"🏆  {gewinner.name} GEWINNT!", True, gewinner.farbe)
        unter = self.fontB.render(
            f"{RUNDEN_MAX} Runden abgeschlossen  ·  Beliebige Taste drücken für Neustart",
            True, TEXT)

        # Zentrierter Leuchtrahmen in Siegerfarbe
        r = pygame.Rect(0, 0, max(titel.get_width(), unter.get_width()) + 80, 180)
        r.center = (W // 2, H // 2)
        pygame.draw.rect(self.win, (*gewinner.farbe, 60), r, border_radius=16)  # Füllung
        pygame.draw.rect(self.win, gewinner.farbe, r, 3, border_radius=16)      # Rand

        self.win.blit(titel, titel.get_rect(center=(W // 2, H // 2 - 30)))
        self.win.blit(unter, unter.get_rect(center=(W // 2, H // 2 + 40)))

    def zeichne(self, gewinner=None):
        """
        Haupt-Rendering-Methode: ruft alle Teil-Renderer auf und
        flippt dann den Display-Buffer (doppeltes Puffern).
        gewinner = Auto-Objekt falls jemand gewonnen hat, sonst None.
        """
        self._draw_feld()
        self._draw_toolbar()
        self._draw_hud()
        if gewinner:
            self._draw_gewonnen(gewinner)
        pygame.display.flip()

    # ── Hauptschleife ────────────────────────────────────────

    def run(self):
        """
        Spielschleife: läuft endlos mit FPS-Begrenzung.
        Reihenfolge pro Frame:
          1. Events verarbeiten
          2. Spiellogik updaten (Autos fahren, Siegprüfung)
          3. Alles rendern
          4. Auf nächsten Frame warten
        """
        gewinner = None  # Wird gesetzt sobald ein Auto RUNDEN_MAX erreicht

        while True:
            self.events()

            if self.modus == Modus.FAHREN:
                # Alle Autos einen Frame vorwärts simulieren
                self.autos.update(self.strecke)

                # Siegprüfung: erstes Auto das RUNDEN_MAX erreicht gewinnt
                for a in [self.auto1, self.auto2]:
                    if a.hat_gewonnen and gewinner is None:
                        gewinner      = a
                        self.modus    = Modus.GEWONNEN

            # Gewinner zurücksetzen wenn Neustart ausgelöst wurde
            if self.modus != Modus.GEWONNEN:
                gewinner = None

            # Gewinner-Overlay nur anzeigen wenn Modus GEWONNEN ist
            self.zeichne(gewinner if self.modus == Modus.GEWONNEN else None)
            self.uhr.tick(FPS)  # Framerate auf FPS begrenzen


# ── Einstiegspunkt ────────────────────────────────────────────
if __name__ == "__main__":
    Spiel().run()