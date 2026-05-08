import pygame
import sys
import json
from pathlib import Path
from enum import Enum, auto

import numpy as np
from scipy.ndimage import uniform_filter1d

# ─────────────────────────────────────────────
#  Konstanten
# ─────────────────────────────────────────────
FENSTER_BREITE  = 1500
FENSTER_HOEHE   = 1000
BILDER_PRO_SEK  = 120

FARBE_WEISS      = (255, 255, 255)
FARBE_SCHWARZ    = (0,   0,   0)
FARBE_DUNKELGRAU = (80,  80,  80)
FARBE_HOVER      = (120, 120, 120)
FARBE_ROT        = (180, 40,  40)
FARBE_ROT_HOVER  = (220, 60,  60)
FARBE_BLAU       = (40,  80,  180)
FARBE_BLAU_HOVER = (60,  110, 220)
FARBE_AKTIV      = (30,  140, 80)

DICKE_MIN     = 2
DICKE_MAX     = 50
DICKE_SCHRITT = 2

# Wie stark geglättet wird (größer = glatter, aber weiter vom Original entfernt)
# Muss ungerade sein für uniform_filter1d; wird intern auf nächste ungerade Zahl gerundet
GLAETTUNGS_STAERKE = 25

# Mindestanzahl Punkte damit Glättung sinnvoll ist
GLAETTUNGS_MINPUNKTE = 4

SPEICHERDATEI = Path(__file__).parent / "linien.json"


# ─────────────────────────────────────────────
#  Enum: Darstellungsform
# ─────────────────────────────────────────────
class Darstellungsform(Enum):
    LINIE   = auto()
    KREIS   = auto()   # mit Glättung
    QUADRAT = auto()


# ─────────────────────────────────────────────
#  Glättungs-Hilfsfunktion
# ─────────────────────────────────────────────
def glaette_punkte(
    punkte: list[tuple[int, int]],
    staerke: int = GLAETTUNGS_STAERKE,
) -> list[tuple[int, int]]:
    """Glättet eine Punkteliste mit einem gleitenden Durchschnitt.

    Funktionsweise
    --------------
    scipy.ndimage.uniform_filter1d berechnet für jeden Punkt den Durchschnitt
    seiner Nachbarn in einem Fenster der Breite `staerke`. Das macht die Kurve
    weich, ohne dass Ausreißer-Punkte stark ins Gewicht fallen.
    Der erste und letzte Punkt bleiben verankert, damit die Linie an den
    Enden nicht „wegdriftet".

    Parameter
    ---------
    punkte  : Rohe Mauskoordinaten
    staerke : Fenstergröße des gleitenden Durchschnitts (größer = glatter)
    """
    if len(punkte) < GLAETTUNGS_MINPUNKTE:
        return punkte  # zu wenig Punkte – nichts zu glätten

    # x- und y-Koordinaten getrennt glätten
    xs = np.array([p[0] for p in punkte], dtype=float)
    ys = np.array([p[1] for p in punkte], dtype=float)

    xs_glatt = uniform_filter1d(xs, size=staerke, mode="nearest")
    ys_glatt = uniform_filter1d(ys, size=staerke, mode="nearest")

    # Anfangs- und Endpunkt fixieren (verhindert Abrutschen der Enden)
    xs_glatt[0],  ys_glatt[0]  = xs[0],  ys[0]
    xs_glatt[-1], ys_glatt[-1] = xs[-1], ys[-1]

    return [(int(x), int(y)) for x, y in zip(xs_glatt, ys_glatt)]


# ─────────────────────────────────────────────
#  Knopf-Klasse
# ─────────────────────────────────────────────
class Knopf:
    """Klickbarer Button mit Hover- und Aktiv-Zustand."""

    def __init__(
        self,
        rect: pygame.Rect,
        beschriftung: str,
        hintergrund: tuple       = None,
        hintergrund_hover: tuple = None,
    ) -> None:
        self._rect              = rect
        self._beschriftung      = beschriftung
        self._hintergrund       = hintergrund       or FARBE_DUNKELGRAU
        self._hintergrund_hover = hintergrund_hover or FARBE_HOVER
        self.aktiv: bool        = False

    def zeichne(self, fenster: pygame.Surface) -> None:
        schrift = pygame.font.SysFont(None, 20)
        mauspos = pygame.mouse.get_pos()

        if self.aktiv:
            farbe = FARBE_AKTIV
        elif self._rect.collidepoint(mauspos):
            farbe = self._hintergrund_hover
        else:
            farbe = self._hintergrund

        pygame.draw.rect(fenster, farbe, self._rect, border_radius=6)
        text    = schrift.render(self._beschriftung, True, FARBE_WEISS)
        fenster.blit(text, text.get_rect(center=self._rect.center))

    def wurde_geklickt(self, ereignis: pygame.event.Event) -> bool:
        return (
            ereignis.type == pygame.MOUSEBUTTONDOWN
            and ereignis.button == 1
            and self._rect.collidepoint(ereignis.pos)
        )


# ─────────────────────────────────────────────
#  Linienspeicher-Klasse
# ─────────────────────────────────────────────
class Linienspeicher:
    """Speichert Roh-Punkte der gezeichneten Linien und kümmert sich um JSON."""

    def __init__(self) -> None:
        # Rohe Mauskoordinaten – werden erst beim Rendern geglättet
        self._linien: list[list[tuple[int, int]]] = []

    def starte_neue_linie(self, startpunkt: tuple[int, int]) -> None:
        self._linien.append([startpunkt])

    def fuege_punkt_hinzu(self, punkt: tuple[int, int]) -> None:
        if self._linien:
            self._linien[-1].append(punkt)

    def alle_linien(self) -> list[list[tuple[int, int]]]:
        return self._linien

    def leere(self) -> None:
        self._linien.clear()

    def speichere_als_json(self, dateipfad: Path) -> None:
        """Speichert die rohen Punkte – die Glättung wird beim Laden neu berechnet."""
        daten = {"linien": self._linien}
        dateipfad.write_text(json.dumps(daten, indent=2), encoding="utf-8")
        print(f"Gespeichert: {dateipfad.resolve()}")

    def lade_aus_json(self, dateipfad: Path) -> None:
        if not dateipfad.exists():
            return
        daten = json.loads(dateipfad.read_text(encoding="utf-8"))
        self._linien = [
            [tuple(p) for p in linie]
            for linie in daten.get("linien", [])
        ]
        print(f"Geladen: {dateipfad.resolve()} ({len(self._linien)} Linien)")


# ─────────────────────────────────────────────
#  Zeichenprogramm-Klasse
# ─────────────────────────────────────────────
class Zeichenprogramm:
    """Hauptklasse: Fenster, Eingaben, Game-Loop."""

    KNOPF_BREITE  = 110
    KNOPF_HOEHE   = 32
    RAND          = 14
    KNOPF_ABSTAND = 8

    def __init__(self) -> None:
        pygame.init()
        self._fenster = pygame.display.set_mode((FENSTER_BREITE, FENSTER_HOEHE))
        pygame.display.set_caption("Zeichenprogramm")
        self._uhr = pygame.time.Clock()

        self._speicher        = Linienspeicher()
        self._speicher.lade_aus_json(SPEICHERDATEI)

        self._zeichnet_gerade = False
        self._form            = Darstellungsform.KREIS
        self._dicke           = 10

        self._erstelle_knoepfe()
        self._aktualisiere_form_knoepfe()

    # ── Initialisierung ───────────────────────

    def _erstelle_knoepfe(self) -> None:
        r  = self.RAND
        kh = self.KNOPF_HOEHE
        kb = self.KNOPF_BREITE
        ab = self.KNOPF_ABSTAND
        y  = FENSTER_HOEHE - kh - r

        x_speichern = FENSTER_BREITE - r - kb
        x_reset     = x_speichern - ab - kb

        self._knopf_speichern = Knopf(pygame.Rect(x_speichern, y, kb, kh), "💾 Speichern")
        self._knopf_reset     = Knopf(
            pygame.Rect(x_reset, y, kb, kh), "🗑 Reset",
            hintergrund=FARBE_ROT, hintergrund_hover=FARBE_ROT_HOVER,
        )

        pfeil_b = 36
        mitte_x = FENSTER_BREITE // 2
        self._knopf_dicker  = Knopf(
            pygame.Rect(mitte_x + 30, y, pfeil_b, kh), "+",
            hintergrund=FARBE_BLAU, hintergrund_hover=FARBE_BLAU_HOVER,
        )
        self._knopf_duenner = Knopf(
            pygame.Rect(mitte_x - 30 - pfeil_b, y, pfeil_b, kh), "−",
            hintergrund=FARBE_BLAU, hintergrund_hover=FARBE_BLAU_HOVER,
        )

        self._knopf_linie   = Knopf(pygame.Rect(r,               y, kb, kh), "— Linie")
        self._knopf_kreis   = Knopf(pygame.Rect(r + kb + ab,     y, kb, kh), "● Kreis (glatt)")
        self._knopf_quadrat = Knopf(pygame.Rect(r + 2*(kb + ab), y, kb, kh), "■ Quadrat")

        self._form_knoepfe = {
            Darstellungsform.LINIE:   self._knopf_linie,
            Darstellungsform.KREIS:   self._knopf_kreis,
            Darstellungsform.QUADRAT: self._knopf_quadrat,
        }

    def _aktualisiere_form_knoepfe(self) -> None:
        for form, knopf in self._form_knoepfe.items():
            knopf.aktiv = (form == self._form)

    # ── Ereignisverarbeitung ──────────────────

    def _verarbeite_ereignisse(self) -> None:
        for ereignis in pygame.event.get():
            if ereignis.type == pygame.QUIT:
                self._beenden()
            if ereignis.type == pygame.KEYDOWN and ereignis.key == pygame.K_ESCAPE:
                self._beenden()
            if ereignis.type == pygame.MOUSEBUTTONDOWN and ereignis.button == 1:
                self._verarbeite_klick(ereignis)
            if ereignis.type == pygame.MOUSEMOTION and self._zeichnet_gerade:
                self._speicher.fuege_punkt_hinzu(ereignis.pos)
            if ereignis.type == pygame.MOUSEBUTTONUP and ereignis.button == 1:
                self._zeichnet_gerade = False

    def _verarbeite_klick(self, ereignis: pygame.event.Event) -> None:
        if self._knopf_speichern.wurde_geklickt(ereignis):
            self._speicher.speichere_als_json(SPEICHERDATEI)
            return
        if self._knopf_reset.wurde_geklickt(ereignis):
            self._speicher.leere()
            return
        if self._knopf_dicker.wurde_geklickt(ereignis):
            self._dicke = min(self._dicke + DICKE_SCHRITT, DICKE_MAX)
            return
        if self._knopf_duenner.wurde_geklickt(ereignis):
            self._dicke = max(self._dicke - DICKE_SCHRITT, DICKE_MIN)
            return
        for form, knopf in self._form_knoepfe.items():
            if knopf.wurde_geklickt(ereignis):
                self._form = form
                self._aktualisiere_form_knoepfe()
                return

        self._zeichnet_gerade = True
        self._speicher.starte_neue_linie(ereignis.pos)

    # ── Zeichnen ──────────────────────────────

    def _zeichne_alles(self) -> None:
        self._fenster.fill(FARBE_WEISS)
        self._zeichne_linien()
        self._zeichne_leiste()
        pygame.display.flip()

    def _zeichne_linien(self) -> None:
        """Zeichnet alle Linien; Kreis-Modus verwendet geglättete Punkte."""
        for rohe_linie in self._speicher.alle_linien():
            if not rohe_linie:
                continue

            if self._form == Darstellungsform.LINIE:
                if len(rohe_linie) >= 2:
                    pygame.draw.lines(
                        self._fenster, FARBE_SCHWARZ, False,
                        rohe_linie, max(1, self._dicke // 4),
                    )

            elif self._form == Darstellungsform.KREIS:
                # Punkte glätten, dann Kreise entlang der glatten Kurve zeichnen.
                # Durch die Glättung überlappen die Kreise gleichmäßig → saubere Fläche.
                glatte_linie = glaette_punkte(rohe_linie)
                for punkt in glatte_linie:
                    pygame.draw.circle(self._fenster, FARBE_SCHWARZ, punkt, self._dicke)

            elif self._form == Darstellungsform.QUADRAT:
                seite = self._dicke * 2
                for x, y in rohe_linie:
                    pygame.draw.rect(
                        self._fenster, FARBE_SCHWARZ,
                        pygame.Rect(x - self._dicke, y - self._dicke, seite, seite),
                    )

    def _zeichne_leiste(self) -> None:
        leisten_hoehe = self.KNOPF_HOEHE + self.RAND * 2
        pygame.draw.rect(
            self._fenster, (230, 230, 230),
            pygame.Rect(0, FENSTER_HOEHE - leisten_hoehe, FENSTER_BREITE, leisten_hoehe),
        )

        schrift = pygame.font.SysFont(None, 20)
        label   = schrift.render(f"Dicke: {self._dicke}", True, FARBE_SCHWARZ)
        label_y = FENSTER_HOEHE - self.KNOPF_HOEHE // 2 - self.RAND
        self._fenster.blit(label, label.get_rect(center=(FENSTER_BREITE // 2, label_y)))

        for knopf in [
            self._knopf_linie, self._knopf_kreis, self._knopf_quadrat,
            self._knopf_duenner, self._knopf_dicker,
            self._knopf_reset, self._knopf_speichern,
        ]:
            knopf.zeichne(self._fenster)

    @staticmethod
    def _beenden() -> None:
        pygame.quit()
        sys.exit()

    def starte(self) -> None:
        while True:
            self._verarbeite_ereignisse()
            self._zeichne_alles()
            self._uhr.tick(BILDER_PRO_SEK)


# ─────────────────────────────────────────────
#  Einstiegspunkt
# ─────────────────────────────────────────────
if __name__ == "__main__":
    Zeichenprogramm().starte()