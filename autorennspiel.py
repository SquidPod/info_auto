import pygame
import sys
import json
from pathlib import Path
from enum import Enum, auto

import numpy as np
from scipy.ndimage import uniform_filter1d

# ══════════════════════════════════════════════
#  KONSTANTEN
# ══════════════════════════════════════════════
FENSTER_BREITE   = 1500
FENSTER_HOEHE    = 1000
BILDER_PRO_SEK   = 60

FARBE_WEISS      = (255, 255, 255)
FARBE_SCHWARZ    = (0,   0,   0)
FARBE_GRAU       = (230, 230, 230)
FARBE_DUNKELGRAU = (80,  80,  80)
FARBE_HOVER      = (120, 120, 120)
FARBE_ROT        = (200, 50,  50)
FARBE_ROT_HELL   = (240, 80,  80)
FARBE_BLAU       = (40,  80,  180)
FARBE_BLAU_HELL  = (60,  110, 220)
FARBE_GRUEN      = (30,  140, 80)
FARBE_ORANGE     = (220, 130, 30)

# Strecken-Zeichnen
DICKE_MIN        = 4
DICKE_MAX        = 60
DICKE_SCHRITT    = 4
GLAETTUNGS_STAERKE   = 25
GLAETTUNGS_MINPUNKTE = 4

# Autos
AUTO_BREITE      = 20
AUTO_HOEHE       = 34
AUTO_GESCHW      = 3

# Wie viele Pixel des Auto-Rects auf der Strecke liegen müssen (0.0–1.0)
STRECKEN_SCHWELLE = 0.5   # 50 % der Eckpunkte müssen auf Schwarz liegen

SPEICHERDATEI    = Path(__file__).parent / "strecke.json"


# ══════════════════════════════════════════════
#  ENUM: Programmzustand
# ══════════════════════════════════════════════
class Modus(Enum):
    ZEICHNEN = auto()   # Strecke malen
    PLATZIEREN = auto() # Autos auf die Strecke ziehen
    FAHREN   = auto()   # Rennen


# ══════════════════════════════════════════════
#  ENUM: Darstellungsform (Strecke)
# ══════════════════════════════════════════════
class Darstellungsform(Enum):
    KREIS   = auto()
    QUADRAT = auto()


# ══════════════════════════════════════════════
#  HILFSFUNKTION: Punkte glätten
# ══════════════════════════════════════════════
def glaette_punkte(
    punkte: list[tuple[int, int]],
    staerke: int = GLAETTUNGS_STAERKE,
) -> list[tuple[int, int]]:
    """Glättet Mauskoordinaten mit einem gleitenden Durchschnitt (scipy)."""
    if len(punkte) < GLAETTUNGS_MINPUNKTE:
        return punkte
    xs = np.array([p[0] for p in punkte], dtype=float)
    ys = np.array([p[1] for p in punkte], dtype=float)
    xs_g = uniform_filter1d(xs, size=staerke, mode="nearest")
    ys_g = uniform_filter1d(ys, size=staerke, mode="nearest")
    xs_g[0],  ys_g[0]  = xs[0],  ys[0]
    xs_g[-1], ys_g[-1] = xs[-1], ys[-1]
    return [(int(x), int(y)) for x, y in zip(xs_g, ys_g)]


# ══════════════════════════════════════════════
#  KLASSE: Knopf
# ══════════════════════════════════════════════
class Knopf:
    """Button mit Hover- und Aktiv-Zustand."""

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
        maus    = pygame.mouse.get_pos()
        if self.aktiv:
            farbe = FARBE_GRUEN
        elif self._rect.collidepoint(maus):
            farbe = self._hintergrund_hover
        else:
            farbe = self._hintergrund
        pygame.draw.rect(fenster, farbe, self._rect, border_radius=6)
        text = schrift.render(self._beschriftung, True, FARBE_WEISS)
        fenster.blit(text, text.get_rect(center=self._rect.center))

    def wurde_geklickt(self, ereignis: pygame.event.Event) -> bool:
        return (
            ereignis.type == pygame.MOUSEBUTTONDOWN
            and ereignis.button == 1
            and self._rect.collidepoint(ereignis.pos)
        )


# ══════════════════════════════════════════════
#  KLASSE: Linienspeicher
# ══════════════════════════════════════════════
class Linienspeicher:
    """Verwaltet die Rohdaten der gezeichneten Streckenlinien."""

    def __init__(self) -> None:
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

    def speichere(self, pfad: Path, form: Darstellungsform, dicke: int) -> None:
        daten = {
            "form":   form.name,
            "dicke":  dicke,
            "linien": self._linien,
        }
        pfad.write_text(json.dumps(daten, indent=2), encoding="utf-8")
        print(f"Strecke gespeichert: {pfad.resolve()}")

    def lade(self, pfad: Path) -> tuple["Darstellungsform", int]:
        """Lädt Linien und gibt (form, dicke) zurück."""
        if not pfad.exists():
            return Darstellungsform.KREIS, 20
        daten = json.loads(pfad.read_text(encoding="utf-8"))
        self._linien = [
            [tuple(p) for p in linie]
            for linie in daten.get("linien", [])
        ]
        form  = Darstellungsform[daten.get("form", "KREIS")]
        dicke = daten.get("dicke", 20)
        print(f"Strecke geladen: {pfad.resolve()} ({len(self._linien)} Linien)")
        return form, dicke


# ══════════════════════════════════════════════
#  KLASSE: Strecke
#  Rendert die Linien auf eine eigene Surface,
#  die dann als Kollisionsmaske genutzt wird.
# ══════════════════════════════════════════════
class Strecke:
    """Hält die gerenderte Strecken-Surface und prüft Kollisionen."""

    def __init__(self) -> None:
        # Surface in Fenstergröße, weißer Hintergrund
        self._surface = pygame.Surface(
            (FENSTER_BREITE, FENSTER_HOEHE - 60)  # 60 px für die Leiste
        )
        self._surface.fill(FARBE_WEISS)

    def rendere(
        self,
        linien: list[list[tuple[int, int]]],
        form: Darstellungsform,
        dicke: int,
    ) -> None:
        """Zeichnet alle Linien auf die interne Surface (wird als Maske genutzt)."""
        self._surface.fill(FARBE_WEISS)
        for rohe_linie in linien:
            if not rohe_linie:
                continue
            punkte = glaette_punkte(rohe_linie) if form == Darstellungsform.KREIS else rohe_linie
            if form == Darstellungsform.KREIS:
                for p in punkte:
                    pygame.draw.circle(self._surface, FARBE_SCHWARZ, p, dicke)
            else:
                seite = dicke * 2
                for x, y in punkte:
                    pygame.draw.rect(
                        self._surface, FARBE_SCHWARZ,
                        pygame.Rect(x - dicke, y - dicke, seite, seite),
                    )

    def auf_strecke(self, rect: pygame.Rect) -> bool:
        """Gibt True zurück wenn genügend Eckpunkte des Rects auf Schwarz liegen."""
        # Wir testen 9 Punkte (4 Ecken + 4 Kantenmittelpunkte + Mitte)
        testpunkte = [
            (rect.left,   rect.top),
            (rect.right,  rect.top),
            (rect.left,   rect.bottom),
            (rect.right,  rect.bottom),
            (rect.centerx, rect.top),
            (rect.centerx, rect.bottom),
            (rect.left,   rect.centery),
            (rect.right,  rect.centery),
            (rect.centerx, rect.centery),
        ]
        breite, hoehe = self._surface.get_size()
        treffer = 0
        for x, y in testpunkte:
            # Clamp damit wir nicht außerhalb der Surface lesen
            x = max(0, min(x, breite - 1))
            y = max(0, min(y, hoehe - 1))
            farbe = self._surface.get_at((x, y))[:3]
            if farbe == FARBE_SCHWARZ:
                treffer += 1
        return (treffer / len(testpunkte)) >= STRECKEN_SCHWELLE

    @property
    def surface(self) -> pygame.Surface:
        return self._surface


# ══════════════════════════════════════════════
#  KLASSE: Auto (pygame.sprite.Sprite)
# ══════════════════════════════════════════════
class Auto(pygame.sprite.Sprite):
    """Steuerbares Auto – fährt nur auf der Strecke.

    Im Platzierungsmodus kann es per Drag & Drop verschoben werden.
    Im Fahrmodus reagiert es auf Tastatureingaben; bewegt es sich
    von der Strecke weg, wird die Bewegung rückgängig gemacht.
    """

    def __init__(
        self,
        x: int,
        y: int,
        farbe: tuple[int, int, int],
        steuerung: dict[str, int],
        name: str,
    ) -> None:
        super().__init__()
        self._farbe     = farbe
        self._steuerung = steuerung
        self.name       = name

        self.image = pygame.Surface((AUTO_BREITE, AUTO_HOEHE), pygame.SRCALPHA)
        self._zeichne_auto()
        self.rect  = self.image.get_rect(center=(x, y))

        self._geschwindigkeit = AUTO_GESCHW
        self._wird_gezogen    = False
        self._zieh_offset     = (0, 0)   # Mausversatz beim Drag-Start
        self.auf_strecke_platziert = False  # wird gesetzt wenn Drop auf gültiger Stelle

    # ── Darstellung ───────────────────────────

    def _zeichne_auto(self) -> None:
        self.image.fill((0, 0, 0, 0))
        pygame.draw.rect(
            self.image, self._farbe,
            (0, 0, AUTO_BREITE, AUTO_HOEHE), border_radius=5,
        )
        scheibenfarbe = (180, 220, 255)
        pygame.draw.rect(self.image, scheibenfarbe, (3, 4,  AUTO_BREITE - 6, 10), border_radius=2)
        pygame.draw.rect(self.image, scheibenfarbe, (3, AUTO_HOEHE - 14, AUTO_BREITE - 6, 10), border_radius=2)

    # ── Drag & Drop ───────────────────────────

    def starte_ziehen(self, mauspos: tuple[int, int]) -> None:
        self._wird_gezogen = True
        self._zieh_offset  = (
            self.rect.x - mauspos[0],
            self.rect.y - mauspos[1],
        )

    def ziehe_zu(self, mauspos: tuple[int, int]) -> None:
        if self._wird_gezogen:
            self.rect.x = mauspos[0] + self._zieh_offset[0]
            self.rect.y = mauspos[1] + self._zieh_offset[1]

    def beende_ziehen(self, strecke: "Strecke") -> bool:
        """Lässt das Auto los. Gibt True zurück wenn es auf der Strecke landet."""
        self._wird_gezogen = False
        self.auf_strecke_platziert = strecke.auf_strecke(self.rect)
        return self.auf_strecke_platziert

    @property
    def wird_gezogen(self) -> bool:
        return self._wird_gezogen

    # ── Fahren ────────────────────────────────

    def update(self, strecke: "Strecke" = None) -> None:  # type: ignore[override]
        """Bewegt das Auto anhand der Tastatureingaben; rollt zurück wenn off-track."""
        if strecke is None:
            return  # im Zeichenmodus nichts tun

        tasten = pygame.key.get_pressed()
        dx = dy = 0

        if tasten[self._steuerung["oben"]]:
            dy -= self._geschwindigkeit
        if tasten[self._steuerung["unten"]]:
            dy += self._geschwindigkeit
        if tasten[self._steuerung["links"]]:
            dx -= self._geschwindigkeit
        if tasten[self._steuerung["rechts"]]:
            dx += self._geschwindigkeit

        if dx == 0 and dy == 0:
            return

        alte_pos = self.rect.topleft

        # x und y getrennt testen → erlaubt Gleiten an der Streckenwand
        self.rect.x += dx
        if not strecke.auf_strecke(self.rect):
            self.rect.x = alte_pos[0]

        self.rect.y += dy
        if not strecke.auf_strecke(self.rect):
            self.rect.y = alte_pos[1]

        # Fensterrand
        spielfeld = pygame.Rect(0, 0, FENSTER_BREITE, FENSTER_HOEHE - 60)
        self.rect.clamp_ip(spielfeld)


# ══════════════════════════════════════════════
#  KLASSE: Hauptspiel
# ══════════════════════════════════════════════
class Spiel:
    """Vereint Streckeneditor und Rennspiel in einem Programm.

    Modus-Reihenfolge
    -----------------
    ZEICHNEN   → Strecke malen, speichern, Reset
    PLATZIEREN → Autos per Drag & Drop auf die Strecke setzen
    FAHREN     → WASD / Pfeiltasten, Autos bleiben auf der Strecke
    """

    LEISTEN_HOEHE = 60
    KNOPF_B       = 120
    KNOPF_H       = 32
    RAND          = 14
    AB            = 8

    def __init__(self) -> None:
        pygame.init()
        self._fenster = pygame.display.set_mode((FENSTER_BREITE, FENSTER_HOEHE))
        pygame.display.set_caption("Autorennspiel – Streckeneditor")
        self._uhr = pygame.time.Clock()

        # Strecke
        self._speicher = Linienspeicher()
        self._strecke  = Strecke()
        form_geladen, dicke_geladen = self._speicher.lade(SPEICHERDATEI)
        self._form     = form_geladen
        self._dicke    = dicke_geladen
        self._strecke.rendere(self._speicher.alle_linien(), self._form, self._dicke)

        # Autos
        self._autos = pygame.sprite.Group()
        self._auto1, self._auto2 = self._erstelle_autos()
        self._gezogenes_auto: Auto | None = None

        # Zustand
        self._modus           = Modus.ZEICHNEN
        self._zeichnet_gerade = False

        self._erstelle_knoepfe()
        self._aktualisiere_knoepfe()

    # ── Initialisierung ───────────────────────

    def _erstelle_autos(self) -> tuple[Auto, Auto]:
        steuerung1 = {"oben": pygame.K_w, "unten": pygame.K_s,
                      "links": pygame.K_a, "rechts": pygame.K_d}
        steuerung2 = {"oben": pygame.K_UP, "unten": pygame.K_DOWN,
                      "links": pygame.K_LEFT, "rechts": pygame.K_RIGHT}
        auto1 = Auto(150, 100, FARBE_ROT,  steuerung1, "Spieler 1")
        auto2 = Auto(200, 100, FARBE_BLAU, steuerung2, "Spieler 2")
        self._autos.add(auto1, auto2)
        return auto1, auto2

    def _erstelle_knoepfe(self) -> None:
        y  = FENSTER_HOEHE - self.KNOPF_H - self.RAND
        r  = self.RAND
        kb = self.KNOPF_B
        ab = self.AB

        # ── rechts: Modus-Wechsel & Reset & Speichern ──
        x = FENSTER_BREITE - r - kb
        self._knopf_speichern  = Knopf(pygame.Rect(x, y, kb, self.KNOPF_H), "💾 Speichern")
        x -= ab + kb
        self._knopf_reset      = Knopf(
            pygame.Rect(x, y, kb, self.KNOPF_H), "🗑 Reset Strecke",
            hintergrund=FARBE_ROT, hintergrund_hover=FARBE_ROT_HELL,
        )
        x -= ab + kb
        self._knopf_fahren     = Knopf(
            pygame.Rect(x, y, kb, self.KNOPF_H), "🏁 Fahren",
            hintergrund=FARBE_GRUEN, hintergrund_hover=(50, 180, 100),
        )
        x -= ab + kb
        self._knopf_platzieren = Knopf(
            pygame.Rect(x, y, kb, self.KNOPF_H), "🚗 Platzieren",
            hintergrund=FARBE_ORANGE, hintergrund_hover=(240, 160, 60),
        )
        x -= ab + kb
        self._knopf_zeichnen   = Knopf(
            pygame.Rect(x, y, kb, self.KNOPF_H), "✏ Zeichnen",
        )

        # ── links: Strecken-Optionen (nur im Zeichenmodus sichtbar) ──
        self._knopf_kreis   = Knopf(pygame.Rect(r,          y, kb, self.KNOPF_H), "● Kreis")
        self._knopf_quadrat = Knopf(pygame.Rect(r + kb + ab,y, kb, self.KNOPF_H), "■ Quadrat")
        pfeil = 36
        mx = r + 2*(kb + ab) + 60
        self._knopf_duenner = Knopf(
            pygame.Rect(mx, y, pfeil, self.KNOPF_H), "−",
            hintergrund=FARBE_BLAU, hintergrund_hover=FARBE_BLAU_HELL,
        )
        self._knopf_dicker  = Knopf(
            pygame.Rect(mx + pfeil + 6, y, pfeil, self.KNOPF_H), "+",
            hintergrund=FARBE_BLAU, hintergrund_hover=FARBE_BLAU_HELL,
        )

        self._form_knoepfe = {
            Darstellungsform.KREIS:   self._knopf_kreis,
            Darstellungsform.QUADRAT: self._knopf_quadrat,
        }

    def _aktualisiere_knoepfe(self) -> None:
        """Setzt Aktiv-Zustände aller Buttons passend zum aktuellen Modus."""
        self._knopf_zeichnen.aktiv   = (self._modus == Modus.ZEICHNEN)
        self._knopf_platzieren.aktiv = (self._modus == Modus.PLATZIEREN)
        self._knopf_fahren.aktiv     = (self._modus == Modus.FAHREN)
        for form, knopf in self._form_knoepfe.items():
            knopf.aktiv = (form == self._form)

    def _wechsle_modus(self, neuer_modus: Modus) -> None:
        self._modus = neuer_modus
        self._aktualisiere_knoepfe()
        titel = {
            Modus.ZEICHNEN:   "Autorennspiel – Streckeneditor",
            Modus.PLATZIEREN: "Autorennspiel – Autos platzieren",
            Modus.FAHREN:     "Autorennspiel – Fahren!",
        }
        pygame.display.set_caption(titel[neuer_modus])

    # ── Ereignisverarbeitung ──────────────────

    def _verarbeite_ereignisse(self) -> None:
        for ereignis in pygame.event.get():
            if ereignis.type == pygame.QUIT:
                self._beenden()
            if ereignis.type == pygame.KEYDOWN and ereignis.key == pygame.K_ESCAPE:
                self._beenden()

            if ereignis.type == pygame.MOUSEBUTTONDOWN and ereignis.button == 1:
                self._verarbeite_klick(ereignis)
            if ereignis.type == pygame.MOUSEMOTION:
                self._verarbeite_mausbewegung(ereignis)
            if ereignis.type == pygame.MOUSEBUTTONUP and ereignis.button == 1:
                self._verarbeite_loslassen(ereignis)

    def _verarbeite_klick(self, ereignis: pygame.event.Event) -> None:
        pos = ereignis.pos

        # ── Modus-Buttons (immer aktiv) ───────
        if self._knopf_zeichnen.wurde_geklickt(ereignis):
            self._wechsle_modus(Modus.ZEICHNEN)
            return
        if self._knopf_platzieren.wurde_geklickt(ereignis):
            self._wechsle_modus(Modus.PLATZIEREN)
            return
        if self._knopf_fahren.wurde_geklickt(ereignis):
            self._wechsle_modus(Modus.FAHREN)
            return
        if self._knopf_speichern.wurde_geklickt(ereignis):
            self._speicher.speichere(SPEICHERDATEI, self._form, self._dicke)
            return
        if self._knopf_reset.wurde_geklickt(ereignis):
            self._speicher.leere()
            self._strecke.rendere([], self._form, self._dicke)
            return

        # ── Zeichenmodus-Buttons ───────────────
        if self._modus == Modus.ZEICHNEN:
            if self._knopf_dicker.wurde_geklickt(ereignis):
                self._dicke = min(self._dicke + DICKE_SCHRITT, DICKE_MAX)
                return
            if self._knopf_duenner.wurde_geklickt(ereignis):
                self._dicke = max(self._dicke - DICKE_SCHRITT, DICKE_MIN)
                return
            for form, knopf in self._form_knoepfe.items():
                if knopf.wurde_geklickt(ereignis):
                    self._form = form
                    self._aktualisiere_knoepfe()
                    # Strecke mit neuer Form neu rendern
                    self._strecke.rendere(self._speicher.alle_linien(), self._form, self._dicke)
                    return
            # Zeichenfläche → neue Linie starten
            if pos[1] < FENSTER_HOEHE - self.LEISTEN_HOEHE:
                self._zeichnet_gerade = True
                self._speicher.starte_neue_linie(pos)

        # ── Platziermodus: Drag starten ────────
        elif self._modus == Modus.PLATZIEREN:
            for auto in [self._auto1, self._auto2]:
                if auto.rect.collidepoint(pos):
                    auto.starte_ziehen(pos)
                    self._gezogenes_auto = auto
                    break

    def _verarbeite_mausbewegung(self, ereignis: pygame.event.Event) -> None:
        if self._modus == Modus.ZEICHNEN and self._zeichnet_gerade:
            if ereignis.pos[1] < FENSTER_HOEHE - self.LEISTEN_HOEHE:
                self._speicher.fuege_punkt_hinzu(ereignis.pos)
                # Strecke live neu rendern damit die Kollisionsmaske aktuell ist
                self._strecke.rendere(self._speicher.alle_linien(), self._form, self._dicke)
        elif self._modus == Modus.PLATZIEREN and self._gezogenes_auto:
            self._gezogenes_auto.ziehe_zu(ereignis.pos)

    def _verarbeite_loslassen(self, ereignis: pygame.event.Event) -> None:
        if self._modus == Modus.ZEICHNEN:
            self._zeichnet_gerade = False

        elif self._modus == Modus.PLATZIEREN and self._gezogenes_auto:
            auf_strecke = self._gezogenes_auto.beende_ziehen(self._strecke)
            if not auf_strecke:
                # Visuelles Feedback: kurz rot blinken (einfach Konsolenhinweis)
                print(f"{self._gezogenes_auto.name}: nicht auf der Strecke – bitte neu platzieren!")
            self._gezogenes_auto = None

    # ── Zeichnen ──────────────────────────────

    def _zeichne_alles(self) -> None:
        # Strecken-Surface als Hintergrund
        self._fenster.blit(self._strecke.surface, (0, 0))

        # Autos
        self._autos.draw(self._fenster)

        # Warnung wenn Auto nicht auf Strecke (Platziermodus)
        if self._modus == Modus.PLATZIEREN:
            self._zeichne_platzier_hinweise()

        # Leiste
        self._zeichne_leiste()

        pygame.display.flip()

    def _zeichne_platzier_hinweise(self) -> None:
        """Zeichnet einen farbigen Rahmen um Autos je nach Platzierungsstatus."""
        for auto in [self._auto1, self._auto2]:
            farbe = FARBE_GRUEN if auto.auf_strecke_platziert else FARBE_ROT
            pygame.draw.rect(self._fenster, farbe, auto.rect.inflate(4, 4), 2, border_radius=4)

    def _zeichne_leiste(self) -> None:
        """Zeichnet die untere Steuerungsleiste."""
        leiste = pygame.Rect(0, FENSTER_HOEHE - self.LEISTEN_HOEHE,
                             FENSTER_BREITE, self.LEISTEN_HOEHE)
        pygame.draw.rect(self._fenster, FARBE_GRAU, leiste)

        schrift = pygame.font.SysFont(None, 20)

        # Modus-abhängige Info in der Mitte
        if self._modus == Modus.ZEICHNEN:
            info = f"Dicke: {self._dicke}   |   Linke Maustaste = Strecke malen"
        elif self._modus == Modus.PLATZIEREN:
            s1 = "✓" if self._auto1.auf_strecke_platziert else "✗"
            s2 = "✓" if self._auto2.auf_strecke_platziert else "✗"
            info = f"Rot {s1}  Blau {s2}   |   Autos auf die schwarze Strecke ziehen"
        else:
            info = "WASD = Rot   |   Pfeiltasten = Blau   |   ESC = Beenden"

        label = schrift.render(info, True, FARBE_SCHWARZ)
        self._fenster.blit(
            label,
            label.get_rect(center=(FENSTER_BREITE // 2, FENSTER_HOEHE - self.LEISTEN_HOEHE // 2)),
        )

        # Alle Buttons zeichnen
        alle_knoepfe = [
            self._knopf_zeichnen, self._knopf_platzieren, self._knopf_fahren,
            self._knopf_reset, self._knopf_speichern,
        ]
        if self._modus == Modus.ZEICHNEN:
            alle_knoepfe += [
                self._knopf_kreis, self._knopf_quadrat,
                self._knopf_duenner, self._knopf_dicker,
            ]
        for knopf in alle_knoepfe:
            knopf.zeichne(self._fenster)

    # ── Game-Loop ─────────────────────────────

    @staticmethod
    def _beenden() -> None:
        pygame.quit()
        sys.exit()

    def starte(self) -> None:
        while True:
            self._verarbeite_ereignisse()

            # Autos nur im Fahrmodus bewegen, und nur wenn beide platziert sind
            if self._modus == Modus.FAHREN:
                self._autos.update(self._strecke)

            self._zeichne_alles()
            self._uhr.tick(BILDER_PRO_SEK)


# ══════════════════════════════════════════════
#  EINSTIEGSPUNKT
# ══════════════════════════════════════════════
if __name__ == "__main__":
    Spiel().starte()
