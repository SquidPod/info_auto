import sys
import json
from pathlib import Path
import pygame
import numpy as NoneType  # für Typprüfung optional, nicht zwingend
import numpy as np
from scipy.ndimage import uniform_filter1d

# ── KONSTANTEN ────────────────────────────────────────────────────────
FENSTER_BREITE, FENSTER_HOEHE, BILDER_PRO_SEK = 1500, 1000, 60

FARBE_WEISS = (255, 255, 255)
FARBE_SCHWARZ = (0, 0, 0)
FARBE_GRAU = (230, 230, 230)
FARBE_DUNKELGRAU = (80, 80, 80)
FARBE_HOVER = (120, 120, 120)
FARBE_ROT = (200, 50, 50)
FARBE_ROT_HELL = (240, 80, 80)
FARBE_BLAU = (40, 80, 180)
FARBE_BLAU_HELL = (60, 110, 220)
FARBE_GRUEN = (30, 140, 80)
FARBE_ORANGE = (220, 130, 30)

DICKE_MIN, DICKE_MAX, DICKE_SCHRITT = 4, 60, 4
GLAETTUNGS_STAERKE, GLAETTUNGS_MINPUNKTE = 25, 4

AUTO_BREITE, AUTO_HOEHE, AUTO_GESCHW = 20, 34, 3
STRECKEN_SCHWELLE = 0.5  # 50% der Autopunkte müssen auf der Strecke sein
SPEICHERDATEI = Path(__file__).parent / "strecke.json"


class Modus:
    ZEICHNEN, PLATZIEREN, FAHREN = range(3)


def glaette_punkte(punkte: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Glättet die gezeichneten Punkte via gleitendem Durchschnitt."""
    if len(punkte) < GLAETTUNGS_MINPUNKTE:
        return punkte
    xs = np.array([p[0] for p in punkte], dtype=float)
    ys = np.array([p[1] for p in punkte], dtype=float)
    xs_g = uniform_filter1d(xs, size=GLAETTUNGS_STAERKE, mode="nearest")
    ys_g = uniform_filter1d(ys, size=GLAETTUNGS_STAERKE, mode="nearest")
    xs_g[0], ys_g[0], xs_g[-1], ys_g[-1] = xs[0], ys[0], xs[-1], ys[-1]
    return [(int(x), int(y)) for x, y in zip(xs_g, ys_g)]


# ── UI ELEMENTE ───────────────────────────────────────────────────────
class Knopf:
    """Interaktiver Button für das UI-Menü."""

    def __init__(self, rect: pygame.Rect, text: str, bg: tuple = FARBE_DUNKELGRAU, hover_bg: tuple = FARBE_HOVER):
        self.rect = rect
        self.text = text
        self.bg = bg
        self.hover_bg = hover_bg
        self.aktiv = False

    def zeichne(self, fenster: pygame.Surface, schrift: pygame.font.Font):
        farbe = FARBE_GRUEN if self.aktiv else (
            self.hover_bg if self.rect.collidepoint(pygame.mouse.get_pos()) else self.bg)
        pygame.draw.rect(fenster, farbe, self.rect, border_radius=6)
        txt_surf = schrift.render(self.text, True, FARBE_WEISS)
        fenster.blit(txt_surf, txt_surf.get_rect(center=self.rect.center))

    def wurde_geklickt(self, event: pygame.event.Event) -> bool:
        return event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos)


# ── SPIELOBJEKTE ──────────────────────────────────────────────────────
class Strecke:
    """Verwaltet das Zeichnen und die Kollisionsabfrage der Rennstrecke."""

    def __init__(self):
        self.surface = pygame.Surface((FENSTER_BREITE, FENSTER_HOEHE - 60))
        self.linien: list[list[tuple[int, int]]] = []
        self.leere()

    def leere(self):
        self.linien.clear()
        self.surface.fill(FARBE_WEISS)

    def rendere(self, dicke: int):
        """Zeichnet alle geglätteten Linien als Kreisbahnen auf die Masken-Surface."""
        self.surface.fill(FARBE_WEISS)
        for linie in self.linien:
            if linie:
                for p in glaette_punkte(linie):
                    pygame.draw.circle(self.surface, FARBE_SCHWARZ, p, dicke)

    def auf_strecke(self, rect: pygame.Rect) -> bool:
        """Prüft über 9 Punkte, ob das Auto ausreichend auf der Strecke steht."""
        testpunkte = [
            rect.topleft, rect.topright, rect.bottomleft, rect.bottomright,
            rect.midtop, rect.midbottom, rect.midleft, rect.midright, rect.center
        ]
        w, h = self.surface.get_size()
        treffer = sum(
            1 for x, y in testpunkte
            if self.surface.get_at((max(0, min(x, w - 1)), max(0, min(y, h - 1))))[:3] == FARBE_SCHWARZ
        )
        return (treffer / len(testpunkte)) >= STRECKEN_SCHWELLE


class Auto(pygame.sprite.Sprite):
    """Fahrzeug mit Drag & Drop Funktion und Fahr-Logik."""

    def __init__(self, x: int, y: int, farbe: tuple, steuerung: dict, name: str):
        super().__init__()
        self.farbe, self.steuerung, self.name = farbe, steuerung, name
        self.image = pygame.Surface((AUTO_BREITE, AUTO_HOEHE), pygame.SRCALPHA)

        pygame.draw.rect(self.image, farbe, (0, 0, AUTO_BREITE, AUTO_HOEHE), border_radius=5)
        for y_pos in (4, AUTO_HOEHE - 14):
            pygame.draw.rect(self.image, (180, 220, 255), (3, y_pos, AUTO_BREITE - 6, 10), border_radius=2)

        self.rect = self.image.get_rect(center=(x, y))
        self.offset = (0, 0)
        self.wird_gezogen = False
        self.auf_strecke_platziert = False

    def update(self, strecke: Strecke):
        """Bewegt das Auto per Tastatur und fängt Kollisionen mit dem Streckenrand ab."""
        tasten = pygame.key.get_pressed()
        dx = (tasten[self.steuerung["rechts"]] - tasten[self.steuerung["links"]]) * AUTO_GESCHW
        dy = (tasten[self.steuerung["unten"]] - tasten[self.steuerung["oben"]]) * AUTO_GESCHW

        if dx == 0 and dy == 0: return
        alte_pos = self.rect.topleft

        # Getrennte Achsenprüfung für sanftes Entlanggleiten an Kurvenwänden
        self.rect.x += dx
        if not strecke.auf_strecke(self.rect): self.rect.x = alte_pos[0]

        self.rect.y += dy
        if not strecke.auf_strecke(self.rect): self.rect.y = alte_pos[1]

        self.rect.clamp_ip(pygame.Rect(0, 0, FENSTER_BREITE, FENSTER_HOEHE - 60))


# ── HAUPTSPIEL ────────────────────────────────────────────────────────
class Spiel:
    def __init__(self):
        pygame.init()
        self.fenster = pygame.display.set_mode((FENSTER_BREITE, FENSTER_HOEHE))
        self.uhr = pygame.time.Clock()
        self.schrift = pygame.font.SysFont(None, 20)

        self.strecke = Strecke()
        self.dicke = 20
        self.autos = pygame.sprite.Group()
        self._init_autos()
        self.lade_strecke()

        self.modus = Modus.ZEICHNEN
        self.zeichnet_gerade = False
        self.gezogenes_auto = None

        self._init_knoepfe()
        self._update_knopf_states()

    def _init_autos(self):
        self.auto1 = Auto(150, 100, FARBE_ROT,
                          {"oben": pygame.K_w, "unten": pygame.K_s, "links": pygame.K_a, "rechts": pygame.K_d},
                          "Spieler 1")
        self.auto2 = Auto(200, 100, FARBE_BLAU, {"oben": pygame.K_UP, "unten": pygame.K_DOWN, "links": pygame.K_LEFT,
                                                 "rechts": pygame.K_RIGHT}, "Spieler 2")
        self.autos.add(self.auto1, self.auto2)

    def _init_knoepfe(self):
        """Erstellt das kompakte Menü in der unteren Leiste."""
        y = FENSTER_HOEHE - 46
        self.knoepfe = {
            "zeichnen": Knopf(pygame.Rect(14, y, 120, 32), "✏ Zeichnen"),
            "platzieren": Knopf(pygame.Rect(142, y, 120, 32), "🚗 Platzieren", FARBE_ORANGE, (240, 160, 60)),
            "fahren": Knopf(pygame.Rect(270, y, 120, 32), "🏁 Fahren", FARBE_GRUEN, (50, 180, 100)),
            "duenner": Knopf(pygame.Rect(406, y, 36, 32), "−", FARBE_BLAU, FARBE_BLAU_HELL),
            "dicker": Knopf(pygame.Rect(448, y, 36, 32), "+", FARBE_BLAU, FARBE_BLAU_HELL),
            "reset": Knopf(pygame.Rect(1216, y, 120, 32), "🗑 Reset", FARBE_ROT, FARBE_ROT_HELL),
            "speichern": Knopf(pygame.Rect(1344, y, 120, 32), "💾 Speichern")
        }

    def _update_knopf_states(self):
        self.knoepfe["zeichnen"].aktiv = (self.modus == Modus.ZEICHNEN)
        self.knoepfe["platzieren"].aktiv = (self.modus == Modus.PLATZIEREN)
        self.knoepfe["fahren"].aktiv = (self.modus == Modus.FAHREN)

    def wechsle_modus(self, neuer_modus: int):
        self.modus = neuer_modus
        self._update_knopf_states()
        titel = {Modus.ZEICHNEN: "Editor", Modus.PLATZIEREN: "Autos Platzieren", Modus.FAHREN: "Rennen!"}
        pygame.display.set_caption(f"Autorennspiel – {titel[neuer_modus]}")

    def speichere_strecke(self):
        json_daten = {"dicke": self.dicke, "linien": self.strecke.linien}
        SPEICHERDATEI.write_text(json.dumps(json_daten), encoding="utf-8")

    def lade_strecke(self):
        if not SPEICHERDATEI.exists(): return
        daten = json.loads(SPEICHERDATEI.read_text(encoding="utf-8"))
        self.strecke.linien = [[tuple(p) for p in l] for l in daten.get("linien", [])]
        self.dicke = daten.get("dicke", 20)
        self.strecke.rendere(self.dicke)

    def verarbeite_ereignisse(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE):
                pygame.quit();
                sys.exit()

            if ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                self._klick_abfangen(ev)
            elif ev.type == pygame.MOUSEMOTION:
                self._maus_bewegen(ev)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1:
                self.zeichnet_gerade = False
                if self.gezogenes_auto:
                    self.gezogenes_auto.wird_gezogen = False
                    self.gezogenes_auto.auf_strecke_platziert = self.strecke.auf_strecke(self.gezogenes_auto.rect)
                    self.gezogenes_auto = None

    def _klick_abfangen(self, ev: pygame.event.Event):
        if self.knoepfe["zeichnen"].wurde_geklickt(ev):
            self.wechsle_modus(Modus.ZEICHNEN)
        elif self.knoepfe["platzieren"].wurde_geklickt(ev):
            self.wechsle_modus(Modus.PLATZIEREN)
        elif self.knoepfe["fahren"].wurde_geklickt(ev):
            self.wechsle_modus(Modus.FAHREN)
        elif self.knoepfe["speichern"].wurde_geklickt(ev):
            self.speichere_strecke()
        elif self.knoepfe["reset"].wurde_geklickt(ev):
            self.strecke.leere()

        elif self.modus == Modus.ZEICHNEN:
            if self.knoepfe["dicker"].wurde_geklickt(ev):
                self.dicke = min(self.dicke + DICKE_SCHRITT, DICKE_MAX)
                self.strecke.rendere(self.dicke)
            elif self.knoepfe["duenner"].wurde_geklickt(ev):
                self.dicke = max(self.dicke - DICKE_SCHRITT, DICKE_MIN)
                self.strecke.rendere(self.dicke)
            elif ev.pos[1] < FENSTER_HOEHE - 60:
                self.zeichnet_gerade = True
                self.strecke.linien.append([ev.pos])

        elif self.modus == Modus.PLATZIEREN:
            for auto in self.autos:
                if auto.rect.collidepoint(ev.pos):
                    auto.wird_gezogen = True
                    auto.offset = (auto.rect.x - ev.pos[0], auto.rect.y - ev.pos[1])
                    self.gezogenes_auto = auto
                    break

    def _maus_bewegen(self, ev: pygame.event.Event):
        if self.modus == Modus.ZEICHNEN and self.zeichnet_gerade and ev.pos[1] < FENSTER_HOEHE - 60:
            self.strecke.linien[-1].append(ev.pos)
            self.strecke.rendere(self.dicke)
        elif self.modus == Modus.PLATZIEREN and self.gezogenes_auto and self.gezogenes_auto.wird_gezogen:
            self.gezogenes_auto.rect.x = ev.pos[0] + self.gezogenes_auto.offset[0]
            self.gezogenes_auto.rect.y = ev.pos[1] + self.gezogenes_auto.offset[1]

    def zeichne_alles(self):
        self.fenster.blit(self.strecke.surface, (0, 0))
        self.autos.draw(self.fenster)

        if self.modus == Modus.PLATZIEREN:
            for a in self.autos:
                pygame.draw.rect(self.fenster, FARBE_GRUEN if a.auf_strecke_platziert else FARBE_ROT,
                                 a.rect.inflate(4, 4), 2, border_radius=4)

        # Vereinfachte Statusleiste
        pygame.draw.rect(self.fenster, FARBE_GRAU, (0, FENSTER_HOEHE - 60, FENSTER_BREITE, 60))

        if self.modus == Modus.ZEICHNEN:
            info = f"Dicke: {self.dicke} | Linksklick & Ziehen = Strecke malen"
        elif self.modus == Modus.PLATZIEREN:
            info = f"Rot: {'✓' if self.auto1.auf_strecke_platziert else '✗'} | Blau: {'✓' if self.auto2.auf_strecke_platziert else '✗'} | Autos auf die Strecke ziehen"
        else:
            info = "Spieler 1: WASD | Spieler 2: Pfeiltasten"

        txt = self.schrift.render(info, True, FARBE_SCHWARZ)
        self.fenster.blit(txt, txt.get_rect(center=(FENSTER_BREITE // 2, FENSTER_HOEHE - 30)))

        # Nur relevante Knöpfe rendern
        for name, knopf in self.knoepfe.items():
            if self.modus == Modus.ZEICHNEN or name not in ["duenner", "dicker"]:

                # Wenn es der Fahren-Button ist und die Autos nicht bereit sind -> Farbe temporär ändern
                if name == "fahren" and not (self.auto1.auf_strecke_platziert and self.auto2.auf_strecke_platziert):
                    alter_bg = knopf.bg
                    knopf.bg = (40, 40, 40)  # Dunkles Grau für "deaktiviert"
                    knopf.zeichne(self.fenster, self.schrift)
                    knopf.bg = alter_bg  # Zurücksetzen für den Hover-Effekt
                else:
                    knopf.zeichne(self.fenster, self.schrift)

        pygame.display.flip()

    def starte(self):
        while True:
            self.verarbeite_ereignisse()
            if self.modus == Modus.FAHREN: self.autos.update(self.strecke)
            self.zeichne_alles()
            self.uhr.tick(BILDER_PRO_SEK)


if __name__ == "__main__":
    Spiel().starte()
