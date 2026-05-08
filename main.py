import pygame
import sys

# ─────────────────────────────────────────────
#  Konstanten
# ─────────────────────────────────────────────
FENSTER_BREITE  = 1024
FENSTER_HOEHE   = 720
BILDER_PRO_SEK  = 60

FARBE_GRUEN     = (34, 139, 34)
FARBE_ROT       = (220, 50, 50)
FARBE_BLAU      = (50, 100, 220)
FARBE_WEISS     = (255, 255, 255)
FARBE_SCHWARZ   = (0, 0, 0)

AUTO_BREITE     = 40
AUTO_HOEHE      = 70
AUTO_GESCHW     = 4   # Pixel pro Frame


# ─────────────────────────────────────────────
#  Auto-Klasse
# ─────────────────────────────────────────────
class Auto(pygame.sprite.Sprite):
    """Ein steuerbares Auto als pygame.sprite.Sprite.

    Parameter
    ---------
    x, y        : Startposition (Mittelpunkt)
    farbe       : Farbe der Karosserie als RGB-Tupel
    steuerung   : Dict mit den vier Steuertasten
                  {'oben': key, 'unten': key, 'links': key, 'rechts': key}
    """

    def __init__(
        self,
        x: int,
        y: int,
        farbe: tuple[int, int, int],
        steuerung: dict[str, int],
    ) -> None:
        super().__init__()

        # Surface für das Auto-Bild (transparent als Basis)
        self.image = pygame.Surface((AUTO_BREITE, AUTO_HOEHE), pygame.SRCALPHA)
        self._farbe = farbe
        self._zeichne_auto()

        # Positions-Rect (Mittelpunkt = Startposition)
        self.rect = self.image.get_rect(center=(x, y))

        # Tastenbelegung für dieses Auto
        self._steuerung = steuerung

        # Bewegungsgeschwindigkeit in Pixel pro Frame
        self._geschwindigkeit = AUTO_GESCHW

    # ── private ──────────────────────────────

    def _zeichne_auto(self) -> None:
        """Zeichnet Karosserie und Scheiben auf die eigene Surface."""
        self.image.fill((0, 0, 0, 0))  # alles transparent zurücksetzen

        # Karosserie
        pygame.draw.rect(
            self.image, self._farbe,
            (0, 0, AUTO_BREITE, AUTO_HOEHE),
            border_radius=6,
        )

        # Scheiben (Windschutzscheibe vorne, Heckscheibe hinten)
        scheibenfarbe = (180, 220, 255)
        pygame.draw.rect(
            self.image, scheibenfarbe,
            (6, 8, AUTO_BREITE - 12, 16),
            border_radius=3,
        )
        pygame.draw.rect(
            self.image, scheibenfarbe,
            (6, AUTO_HOEHE - 24, AUTO_BREITE - 12, 16),
            border_radius=3,
        )

    # ── public ───────────────────────────────

    def aktualisiere(self) -> None:
        """Liest Tasteneingaben und bewegt das Auto entsprechend."""
        gedrueckte_tasten = pygame.key.get_pressed()

        bewegung_x = 0
        bewegung_y = 0

        if gedrueckte_tasten[self._steuerung['oben']]:
            bewegung_y -= self._geschwindigkeit
        if gedrueckte_tasten[self._steuerung['unten']]:
            bewegung_y += self._geschwindigkeit
        if gedrueckte_tasten[self._steuerung['links']]:
            bewegung_x -= self._geschwindigkeit
        if gedrueckte_tasten[self._steuerung['rechts']]:
            bewegung_x += self._geschwindigkeit

        self.rect.x += bewegung_x
        self.rect.y += bewegung_y

        # Auto innerhalb des Fensters halten
        spielfeld = pygame.display.get_surface().get_rect()
        self.rect.clamp_ip(spielfeld)

    # pygame.sprite.Group ruft update() auf – wir leiten es weiter
    def update(self) -> None:
        self.aktualisiere()


# ─────────────────────────────────────────────
#  Spiel-Klasse
# ─────────────────────────────────────────────
class Spiel:
    """Hauptklasse: verwaltet Fenster, Autos und den Game-Loop."""

    def __init__(self) -> None:
        pygame.init()
        self._fenster = pygame.display.set_mode((FENSTER_BREITE, FENSTER_HOEHE))
        pygame.display.set_caption("Autorennspiel")
        self._uhr = pygame.time.Clock()

        self._erstelle_autos()

    # ── Initialisierung ───────────────────────

    def _erstelle_autos(self) -> None:
        """Legt beide Autos an und speichert sie in einer Sprite-Gruppe."""

        # Tastenbelegung Spieler 1 – WASD
        steuerung_s1 = {
            'oben':   pygame.K_w,
            'unten':  pygame.K_s,
            'links':  pygame.K_a,
            'rechts': pygame.K_d,
        }

        # Tastenbelegung Spieler 2 – Pfeiltasten
        steuerung_s2 = {
            'oben':   pygame.K_UP,
            'unten':  pygame.K_DOWN,
            'links':  pygame.K_LEFT,
            'rechts': pygame.K_RIGHT,
        }

        auto1 = Auto(FENSTER_BREITE // 3,     FENSTER_HOEHE // 2, FARBE_ROT,  steuerung_s1)
        auto2 = Auto(FENSTER_BREITE // 3 * 2, FENSTER_HOEHE // 2, FARBE_BLAU, steuerung_s2)

        # Alle Sprites in einer gemeinsamen Gruppe
        self._alle_sprites: pygame.sprite.Group = pygame.sprite.Group(auto1, auto2)

    # ── Game-Loop-Methoden ────────────────────

    def _verarbeite_ereignisse(self) -> None:
        """Prüft alle pygame-Events (Schließen, ESC usw.)."""
        for ereignis in pygame.event.get():
            if ereignis.type == pygame.QUIT:
                self._beenden()
            if ereignis.type == pygame.KEYDOWN and ereignis.key == pygame.K_ESCAPE:
                self._beenden()

    def _zeichne_alles(self) -> None:
        """Zeichnet Hintergrund, Sprites und HUD auf das Fenster."""
        # Grüner Hintergrund
        self._fenster.fill(FARBE_GRUEN)

        # Alle Sprites zeichnen
        self._alle_sprites.draw(self._fenster)

        # Steuerungshinweise einblenden
        self._zeichne_hinweise()

        pygame.display.flip()

    def _zeichne_hinweise(self) -> None:
        """Blendet Steuerungshinweise oben links ein."""
        schrift = pygame.font.SysFont(None, 22)

        hinweise = [
            ("Spieler 1 (Rot):  WASD",        FARBE_ROT,   20),
            ("Spieler 2 (Blau): Pfeiltasten",  FARBE_BLAU,  44),
            ("ESC = Beenden",                  FARBE_WEISS, 68),
        ]

        for text, farbe, y in hinweise:
            # Schatten für bessere Lesbarkeit auf Grün
            schatten = schrift.render(text, True, FARBE_SCHWARZ)
            self._fenster.blit(schatten, (12, y + 1))
            self._fenster.blit(schrift.render(text, True, farbe), (11, y))

    @staticmethod
    def _beenden() -> None:
        """Beendet pygame und das Programm sauber."""
        pygame.quit()
        sys.exit()

    # ── Einstieg ─────────────────────────────

    def starte(self) -> None:
        """Startet den Haupt-Game-Loop."""
        while True:
            self._verarbeite_ereignisse()
            self._alle_sprites.update()
            self._zeichne_alles()
            self._uhr.tick(BILDER_PRO_SEK)


# ─────────────────────────────────────────────
#  Einstiegspunkt
# ─────────────────────────────────────────────
if __name__ == "__main__":
    Spiel().starte()