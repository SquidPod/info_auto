import sys, json, math, random
from pathlib import Path
import pygame
import numpy as np
from scipy.ndimage import uniform_filter1d

# ── Konstanten ────────────────────────────────────────────────
W, H, FPS   = 1500, 1000, 60
TB          = 60
RUNDEN_MAX  = 3
SAVE        = Path(__file__).parent / "strecke.json"

# Farben
BG    = (22, 26, 35);  TB_C  = (18, 21, 30);  PANEL = (32, 37, 50)
RAND  = (55, 62, 80);  TEXT  = (210, 215, 228); DIM  = (110, 118, 140)
WEISS = (255,255,255); SCHWARZ = (0,0,0)
ROT   = (220, 55, 55); BLAU  = (50,100,220);  GRUEN = (45,185,100)
ORANGE= (230,145, 35); LILA  = (155, 65,215); LILA2 = (100, 40,145)
GELB  = (255,220, 40); CYAN  = (45,195,210)
GRAS  = (68,122,50);   GRAS_H= (82,145,62);   GRAS_D= (52, 98,40)

# Physik
MAX_V  = 5.0;  MAX_OFF = 1.8
ACCEL  = 0.15; BREMSE  = 0.25
REIB   = 0.04; REIB_OFF= 0.12
LENKUNG= 2.2;  SCHWELLE= 0.30

# Maße
AUTO_B, AUTO_H = 22, 38
CP_B,   CP_H   = 120, 14
SZ_B,   SZ_H   = 140, 14
HANDLE_R        = 8
DICKE_MIN, DICKE_MAX, DICKE_S = 4, 60, 4

class Modus:
    ZEICHNEN = 0; START = 1; CP = 2; AUTOS = 3; FAHREN = 4; GEWONNEN = 5


# ── Hilfsfunktionen ───────────────────────────────────────────
def glaette(pts):
    if len(pts) < 4: return pts
    xs = np.array([p[0] for p in pts], float)
    ys = np.array([p[1] for p in pts], float)
    xg = uniform_filter1d(xs, 25, mode="nearest")
    yg = uniform_filter1d(ys, 25, mode="nearest")
    xg[0], yg[0] = xs[0], ys[0]; xg[-1], yg[-1] = xs[-1], ys[-1]
    return [(int(x), int(y)) for x, y in zip(xg, yg)]

def rot_pt(px, py, cx, cy, deg):
    r = math.radians(deg); dx, dy = px-cx, py-cy
    return (cx + dx*math.cos(r) - dy*math.sin(r),
            cy + dx*math.sin(r) + dy*math.cos(r))

def obb_ecken(cx, cy, b, h, deg):
    hw, hh = b/2, h/2
    return [rot_pt(cx+ex, cy+ey, cx, cy, deg) for ex, ey in [(-hw,-hh),(hw,-hh),(hw,hh),(-hw,hh)]]

def in_obb(px, py, cx, cy, b, h, deg):
    r = math.radians(-deg); dx, dy = px-cx, py-cy
    return abs(dx*math.cos(r)-dy*math.sin(r)) <= b/2 and abs(dx*math.sin(r)+dy*math.cos(r)) <= h/2

def rect_trifft_obb(rect, cx, cy, b, h, deg):
    pts = [rect.topleft, rect.topright, rect.bottomleft, rect.bottomright,
           rect.center, rect.midtop, rect.midbottom, rect.midleft, rect.midright]
    return any(in_obb(px, py, cx, cy, b, h, deg) for px, py in pts)

def streifen(surf, ecken, n, f1, f2):
    for i in range(n):
        t0, t1 = i/n, (i+1)/n
        def ip(t, e=ecken):
            p0 = (e[0][0]+t*(e[1][0]-e[0][0]), e[0][1]+t*(e[1][1]-e[0][1]))
            p1 = (e[3][0]+t*(e[2][0]-e[3][0]), e[3][1]+t*(e[2][1]-e[3][1]))
            return p0, p1
        a, b_ = ip(t0); c, d = ip(t1)
        pygame.draw.polygon(surf, f1 if i%2==0 else f2,
            [(int(a[0]),int(a[1])),(int(c[0]),int(c[1])),(int(d[0]),int(d[1])),(int(b_[0]),int(b_[1]))])

def gras_textur(w, h):
    s = pygame.Surface((w, h)); s.fill(GRAS); rng = random.Random(42)
    for _ in range(3500):
        x, y, r = rng.randint(0,w-1), rng.randint(0,h-1), rng.randint(6,45)
        pygame.draw.circle(s, GRAS_H if rng.random()>.5 else GRAS_D, (x,y), r)
    for _ in range(7000):
        x, y = rng.randint(0,w-1), rng.randint(0,h-1)
        l = rng.randint(3,10); ang = rng.uniform(-.45,.45); g = rng.randint(115,180)
        pygame.draw.line(s, (rng.randint(28,55),g,rng.randint(28,55)),
                         (x,y),(x+int(math.sin(ang)*l),y-l), 1)
    return s


# ── Linie (Basis für CP & Start/Ziel) ────────────────────────
class Linie:
    def __init__(self, cx, cy, b, h, farbe, rand_f):
        self.cx, self.cy = float(cx), float(cy)
        self.b, self.h   = b, h
        self.farbe, self.rand_f = farbe, rand_f
        self.winkel      = 0.0
        self.gedreht = self.gezogen = False
        self.drag_off    = (0., 0.)
        self.dreh_maus0  = self.dreh_obj0 = 0.

    def handle_pos(self):
        return rot_pt(self.cx + self.b/2 + HANDLE_R + 4, self.cy, self.cx, self.cy, self.winkel)

    def body_hit(self, mx, my):   return in_obb(mx, my, self.cx, self.cy, self.b, self.h+14, self.winkel)
    def handle_hit(self, mx, my):
        hx, hy = self.handle_pos()
        return math.hypot(mx-hx, my-hy) <= HANDLE_R+3

    def start_drehen(self, mx, my):
        self.gedreht    = True
        self.dreh_maus0 = math.degrees(math.atan2(my-self.cy, mx-self.cx))
        self.dreh_obj0  = self.winkel

    def start_ziehen(self, mx, my):
        self.gezogen  = True
        self.drag_off = (self.cx-mx, self.cy-my)

    def update(self, mx, my):
        if self.gedreht:
            d = math.degrees(math.atan2(my-self.cy, mx-self.cx)) - self.dreh_maus0
            self.winkel = (self.dreh_obj0 + d) % 360
        elif self.gezogen:
            self.cx = mx + self.drag_off[0]; self.cy = my + self.drag_off[1]

    def stop(self):      self.gedreht = self.gezogen = False
    def kollidiert(self, rect): return rect_trifft_obb(rect, self.cx, self.cy, self.b, self.h, self.winkel)

    def zeichne(self, surf, schrift, sel=False, edit=False):
        ecken = obb_ecken(self.cx, self.cy, self.b, self.h, self.winkel)
        pts   = [(int(x),int(y)) for x,y in ecken]
        pygame.draw.polygon(surf, self.farbe, pts)
        pygame.draw.polygon(surf, WEISS if sel else self.rand_f, pts, 2 if sel else 1)
        self._streifen(surf, ecken)
        lbl = self._label()
        if schrift and lbl:
            t = pygame.transform.rotate(schrift.render(lbl, True, WEISS), -self.winkel)
            surf.blit(t, t.get_rect(center=(int(self.cx), int(self.cy))))
        if edit:
            hx, hy = (int(v) for v in self.handle_pos())
            ex = int(self.cx + self.b/2 * math.cos(math.radians(self.winkel)))
            ey = int(self.cy + self.b/2 * math.sin(math.radians(self.winkel)))
            pygame.draw.line(surf, DIM, (ex,ey), (hx,hy), 1)
            pygame.draw.circle(surf, GELB if self.gedreht else CYAN, (hx,hy), HANDLE_R)
            pygame.draw.circle(surf, WEISS, (hx,hy), HANDLE_R, 1)

    def _streifen(self, surf, ecken): pass
    def _label(self): return ""
    def als_dict(self): return {"cx": self.cx, "cy": self.cy, "winkel": self.winkel}


class StartZiel(Linie):
    def __init__(self, cx, cy):
        super().__init__(cx, cy, SZ_B, SZ_H, ORANGE, (255,175,65))

    def _streifen(self, s, e): streifen(s, e, 8, SCHWARZ, WEISS)

    @classmethod
    def von_dict(cls, d):
        o = cls(d["cx"], d["cy"]); o.winkel = d.get("winkel", 0.); return o


class Checkpoint(Linie):
    def __init__(self, cx, cy, idx=0):
        super().__init__(cx, cy, CP_B, CP_H, LILA, (190,100,250))
        self.idx = idx

    def _streifen(self, s, e): streifen(s, e, 6, (190,100,250), LILA2)
    def _label(self): return str(self.idx + 1)
    def als_dict(self): return {**super().als_dict(), "idx": self.idx}

    @classmethod
    def von_dict(cls, d):
        o = cls(d["cx"], d["cy"], d.get("idx",0)); o.winkel = d.get("winkel",0.); return o


# ── Strecke ───────────────────────────────────────────────────
class Strecke:
    def __init__(self, gras):
        self.gras  = gras
        self.surf  = pygame.Surface((W, H-TB))
        self.linien: list = []
        self.start: StartZiel | None = None
        self.cps:   list[Checkpoint] = []
        self.dicke = 20

    def leere(self):
        self.linien.clear(); self.cps.clear(); self.start = None; self.rendere()

    def rendere(self, dicke=None):
        if dicke is not None: self.dicke = dicke
        self.surf.blit(self.gras, (0,0))
        for l in self.linien:
            if l:
                for p in glaette(l): pygame.draw.circle(self.surf, SCHWARZ, p, self.dicke)

    def pixel_ok(self, x, y):
        x = max(0, min(int(x), W-1)); y = max(0, min(int(y), H-TB-1))
        return self.surf.get_at((x,y))[:3] == SCHWARZ

    def auf_strecke(self, rect):
        pts = [rect.topleft, rect.topright, rect.bottomleft, rect.bottomright,
               rect.center, rect.midtop, rect.midbottom, rect.midleft, rect.midright]
        a = sum(1 for x,y in pts if self.pixel_ok(x,y)) / len(pts)
        return a, a >= SCHWELLE


# ── Auto ──────────────────────────────────────────────────────
def baue_auto_image(farbe):
    s = pygame.Surface((AUTO_B, AUTO_H), pygame.SRCALPHA)
    pygame.draw.rect(s, farbe, (0,0,AUTO_B,AUTO_H), border_radius=5)
    wf = tuple(min(255,c+60) for c in farbe)
    pygame.draw.rect(s, wf, (3,8,AUTO_B-6,12), border_radius=3)
    for x in (4, AUTO_B-4):
        pygame.draw.circle(s, (255,255,200), (x,3), 2)
        pygame.draw.circle(s, ROT, (x,AUTO_H-4), 2)
    return s

class Auto(pygame.sprite.Sprite):
    def __init__(self, x, y, farbe, keys, name):
        super().__init__()
        self.name, self.farbe, self.keys = name, farbe, keys
        self.base  = baue_auto_image(farbe)
        self.image = self.base.copy()
        self.pos   = pygame.Vector2(x, y)
        self.angle = 0.; self.speed = 0.
        self.rect  = self.image.get_rect(center=(x,y))
        self.runden  = 0
        self.im_ziel = False
        self.cps_ok: set[int] = set()
        self.gezogen = False; self.offset = (0,0)
        self.platziert = False; self._shake = 0

    def update(self, strecke):
        k = pygame.key.get_pressed()
        _, auf = strecke.auf_strecke(self.rect)
        vm = MAX_V if auf else MAX_OFF

        if k[self.keys["vor"]]:    self.speed = min(self.speed + ACCEL, vm)
        elif k[self.keys["rück"]]: self.speed = max(self.speed - BREMSE, -vm/2)
        else:
            r = REIB if auf else REIB_OFF
            self.speed = max(0., self.speed-r) if self.speed>0 else min(0., self.speed+r)

        if abs(self.speed) > 0.1:
            lw = LENKUNG * (1. - abs(self.speed)/MAX_V * 0.35) * (1 if self.speed>0 else -1)
            if k[self.keys["li"]]: self.angle += lw
            if k[self.keys["re"]]: self.angle -= lw
        self.angle %= 360

        self.image = pygame.transform.rotate(self.base, self.angle)
        richt = pygame.Vector2(0,-1).rotate(-self.angle)
        self.pos += richt * self.speed
        self.rect  = self.image.get_rect(center=(int(self.pos.x), int(self.pos.y)))

        if not pygame.Rect(0, 0, W, H-TB).contains(self.rect):
            self.pos -= richt * self.speed
            self.rect  = self.image.get_rect(center=(int(self.pos.x), int(self.pos.y)))
            self.speed *= -0.3

        self._shake = (self._shake+1)%6 if not auf and abs(self.speed)>.2 else 0

        for cp in strecke.cps:
            if cp.kollidiert(self.rect): self.cps_ok.add(cp.idx)

        if strecke.start:
            alle = len(self.cps_ok) >= len(strecke.cps)
            if strecke.start.kollidiert(self.rect):
                if not self.im_ziel and self.speed > 0 and alle:
                    self.runden += 1; self.cps_ok.clear(); self.im_ziel = True
            else:
                self.im_ziel = False

    def shake(self):
        return (random.randint(-1,1), random.randint(-1,1)) if self._shake else (0,0)

    @property
    def hat_gewonnen(self): return self.runden >= RUNDEN_MAX


# ── UI-Widgets ────────────────────────────────────────────────
class Knopf:
    def __init__(self, rect, text, aktiv_f=GRUEN):
        self.rect = rect; self.text = text; self.aktiv_f = aktiv_f; self.aktiv = False

    def zeichne(self, surf, font):
        hover = self.rect.collidepoint(pygame.mouse.get_pos())
        bg    = self.aktiv_f if self.aktiv else (RAND if hover else PANEL)
        rand  = tuple(min(255,c+40) for c in bg) if self.aktiv else RAND
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, rand, self.rect, 1, border_radius=6)
        t = font.render(self.text, True, WEISS if (self.aktiv or hover) else TEXT)
        surf.blit(t, t.get_rect(center=self.rect.center))

    def hit(self, ev):
        return ev.type==pygame.MOUSEBUTTONDOWN and ev.button==1 and self.rect.collidepoint(ev.pos)


class Regler:
    def __init__(self, x, y):
        self.rm = pygame.Rect(x,    y, 26, 26)
        self.rp = pygame.Rect(x+66, y, 26, 26)
        self.rw = pygame.Rect(x+28, y, 36, 26)
        self.wert = 20

    def zeichne(self, surf, font):
        for r, t in [(self.rm,"−"), (self.rp,"+")]:
            h = r.collidepoint(pygame.mouse.get_pos())
            pygame.draw.rect(surf, RAND if h else PANEL, r, border_radius=4)
            pygame.draw.rect(surf, DIM, r, 1, border_radius=4)
            lbl = font.render(t, True, TEXT)
            surf.blit(lbl, lbl.get_rect(center=r.center))
        pygame.draw.rect(surf, SCHWARZ, self.rw, border_radius=3)
        v = font.render(str(self.wert), True, GELB)
        surf.blit(v, v.get_rect(center=self.rw.center))
        surf.blit(font.render("Dicke", True, DIM), (self.rm.x, self.rm.y-17))

    def click(self, ev):
        if ev.type != pygame.MOUSEBUTTONDOWN or ev.button != 1: return False
        if self.rm.collidepoint(ev.pos): self.wert = max(DICKE_MIN, self.wert-DICKE_S); return True
        if self.rp.collidepoint(ev.pos): self.wert = min(DICKE_MAX, self.wert+DICKE_S); return True
        return False


# ── Spiel ─────────────────────────────────────────────────────
class Spiel:
    HINTS = {
        Modus.ZEICHNEN: "Maus halten: Strecke zeichnen",
        Modus.START:    "Klick: setzen | Ziehen: verschieben | Handle: drehen",
        Modus.CP:       "Klick: neu | Ziehen: verschieben | Handle: drehen | Rechtsklick: loeschen",
        Modus.AUTOS:    "Auto auf Strecke ziehen (gruen = OK) | Scroll: drehen",
        Modus.FAHREN:   "ROT: WASD   |   BLAU: Pfeiltasten",
    }

    def __init__(self):
        pygame.init()
        self.win   = pygame.display.set_mode((W, H))
        pygame.display.set_caption("Rennspiel")
        self.uhr   = pygame.time.Clock()
        self.font  = pygame.font.SysFont("segoeui", 20)
        self.fontB = pygame.font.SysFont("segoeui", 22, bold=True)
        self.fontS = pygame.font.SysFont("segoeui", 16)
        self.fontG = pygame.font.SysFont("segoeui", 72, bold=True)

        gras = gras_textur(W, H-TB)
        self.strecke = Strecke(gras)
        self.modus   = Modus.ZEICHNEN
        self.edit    = None
        self.edit_auto = None
        self.zeichnet  = False
        self.feedback  = ("", 0)

        ty, bh = H - TB + TB//2 - 13, 28
        self.knoepfe = {
            "zeichnen":  Knopf(pygame.Rect(  8, ty, 100, bh), "Strecke"),
            "start":     Knopf(pygame.Rect(116, ty, 110, bh), "Start/Ziel",  ORANGE),
            "cp":        Knopf(pygame.Rect(234, ty, 120, bh), "Checkpoints", LILA),
            "autos":     Knopf(pygame.Rect(362, ty,  80, bh), "Autos",       CYAN),
            "fahren":    Knopf(pygame.Rect(450, ty,  90, bh), "Fahren",      GRUEN),
            "reset":     Knopf(pygame.Rect(W-230, ty, 80, bh), "Reset",      ROT),
            "speichern": Knopf(pygame.Rect(W-142, ty,130, bh), "Speichern",  BLAU),
        }
        self.regler = Regler(590, H-TB+16)

        self.auto1 = Auto(120, 100, ROT,
            {"vor":pygame.K_w,"rück":pygame.K_s,"li":pygame.K_a,"re":pygame.K_d}, "ROT")
        self.auto2 = Auto(180, 100, BLAU,
            {"vor":pygame.K_UP,"rück":pygame.K_DOWN,"li":pygame.K_LEFT,"re":pygame.K_RIGHT}, "BLAU")
        self.autos = pygame.sprite.Group(self.auto1, self.auto2)
        self._lade()
        self._sync_knoepfe()

    # ── Modus & Feedback ──────────────────────────────────────
    def _modus(self, m): self.modus = m; self.edit = None; self._sync_knoepfe()

    def _sync_knoepfe(self):
        for key, mod in [("zeichnen",Modus.ZEICHNEN),("start",Modus.START),
                         ("cp",Modus.CP),("autos",Modus.AUTOS),("fahren",Modus.FAHREN)]:
            self.knoepfe[key].aktiv = self.modus == mod

    def _info(self, t, d=110): self.feedback = (t, d)

    # ── Speichern / Laden ─────────────────────────────────────
    def _speichere(self):
        d = {"linien": self.strecke.linien, "dicke": self.regler.wert,
             "start": self.strecke.start.als_dict() if self.strecke.start else None,
             "cps":   [c.als_dict() for c in self.strecke.cps]}
        SAVE.write_text(json.dumps(d), encoding="utf-8")
        self._info("Gespeichert")

    def _lade(self):
        if not SAVE.exists(): return
        d = json.loads(SAVE.read_text(encoding="utf-8"))
        self.strecke.linien = [[tuple(p) for p in l] for l in d.get("linien",[])]
        self.regler.wert    = d.get("dicke", 20)
        if d.get("start"): self.strecke.start = StartZiel.von_dict(d["start"])
        for c in d.get("cps",[]): self.strecke.cps.append(Checkpoint.von_dict(c))
        self.strecke.rendere(self.regler.wert)

    # ── Events ────────────────────────────────────────────────
    def events(self):
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (ev.type==pygame.KEYDOWN and ev.key==pygame.K_ESCAPE):
                # Neustart aus Gewonnen-Bildschirm mit beliebiger Taste
                if self.modus == Modus.GEWONNEN and ev.type == pygame.KEYDOWN:
                    self._neustart(); return
                pygame.quit(); sys.exit()

            if ev.type == pygame.KEYDOWN and self.modus == Modus.GEWONNEN:
                self._neustart(); return

            if ev.type == pygame.MOUSEWHEEL:
                if self.modus == Modus.AUTOS:
                    mx, my = pygame.mouse.get_pos()
                    for a in self.autos:
                        if a.rect.collidepoint(mx, my):
                            a.angle = (a.angle + ev.y*5) % 360
                            a.image = pygame.transform.rotate(a.base, a.angle)
                            a.rect  = a.image.get_rect(center=(int(a.pos.x), int(a.pos.y)))
                elif self.edit and isinstance(self.edit, Linie):
                    self.edit.winkel = (self.edit.winkel + ev.y*5) % 360
                    self.strecke.rendere()

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if   ev.button == 1: self._down(ev)
                elif ev.button == 3: self._rechts(ev)
            elif ev.type == pygame.MOUSEMOTION:              self._move(ev)
            elif ev.type == pygame.MOUSEBUTTONUP and ev.button == 1: self._up(ev)

    def _up(self, ev):
        self.zeichnet = False
        if self.edit and isinstance(self.edit, Linie):
            self.edit.stop(); self.strecke.rendere()
        if self.edit_auto:
            a = self.edit_auto; a.gezogen = False
            _, ok = self.strecke.auf_strecke(a.rect)
            a.platziert = ok; a.pos = pygame.Vector2(a.rect.center)
            self.edit_auto = None

    def _rechts(self, ev):
        if self.modus != Modus.CP: return
        mx, my = ev.pos
        for i, cp in enumerate(self.strecke.cps):
            if cp.body_hit(mx, my):
                self.strecke.cps.pop(i)
                for j, c in enumerate(self.strecke.cps): c.idx = j
                if self.edit is cp: self.edit = None
                self.strecke.rendere(); self._info("Checkpoint entfernt"); break

    def _down(self, ev):
        for n, k in self.knoepfe.items():
            if k.hit(ev): self._knopf(n); return

        if self.modus == Modus.ZEICHNEN and self.regler.click(ev):
            self.strecke.rendere(self.regler.wert); return

        if ev.pos[1] >= H-TB: return
        mx, my = ev.pos

        if self.modus == Modus.ZEICHNEN:
            self.zeichnet = True; self.strecke.linien.append([ev.pos])

        elif self.modus == Modus.START:
            sz = self.strecke.start
            if sz and sz.handle_hit(mx, my): sz.start_drehen(mx,my); self.edit = sz
            elif sz and sz.body_hit(mx, my): sz.start_ziehen(mx,my); self.edit = sz
            else:
                self.strecke.start = StartZiel(mx,my); self.edit = self.strecke.start
                self.strecke.rendere()

        elif self.modus == Modus.CP:
            hit = None
            for cp in self.strecke.cps:
                if   cp.handle_hit(mx,my): cp.start_drehen(mx,my); self.edit = cp; hit = cp; break
                elif cp.body_hit(mx,my):   cp.start_ziehen(mx,my); self.edit = cp; hit = cp; break
            if not hit:
                cp = Checkpoint(mx, my, len(self.strecke.cps))
                self.strecke.cps.append(cp); self.edit = cp; self.strecke.rendere()

        elif self.modus == Modus.AUTOS:
            for a in self.autos:
                if a.rect.collidepoint(ev.pos):
                    a.gezogen = True; a.offset = (a.rect.x-mx, a.rect.y-my)
                    self.edit_auto = a; break

    def _knopf(self, n):
        aktionen = {
            "zeichnen":  lambda: self._modus(Modus.ZEICHNEN),
            "start":     lambda: self._modus(Modus.START),
            "cp":        lambda: self._modus(Modus.CP),
            "autos":     lambda: self._modus(Modus.AUTOS),
            "speichern": self._speichere,
            "reset":     lambda: (self.strecke.leere(), self.edit.__setattr__('edit', None) if False else None, self._info("Strecke geleert")),
        }
        if n == "fahren":
            if self.auto1.platziert and self.auto2.platziert and self.strecke.start:
                self._modus(Modus.FAHREN)
            else:
                self._info("Autos platzieren + Start/Ziel setzen")
        elif n == "reset":
            self.strecke.leere(); self.edit = None; self._info("Strecke geleert")
        elif n in aktionen:
            aktionen[n]()

    def _move(self, ev):
        mx, my = ev.pos
        if self.modus == Modus.ZEICHNEN and self.zeichnet and my < H-TB:
            self.strecke.linien[-1].append(ev.pos); self.strecke.rendere()
        elif self.edit and isinstance(self.edit, Linie) and (self.edit.gedreht or self.edit.gezogen):
            self.edit.update(mx, my); self.strecke.rendere()
        elif self.edit_auto:
            a = self.edit_auto
            a.rect.x = mx + a.offset[0]; a.rect.y = my + a.offset[1]

    # ── Neustart ──────────────────────────────────────────────
    def _neustart(self):
        for a in self.autos:
            a.runden = 0; a.im_ziel = False; a.cps_ok.clear(); a.speed = 0
        self._modus(Modus.AUTOS)

    # ── Zeichnen ──────────────────────────────────────────────
    def _draw_feld(self):
        self.win.blit(self.strecke.surf, (0,0))

        for cp in self.strecke.cps:
            cp.zeichne(self.win, self.fontS, cp is self.edit, self.modus==Modus.CP)
        if self.strecke.start:
            self.strecke.start.zeichne(self.win, self.fontS,
                self.strecke.start is self.edit, self.modus==Modus.START)

        for a in self.autos:
            ox, oy = a.shake()
            self.win.blit(a.image, (a.rect.x+ox, a.rect.y+oy))

        if self.modus == Modus.AUTOS:
            for a in self.autos:
                c = GRUEN if a.platziert else ROT
                pygame.draw.rect(self.win, c, a.rect.inflate(6,6), 2, border_radius=4)

        if self.modus == Modus.CP:
            mx, my = pygame.mouse.get_pos()
            if my < H-TB and not any(cp.body_hit(mx,my) for cp in self.strecke.cps):
                s = pygame.Surface((CP_B,CP_H), pygame.SRCALPHA)
                s.fill((*LILA,90)); self.win.blit(s, (mx-CP_B//2, my-CP_H//2))

        if self.modus == Modus.FAHREN:
            for a in [self.auto1, self.auto2]:
                _, auf = self.strecke.auf_strecke(a.rect)
                if not auf and abs(a.speed) > .3:
                    t = self.fontS.render("OFFROAD", True, a.farbe)
                    self.win.blit(t, (a.rect.centerx-t.get_width()//2, a.rect.top-20))

    def _draw_toolbar(self):
        pygame.draw.rect(self.win, TB_C, (0, H-TB, W, TB))
        pygame.draw.line(self.win, RAND, (0,H-TB), (W,H-TB), 1)
        for k in self.knoepfe.values(): k.zeichne(self.win, self.font)
        if self.modus == Modus.ZEICHNEN: self.regler.zeichne(self.win, self.fontS)
        t = self.fontS.render(self.HINTS.get(self.modus,""), True, DIM)
        self.win.blit(t, t.get_rect(center=(W//2, H-12)))

    def _draw_hud(self):
        ncps = len(self.strecke.cps)
        for i, a in enumerate([self.auto1, self.auto2]):
            px, py = 12, 12 + i*58
            s = pygame.Surface((190,46), pygame.SRCALPHA)
            s.fill((*PANEL,200)); self.win.blit(s, (px,py))
            pygame.draw.rect(self.win, a.farbe, (px,py,190,46), 1, border_radius=5)

            n = self.fontB.render(a.name, True, a.farbe)
            r = self.fontB.render(f"Runde {a.runden}/{RUNDEN_MAX}", True, TEXT)
            self.win.blit(n, (px+6, py+4))
            self.win.blit(r, (px+6+n.get_width()+8, py+4))

            if ncps > 0 and self.modus == Modus.FAHREN:
                ok = len(a.cps_ok); bw = 178
                pygame.draw.rect(self.win, RAND,    (px+6, py+30, bw, 8), border_radius=3)
                if ok: pygame.draw.rect(self.win, a.farbe, (px+6, py+30, int(bw*ok/ncps), 8), border_radius=3)
                ct = self.fontS.render(f"CP {ok}/{ncps}", True, DIM)
                self.win.blit(ct, (px+bw-ct.get_width()+6, py+30))

        txt, t = self.feedback
        if t > 0:
            ft = self.font.render(txt, True, GELB)
            bg = pygame.Surface((ft.get_width()+16, ft.get_height()+8), pygame.SRCALPHA)
            bg.fill((0,0,0,180)); cx, cy = W//2, H-TB-36
            self.win.blit(bg, (cx-bg.get_width()//2, cy))
            self.win.blit(ft, ft.get_rect(center=(cx, cy+bg.get_height()//2)))
            self.feedback = (txt, t-1)

    def _draw_gewonnen(self, gewinner: Auto):
        overlay = pygame.Surface((W, H), pygame.SRCALPHA)
        overlay.fill((0,0,0,170)); self.win.blit(overlay, (0,0))

        titel = self.fontG.render(f"🏆  {gewinner.name} GEWINNT!", True, gewinner.farbe)
        unter = self.fontB.render(f"{RUNDEN_MAX} Runden abgeschlossen  ·  Beliebige Taste drücken für Neustart", True, TEXT)

        # Leuchtrahmen
        r = pygame.Rect(0,0, max(titel.get_width(),unter.get_width())+80, 180)
        r.center = (W//2, H//2)
        pygame.draw.rect(self.win, (*gewinner.farbe, 60), r, border_radius=16)
        pygame.draw.rect(self.win, gewinner.farbe, r, 3, border_radius=16)

        self.win.blit(titel, titel.get_rect(center=(W//2, H//2 - 30)))
        self.win.blit(unter, unter.get_rect(center=(W//2, H//2 + 40)))

    def zeichne(self, gewinner=None):
        self._draw_feld()
        self._draw_toolbar()
        self._draw_hud()
        if gewinner: self._draw_gewonnen(gewinner)
        pygame.display.flip()

    # ── Hauptschleife ─────────────────────────────────────────
    def run(self):
        gewinner = None
        while True:
            self.events()

            if self.modus == Modus.FAHREN:
                self.autos.update(self.strecke)
                # Siegprüfung
                for a in [self.auto1, self.auto2]:
                    if a.hat_gewonnen and gewinner is None:
                        gewinner = a
                        self.modus = Modus.GEWONNEN

            if self.modus == Modus.GEWONNEN and gewinner is None:
                # Neustart wurde ausgelöst
                gewinner = None

            # Gewinner-Overlay nach Neustart zurücksetzen
            if self.modus != Modus.GEWONNEN:
                gewinner = None

            self.zeichne(gewinner if self.modus == Modus.GEWONNEN else None)
            self.uhr.tick(FPS)


if __name__ == "__main__":
    Spiel().run()
