

import pygame
import heapq
import random
import time
import math
import sys

pygame.init()

# ─── Colours  ───────────────────────────────────────────
BG         = (255, 230, 242)
PANEL_BG   = (255, 214, 235)
GRID_BG    = (255, 240, 248)
GRID_LINE  = (224, 179, 208)
EMPTY      = (255, 240, 248)
WALL_C     = ( 77,  77, 102)
VISITED_C  = (255, 153, 204)   # pink
FRONTIER_C = (255, 204,   0)   # yellow
PATH_C     = (255,  51, 153)   # magenta
START_C    = (255, 102, 178)
GOAL_C     = (153, 102, 255)
AGENT_C    = (255, 255, 255)
TEXT_DARK  = ( 77,   0,  51)
TEXT_LIGHT = (255, 255, 255)
ACCENT     = (204,   0, 102)
BTN_NORMAL = (204, 153, 255)
BTN_HOVER  = (255, 153, 204)
BTN_ACTIVE = (255, 102, 178)
DIVIDER    = (255, 153, 204)

PANEL_W  = 295
FPS      = 60
MIN_CELL = 18
MAX_CELL = 56


# ══════════════════════════════════════════════════════════════════════════════
#  Pathfinder — pure Python lists, no numpy
# ══════════════════════════════════════════════════════════════════════════════
class Pathfinder:
    def __init__(self, rows, cols):
        self.rows   = rows
        self.cols   = cols
        # grid: 0 = empty, -1 = wall  (plain list-of-lists)
        self.grid   = [[0]*cols for _ in range(rows)]
        self.start  = (0, 0)
        self.target = (rows-1, cols-1)
        self.moves  = [(-1,0),(0,1),(1,0),(0,-1)]   # U R D L

    def is_valid(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols and self.grid[r][c] != -1

    def toggle_wall(self, r, c):
        if (r, c) in (self.start, self.target):
            return
        self.grid[r][c] = 0 if self.grid[r][c] == -1 else -1

    def set_wall(self, r, c):
        if (r, c) not in (self.start, self.target):
            self.grid[r][c] = -1

    def random_maze(self, density):
        self.grid = [[0]*self.cols for _ in range(self.rows)]
        cands = [(r,c) for r in range(self.rows) for c in range(self.cols)
                 if (r,c) != self.start and (r,c) != self.target]
        n = int(self.rows * self.cols * density)
        for r, c in random.sample(cands, min(n, len(cands))):
            self.grid[r][c] = -1

    def clear(self):
        self.grid = [[0]*self.cols for _ in range(self.rows)]

    # ── Heuristics ────────────────────────────────────────────────────────────
    def manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def euclidean(self, a, b):
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def hval(self, pos, heuristic):
        fn = self.manhattan if heuristic == 'Manhattan' else self.euclidean
        return fn(pos, self.target)

    # ── Path reconstruction ───────────────────────────────────────────────────
    def get_path(self, pm):
        path, curr = [], self.target
        while curr is not None:
            path.append(curr)
            curr = pm.get(curr)
        return path[::-1]

    # ── GBFS step-generator ───────────────────────────────────────────────────
    def gbfs_gen(self, heuristic):
        t0  = time.time()
        pq  = [(self.hval(self.start, heuristic), self.start)]
        pm  = {self.start: None}
        exp = []
        while pq:
            _, curr = heapq.heappop(pq)
            if curr == self.target:
                yield set(n for _,n in pq), exp[:], self.get_path(pm), True, (time.time()-t0)*1000
                return
            if curr not in exp:
                exp.append(curr)
                for dr, dc in self.moves:
                    nb = (curr[0]+dr, curr[1]+dc)
                    if self.is_valid(nb) and nb not in pm:
                        pm[nb] = curr
                        heapq.heappush(pq, (self.hval(nb, heuristic), nb))
                yield set(n for _,n in pq), exp[:], None, False, (time.time()-t0)*1000
        yield set(), exp[:], None, True, (time.time()-t0)*1000

    # ── A* step-generator ─────────────────────────────────────────────────────
    def astar_gen(self, heuristic):
        t0  = time.time()
        pq  = [(self.hval(self.start, heuristic), 0, self.start)]
        pm  = {self.start: None}
        gs  = {self.start: 0}
        exp = []
        while pq:
            _, g, curr = heapq.heappop(pq)
            if curr == self.target:
                yield set(n for _,_,n in pq), exp[:], self.get_path(pm), True, (time.time()-t0)*1000
                return
            if curr not in exp:
                exp.append(curr)
                for dr, dc in self.moves:
                    nb = (curr[0]+dr, curr[1]+dc)
                    if self.is_valid(nb):
                        ng = gs[curr] + 1
                        if ng < gs.get(nb, 10**9):
                            gs[nb] = ng; pm[nb] = curr
                            heapq.heappush(pq, (ng + self.hval(nb, heuristic), ng, nb))
                yield set(n for _,_,n in pq), exp[:], None, False, (time.time()-t0)*1000
        yield set(), exp[:], None, True, (time.time()-t0)*1000

    # ── Instant plan (for dynamic re-planning, no animation) ──────────────────
    def plan(self, start, heuristic, algorithm):
        old_start   = self.start
        self.start  = start
        t0 = time.time()

        if algorithm == 'A*':
            pq = [(self.hval(start, heuristic), 0, start)]
            pm = {start: None}; gs = {start: 0}; exp = []
            while pq:
                _, g, curr = heapq.heappop(pq)
                if curr == self.target:
                    p = self.get_path(pm)
                    self.start = old_start
                    return p, exp, (time.time()-t0)*1000
                if curr not in exp:
                    exp.append(curr)
                    for dr, dc in self.moves:
                        nb = (curr[0]+dr, curr[1]+dc)
                        if self.is_valid(nb):
                            ng = gs[curr]+1
                            if ng < gs.get(nb, 10**9):
                                gs[nb]=ng; pm[nb]=curr
                                heapq.heappush(pq,(ng+self.hval(nb,heuristic),ng,nb))
        else:  # GBFS
            pq = [(self.hval(start, heuristic), start)]
            pm = {start: None}; exp = []
            while pq:
                _, curr = heapq.heappop(pq)
                if curr == self.target:
                    p = self.get_path(pm)
                    self.start = old_start
                    return p, exp, (time.time()-t0)*1000
                if curr not in exp:
                    exp.append(curr)
                    for dr, dc in self.moves:
                        nb = (curr[0]+dr, curr[1]+dc)
                        if self.is_valid(nb) and nb not in pm:
                            pm[nb] = curr
                            heapq.heappush(pq, (self.hval(nb, heuristic), nb))

        self.start = old_start
        return None, exp, (time.time()-t0)*1000


# ══════════════════════════════════════════════════════════════════════════════
#  UI Widgets
# ══════════════════════════════════════════════════════════════════════════════
class Button:
    def __init__(self, rect, label, active=False):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.active = active
        self._hov   = False

    def draw(self, surf, font):
        color = BTN_ACTIVE if self.active else (BTN_HOVER if self._hov else BTN_NORMAL)
        pygame.draw.rect(surf, color, self.rect, border_radius=7)
        pygame.draw.rect(surf, ACCENT if self.active else DIVIDER, self.rect, 2, border_radius=7)
        txt = font.render(self.label, True, TEXT_DARK)
        surf.blit(txt, txt.get_rect(center=self.rect.center))

    def update(self, mpos):
        self._hov = self.rect.collidepoint(mpos)

    def clicked(self, event):
        return (event.type == pygame.MOUSEBUTTONDOWN
                and event.button == 1
                and self.rect.collidepoint(event.pos))


class RadioGroup:
    """Mutually-exclusive set of buttons."""
    def __init__(self, options, selected, rects, font):
        self.selected = selected
        self.font     = font
        self.buttons  = [Button(r, o, active=(o==selected))
                         for o, r in zip(options, rects)]

    def draw(self, surf):
        for b in self.buttons: b.draw(surf, self.font)

    def update(self, mpos):
        for b in self.buttons: b.update(mpos)

    def handle(self, event):
        for b in self.buttons:
            if b.clicked(event):
                self.selected = b.label
                for x in self.buttons:
                    x.active = (x.label == self.selected)
                return True
        return False


def draw_text(surf, font, text, pos, color=TEXT_DARK, anchor='topleft'):
    s = font.render(text, True, color)
    surf.blit(s, s.get_rect(**{anchor: pos}))


# ══════════════════════════════════════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════════════════════════════════════
class App:
    def __init__(self, rows, cols):
        self.pf   = Pathfinder(rows, cols)
        self.rows = rows
        self.cols = cols

        info = pygame.display.Info()
        max_w = max(300, info.current_w - PANEL_W - 60)
        max_h = max(300, info.current_h - 100)
        cell  = min(MAX_CELL, max(MIN_CELL, min(max_w//cols, max_h//rows)))
        self.cell = cell

        self.grid_w = cols * cell
        self.grid_h = rows * cell
        self.win_w  = self.grid_w + PANEL_W
        self.win_h  = max(self.grid_h, 660)

        self.screen = pygame.display.set_mode((self.win_w, self.win_h))
        pygame.display.set_caption('Dynamic Pathfinding Agent')

        self.font_sm  = pygame.font.SysFont('consolas', 12)
        self.font_med = pygame.font.SysFont('consolas', 13, bold=True)
        self.font_lg  = pygame.font.SysFont('consolas', 15, bold=True)

        # App state
        self.algorithm  = 'A*'
        self.heuristic  = 'Manhattan'
        self.place_mode = 'Wall'
        self.density    = 0.30
        self.dynamic_on = False

        # Search visualisation state
        self.frontier_vis  = set()
        self.visited_vis   = []
        self.path_vis      = []
        self.visited_set   = set()   # fast lookup
        self.gen           = None
        self.search_done   = False
        self.metrics       = {'nodes': 0, 'cost': 0, 'time_ms': 0.0}

        # Dynamic agent state
        self.agent_step    = 0
        self.agent_moving  = False
        self.total_replans = 0
        self.agent_timer   = 0
        self.agent_delay   = 280   # ms between agent steps

        # Search step timer
        self.step_timer = 0
        self.step_delay = 25       # ms between search steps

        self.dragging = False

        self._build_ui()

    # ── Build panel widgets ───────────────────────────────────────────────────
    def _build_ui(self):
        px  = self.grid_w + 10
        pw  = PANEL_W - 20
        bh  = 28
        gap = 5

        def R(y): return (px, y, pw, bh)

        y = 8
        self.lbl_y = {}   # for drawing section labels

        self.lbl_y['algo'] = y; y += 15
        hw = pw//2 - 3
        self.rg_algo = RadioGroup(
            ['A*','GBFS'], self.algorithm,
            [(px, y, hw, bh), (px+hw+6, y, hw, bh)],
            self.font_sm); y += bh+gap

        self.lbl_y['heur'] = y; y += 15
        self.rg_heur = RadioGroup(
            ['Manhattan','Euclidean'], self.heuristic,
            [(px, y, hw, bh), (px+hw+6, y, hw, bh)],
            self.font_sm); y += bh+gap

        self.lbl_y['place'] = y; y += 15
        w3 = pw//3 - 2
        self.rg_place = RadioGroup(
            ['Wall','Start','Goal'], self.place_mode,
            [(px, y, w3, bh), (px+w3+3, y, w3, bh), (px+2*(w3+3), y, w3, bh)],
            self.font_sm); y += bh+gap+4

        self.lbl_y['density'] = y; y += 15
        self.btn_dm = Button((px, y, 28, bh), '-')
        self.btn_dp = Button((px+pw-28, y, 28, bh), '+')
        self.density_rect = pygame.Rect(px+32, y, pw-64, bh)
        y += bh+gap+8

        self.btn_gen   = Button(R(y), 'Random Maze');         y += bh+gap
        self.btn_clear = Button(R(y), 'Clear Walls');         y += bh+gap
        self.btn_run   = Button(R(y), 'Run Search');          y += bh+gap
        self.btn_dyn   = Button(R(y), 'Dynamic Mode: OFF');   y += bh+gap
        self.btn_reset = Button(R(y), 'Reset View');          y += bh+gap+10

        self.metrics_y = y

        self.all_btns = [
            self.btn_dm, self.btn_dp,
            self.btn_gen, self.btn_clear,
            self.btn_run, self.btn_dyn, self.btn_reset
        ]

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        clock = pygame.time.Clock()
        while True:
            dt   = clock.tick(FPS)
            mpos = pygame.mouse.get_pos()

            for rg in (self.rg_algo, self.rg_heur, self.rg_place):
                rg.update(mpos)
            for b in self.all_btns:
                b.update(mpos)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit()
                self._handle(event, mpos)

            # Advance search animation
            if self.gen and not self.search_done:
                self.step_timer += dt
                if self.step_timer >= self.step_delay:
                    self.step_timer = 0
                    self._step()

            # Advance agent (dynamic)
            if self.agent_moving:
                self.agent_timer += dt
                if self.agent_timer >= self.agent_delay:
                    self.agent_timer = 0
                    self._agent_tick()

            self._draw()
            pygame.display.flip()

    # ── Event handling ────────────────────────────────────────────────────────
    def _handle(self, event, mpos):
        if self.rg_algo.handle(event):
            self.algorithm = self.rg_algo.selected; self._reset(); return
        if self.rg_heur.handle(event):
            self.heuristic = self.rg_heur.selected; self._reset(); return
        if self.rg_place.handle(event):
            self.place_mode = self.rg_place.selected; return

        if self.btn_dm.clicked(event):
            self.density = max(0.05, round(self.density-0.05, 2)); return
        if self.btn_dp.clicked(event):
            self.density = min(0.70, round(self.density+0.05, 2)); return
        if self.btn_gen.clicked(event):
            self.pf.random_maze(self.density); self._reset(); return
        if self.btn_clear.clicked(event):
            self.pf.clear(); self._reset(); return
        if self.btn_run.clicked(event):
            self._reset(); self._start(); return
        if self.btn_dyn.clicked(event):
            self.dynamic_on = not self.dynamic_on
            self.btn_dyn.label  = f'Dynamic Mode: {"ON" if self.dynamic_on else "OFF"}'
            self.btn_dyn.active = self.dynamic_on; return
        if self.btn_reset.clicked(event):
            self._reset(); return

        # Grid mouse interaction
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            gp = self._gpos(event.pos)
            if gp:
                self._gclick(*gp)
                self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if event.type == pygame.MOUSEMOTION and self.dragging:
            gp = self._gpos(mpos)
            if gp and self.place_mode == 'Wall':
                self.pf.set_wall(*gp)
                self._reset()

    def _gpos(self, pos):
        x, y = pos
        if x >= self.grid_w or y >= self.grid_h:
            return None
        c, r = x // self.cell, y // self.cell
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return r, c
        return None

    def _gclick(self, r, c):
        if self.place_mode == 'Wall':
            self.pf.toggle_wall(r, c); self._reset()
        elif self.place_mode == 'Start':
            if (r,c) != self.pf.target and self.pf.grid[r][c] != -1:
                self.pf.start = (r,c); self._reset()
        elif self.place_mode == 'Goal':
            if (r,c) != self.pf.start and self.pf.grid[r][c] != -1:
                self.pf.target = (r,c); self._reset()

    # ── Search control ────────────────────────────────────────────────────────
    def _reset(self):
        self.frontier_vis  = set()
        self.visited_vis   = []
        self.visited_set   = set()
        self.path_vis      = []
        self.gen           = None
        self.search_done   = False
        self.agent_step    = 0
        self.agent_moving  = False
        self.total_replans = 0
        self.metrics       = {'nodes': 0, 'cost': 0, 'time_ms': 0.0}

    def _start(self):
        if self.dynamic_on:
            path, vis, elapsed = self.pf.plan(
                self.pf.start, self.heuristic, self.algorithm)
            self.visited_vis        = vis
            self.visited_set        = set(vis)
            self.metrics['nodes']   = len(vis)
            self.metrics['time_ms'] = elapsed
            if not path:
                self.metrics['cost'] = 0
                print('[Dynamic] No initial path found!')
                return
            self.path_vis         = path
            self.metrics['cost']  = len(path) - 1
            self.agent_step       = 0
            self.agent_moving     = True
        else:
            self.gen = (self.pf.astar_gen if self.algorithm == 'A*'
                        else self.pf.gbfs_gen)(self.heuristic)

    def _step(self):
        if not self.gen: return
        try:
            frontier, visited, path, done, elapsed = next(self.gen)
            self.frontier_vis       = frontier
            self.visited_vis        = visited
            self.visited_set        = set(visited)
            self.metrics['nodes']   = len(visited)
            self.metrics['time_ms'] = elapsed
            if done:
                self.gen = None; self.search_done = True
                if path:
                    self.path_vis        = path
                    self.frontier_vis    = set()
                    self.metrics['cost'] = len(path) - 1
        except StopIteration:
            self.gen = None; self.search_done = True

    # ── Dynamic agent tick ────────────────────────────────────────────────────
    def _agent_tick(self):
        if self.agent_step >= len(self.path_vis):
            self.agent_moving = False; return

        self.agent_step += 1
        if self.agent_step >= len(self.path_vis):
            self.agent_moving = False; return

        agent_pos = self.path_vis[self.agent_step - 1]
        remaining = set(self.path_vis[self.agent_step:])

        # Spawn one random wall away from path, S, T, agent
        empty = [
            (r, c)
            for r in range(self.rows) for c in range(self.cols)
            if (r, c) not in remaining
            and (r, c) not in (self.pf.start, self.pf.target, agent_pos)
            and self.pf.grid[r][c] != -1
        ]
        if empty and random.random() < 0.18:
            wr, wc = random.choice(empty)
            self.pf.grid[wr][wc] = -1

        # Check if any future path cell is now a wall
        blocked = any(self.pf.grid[p[0]][p[1]] == -1
                      for p in self.path_vis[self.agent_step:])

        if blocked:
            cur = self.path_vis[self.agent_step - 1]
            new_path, new_vis, new_t = self.pf.plan(
                cur, self.heuristic, self.algorithm)
            self.metrics['nodes']   += len(new_vis)
            self.metrics['time_ms'] += new_t
            self.total_replans      += 1
            print(f'  Re-plan #{self.total_replans} from {cur}')
            if new_path is None:
                self.agent_moving = False
                print('[Dynamic] Grid too blocked — cannot reach goal!')
                return
            self.path_vis        = new_path
            self.agent_step      = 0
            self.metrics['cost'] = len(new_path) - 1

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw(self):
        self.screen.fill(BG)
        self._draw_grid()
        self._draw_panel()

    def _draw_grid(self):
        cell = self.cell
        surf = self.screen
        pf   = self.pf

        pygame.draw.rect(surf, GRID_BG, (0, 0, self.grid_w, self.grid_h))

        for r in range(self.rows):
            for c in range(self.cols):
                x, y = c*cell, r*cell
                pos  = (r, c)

                if   pos == pf.start:              color = START_C
                elif pos == pf.target:             color = GOAL_C
                elif pf.grid[r][c] == -1:          color = WALL_C
                elif pos in self.path_vis:         color = PATH_C
                elif pos in self.frontier_vis:     color = FRONTIER_C
                elif pos in self.visited_set:      color = VISITED_C
                else:                              color = EMPTY

                pygame.draw.rect(surf, color,
                                 (x+2, y+2, cell-4, cell-4),
                                 border_radius=max(2, cell//8))

        # Grid lines
        for r in range(self.rows+1):
            pygame.draw.line(surf, GRID_LINE, (0, r*cell), (self.grid_w, r*cell))
        for c in range(self.cols+1):
            pygame.draw.line(surf, GRID_LINE, (c*cell, 0), (c*cell, self.grid_h))

        # Path line
        if len(self.path_vis) > 1:
            pts = [(c*cell+cell//2, r*cell+cell//2) for r, c in self.path_vis]
            pygame.draw.lines(surf, PATH_C, False, pts, max(2, cell//6))

        # Agent
        if self.agent_moving and self.path_vis:
            step = min(self.agent_step, len(self.path_vis)-1)
            ar, ac = self.path_vis[step]
            cx, cy = ac*cell+cell//2, ar*cell+cell//2
            rd = max(4, cell//3)
            pygame.draw.circle(surf, AGENT_C, (cx, cy), rd)
            pygame.draw.circle(surf, PATH_C,  (cx, cy), rd, 2)

        # S / T labels
        fs  = max(10, cell-8)
        fnt = pygame.font.SysFont('consolas', fs, bold=True)
        for (lr, lc), lbl, bg in [
            (pf.start,  'S', START_C),
            (pf.target, 'T', GOAL_C)
        ]:
            cx, cy = lc*cell+cell//2, lr*cell+cell//2
            s  = fnt.render(lbl, True, TEXT_LIGHT)
            rb = s.get_rect(center=(cx, cy)).inflate(6, 4)
            pygame.draw.rect(surf, bg, rb, border_radius=4)
            surf.blit(s, s.get_rect(center=(cx, cy)))

    def _draw_panel(self):
        px   = self.grid_w
        surf = self.screen

        pygame.draw.rect(surf, PANEL_BG, (px, 0, PANEL_W, self.win_h))
        pygame.draw.line(surf, ACCENT,   (px, 0), (px, self.win_h), 2)

        # Section labels
        labels = {
            'algo':    'Algorithm',
            'heur':    'Heuristic',
            'place':   'Click Mode',
            'density': 'Obstacle Density',
        }
        for key, text in labels.items():
            draw_text(surf, self.font_sm, text,
                      (px+10, self.lbl_y[key]), ACCENT)

        # Radio groups
        self.rg_algo.draw(surf)
        self.rg_heur.draw(surf)
        self.rg_place.draw(surf)

        # Density control
        self.btn_dm.draw(surf, self.font_med)
        self.btn_dp.draw(surf, self.font_med)
        pygame.draw.rect(surf, BTN_NORMAL, self.density_rect, border_radius=5)
        fw = int(self.density_rect.w * (self.density-0.05) / 0.65)
        if fw > 0:
            pygame.draw.rect(surf, BTN_ACTIVE,
                             (self.density_rect.x, self.density_rect.y,
                              fw, self.density_rect.h), border_radius=5)
        pt = self.font_sm.render(f'{int(self.density*100)}%', True, TEXT_DARK)
        surf.blit(pt, pt.get_rect(center=self.density_rect.center))

        # Action buttons
        for b in [self.btn_gen, self.btn_clear, self.btn_run,
                  self.btn_dyn, self.btn_reset]:
            b.draw(surf, self.font_med)

        # ── Metrics ───────────────────────────────────────────────────────────
        my = self.metrics_y
        pygame.draw.line(surf, DIVIDER, (px+8, my-5), (px+PANEL_W-8, my-5))
        draw_text(surf, self.font_lg, '  Metrics', (px+10, my), ACCENT); my += 22

        for line in [
            f"Algorithm : {self.algorithm}",
            f"Heuristic : {self.heuristic}",
            "─" * 22,
            f"Nodes     : {self.metrics['nodes']}",
            f"Path Cost : {self.metrics['cost']}",
            f"Time (ms) : {self.metrics['time_ms']:.2f}",
            f"Re-plans  : {self.total_replans}",
        ]:
            draw_text(surf, self.font_sm, line, (px+10, my), TEXT_DARK)
            my += 17

        # ── Legend ────────────────────────────────────────────────────────────
        my += 10
        pygame.draw.line(surf, DIVIDER, (px+8, my-5), (px+PANEL_W-8, my-5))
        draw_text(surf, self.font_lg, '  Legend', (px+10, my), ACCENT); my += 20

        for color, label in [
            (VISITED_C,  'Visited  (expanded)'),
            (FRONTIER_C, 'Frontier (queue)'),
            (PATH_C,     'Final Path'),
            (WALL_C,     'Wall'),
            (START_C,    'Start  (S)'),
            (GOAL_C,     'Goal   (T)'),
        ]:
            pygame.draw.rect(surf, color, (px+10, my, 14, 14), border_radius=3)
            draw_text(surf, self.font_sm, label, (px+30, my), TEXT_DARK)
            my += 18

        # Tip
        draw_text(surf, self.font_sm,
                  'Drag to draw walls  |  ESC = quit',
                  (px+10, self.win_h-18), DIVIDER)


# ══════════════════════════════════════════════════════════════════════════════
#  Startup
# ══════════════════════════════════════════════════════════════════════════════
def get_grid_size():
    print('\n' + '='*52)
    print('   Dynamic Pathfinding Agent  —  Pygame')
    print('='*52)
    while True:
        try:
            r = input('  Rows   (5-30, press Enter for 15): ').strip()
            c = input('  Cols   (5-30, press Enter for 20): ').strip()
            rows = int(r) if r else 15
            cols = int(c) if c else 20
            if 5 <= rows <= 30 and 5 <= cols <= 30:
                return rows, cols
            print('  Enter values between 5 and 30.')
        except ValueError:
            print('  Integers only.')


def main():
    rows, cols = get_grid_size()
    print(f'\n  Grid: {rows} x {cols}')
    print('  Controls:')
    print('  • Select Click Mode → click/drag grid')
    print('  • Dynamic Mode ON + Run = live obstacles + re-planning')
    print('  • ESC to quit')
    print('='*52)
    app = App(rows, cols)
    app.run()


if __name__ == '__main__':
    main()
