"""pygame window for the platformer: watch the trained agent, or take over with the keyboard.

  uv run python -m platformer_ai.viewer --params runs/latest/params.msgpack

The window only READS the JAX env (one env, one jitted step per tick, on the CPU) - training is untouched.
Keys:  Tab agent <-> you | arrows / A D move, Space / Up / W jump | P pause, . single step | R restart | Esc quit
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")      # the viewer must not touch the GPU that is training
                                                    # (override with:  JAX_PLATFORMS=cuda uv run ... )
import argparse
import functools
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from flax import serialization

from platformer_ai import env as E
from platformer_ai import level as L
from platformer_ai import scene
from platformer_ai import terrain
from platformer_ai import viewlogic as V
from platformer_ai.ppo import ActorCritic

BANNER_SECONDS = 0.9
WHITE, SHADOW, YELLOW = (255, 255, 255), (0, 0, 0), (255, 220, 80)


def load_agent(path: str, hidden: int):
    """-> (net, params) or (net, None) if the file is missing (then you play yourself)."""
    net = ActorCritic(E.N_ACTIONS, hidden)
    template = net.init(jax.random.PRNGKey(0), jnp.zeros((E.OBS_DIM,), jnp.float32))
    p = Path(path)
    if not p.is_file():
        print(f"no params at {p}: starting in keyboard mode (train first to watch the agent)")
        return net, None
    return net, serialization.from_bytes(template, p.read_bytes())


class Snap:
    """The few numbers the picture needs, as plain floats (so drawing never touches jax)."""

    def __init__(self, s: E.EnvState):
        g = jax.device_get(s)
        self.x, self.y, self.vx, self.vy = float(g.x), float(g.y), float(g.vx), float(g.vy)
        self.plat_x = np.asarray(g.plat_x, dtype=float)             # one entry per moving platform
        self.enemy_x = np.asarray(g.enemy_x, dtype=float)           # one entry per enemy
        self.enemy_dir = np.asarray(g.enemy_dir, dtype=float)
        self.coins_alive = np.asarray(g.coins_alive, dtype=bool)
        self.ret = float(g.ep_return)

    @property
    def coins(self) -> int:
        return int((~self.coins_alive).sum())


class Viewer:
    def __init__(self, args):
        self.args = args
        self.assets = scene.load_assets(args.assets, args.theme, not args.no_decor)
        self.net, self.params = load_agent(args.params, args.hidden)
        self.human = self.params is None or args.human
        net = self.net

        @jax.jit
        def act(params, obs, key):
            logits, _ = net.apply(params, obs)
            return jax.random.categorical(key, logits) if args.sample else jnp.argmax(logits)

        self.act = act
        self.step = jax.jit(functools.partial(E.step, random_phases=args.random_phases))
        self.key = jax.random.PRNGKey(args.seed)

        pygame.init()
        pygame.display.set_caption("platformer-ai   Tab: agent/you   P: pause   R: restart   Esc: quit")
        self.screen = pygame.display.set_mode((V.SCREEN_W * args.scale, V.SCREEN_H * args.scale), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font_path = str(Path(args.assets) / "fonts" / "PixelOperator8-Bold.ttf")
        self.font_scale, self.font = 0, None

        self.cam = V.Camera()
        self.player_anim = V.Animator(V.ANIMS["player"])
        self.enemy_anim = V.Animator(V.ANIMS["enemy"])
        self.coin_anim = V.Animator(V.ANIMS["coin"])
        self.face_left = False
        self.paused = False
        self.jump_latch = False
        self.banner, self.hold_until = "", 0.0
        self.restart()

    # ---- simulation ----------------------------------------------------------------------------
    def restart(self):
        self.key, k = jax.random.split(self.key)
        self.state = E.reset(k, random_phases=self.args.random_phases)
        self.obs = E.observe(self.state)
        self.prev = self.cur = Snap(self.state)
        self.cam.snap(*V.to_px(self.cur.x, self.cur.y))
        self.banner, self.hold_until = "", 0.0

    def keyboard_action(self) -> int:
        k = pygame.key.get_pressed()
        left = k[pygame.K_LEFT] or k[pygame.K_a]
        right = k[pygame.K_RIGHT] or k[pygame.K_d]
        jump = k[pygame.K_SPACE] or k[pygame.K_UP] or k[pygame.K_w] or self.jump_latch
        self.jump_latch = False                         # a short tap still counts for one tick
        return V.keys_to_action(bool(left), bool(right), bool(jump))

    def sim_tick(self, now: float):
        if self.human:
            action = self.keyboard_action()
        else:
            self.key, k = jax.random.split(self.key)
            action = int(self.act(self.params, self.obs, k))
        self.prev = self.cur
        self.state, self.obs, _, done, info = self.step(self.state, action)
        self.cur = Snap(self.state)
        if bool(done):                                  # the env already auto-reset: show the last frame + banner
            if bool(info["goal"]):
                self.banner = "Goal reached!"
            elif bool(info["truncated"]):
                self.banner = "Time is up"
            else:
                self.banner = "Game over"
            self.hold_until = now + BANNER_SECONDS      # self.prev (the last picture) stays on screen meanwhile

    # ---- drawing -------------------------------------------------------------------------------
    def fonts(self, s: int):
        if s != self.font_scale:
            self.font_scale = s
            self.font = pygame.font.Font(self.font_path, 8 * s)
        return self.font

    def text(self, surf, s, msg, x, y, color=WHITE, shadow=True):
        f = self.fonts(s)
        if shadow:
            surf.blit(f.render(msg, False, SHADOW), (int(x) + s, int(y) + s))
        surf.blit(f.render(msg, False, color), (int(x), int(y)))

    def draw(self, now: float, alpha: float, dt: float):
        a, b = self.prev, self.cur
        frozen = now < self.hold_until
        if frozen:                                       # banner: freeze on the last picture of the episode
            a = b = self.prev
        x = V.lerp_pos(a.x, b.x, alpha)
        y = V.lerp_pos(a.y, b.y, alpha)
        plat_x = [V.lerp_pos(p, q, alpha) for p, q in zip(a.plat_x, b.plat_x)]
        enemy_x = [V.lerp_pos(p, q, alpha) for p, q in zip(a.enemy_x, b.enemy_x)]
        self.face_left = V.facing_left(b.vx, self.face_left)
        px, py = V.to_px(x, y)
        self.cam.update(px, py, dt)

        frame = scene.compose(
            self.assets, self.cam.top_left(),
            player_xy=(x, y), player_flip=self.face_left,
            player_frame=self.player_anim.frame(V.player_anim(b.vx, b.vy), now),
            enemy_x=enemy_x, enemy_flip=list(b.enemy_dir < 0),
            enemy_frame=self.enemy_anim.frame("walk", now),
            coin_frame=self.coin_anim.frame("spin", now),
            coins_alive=b.coins_alive, plat_x=plat_x)

        surf = self.screen = pygame.display.get_surface()
        w, h = surf.get_size()
        s = max(1, min(w // V.SCREEN_W, h // V.SCREEN_H))        # integer scale: crisp pixels
        img = pygame.image.frombuffer(frame.tobytes(), (V.SCREEN_W, V.SCREEN_H), "RGB")
        img = pygame.transform.scale(img, (V.SCREEN_W * s, V.SCREEN_H * s))
        ox, oy = (w - V.SCREEN_W * s) // 2, (h - V.SCREEN_H * s) // 2
        surf.fill((0, 0, 0))
        surf.blit(img, (ox, oy))

        cx, cy = self.cam.top_left()

        def to_screen(wx, wy):
            return ox + (wx - cx) * s, oy + (wy - cy) * s

        f = self.fonts(s)
        for msg, left, top in V.world_labels():                   # signs of the level (level.LABELS)
            sx, sy = to_screen(left, top)
            self.text(surf, s, msg, sx, sy, shadow=False)
        anchor = V.score_label_anchor()                           # "You collected N coins." above the goal flag
        if anchor is not None:
            lines = V.wrap_words(V.score_text(b.coins), V.SCORE_LABEL_WIDTH, lambda t: f.size(t)[0] / s)
            for i, line in enumerate(lines):
                lw = f.size(line)[0]
                sx, sy = to_screen(anchor[0], anchor[1] - (len(lines) - i) * 9)
                self.text(surf, s, line, sx - lw / 2, sy, shadow=False)

        # HUD (screen space)
        self.text(surf, s, f"Coins {b.coins}/{L.N_COINS}", ox + 4 * s, oy + 4 * s, YELLOW)
        mode = "YOU" if self.human else "AGENT"
        self.text(surf, s, f"{mode}   return {b.ret:.1f}", ox + 4 * s, oy + 14 * s)
        if self.paused:
            self.text(surf, s, "PAUSED", ox + 4 * s, oy + 24 * s, YELLOW)
        if frozen and self.banner:
            lw = f.size(self.banner)[0]
            self.text(surf, s, self.banner, ox + (V.SCREEN_W * s - lw) // 2, oy + 40 * s, YELLOW)
        pygame.display.flip()

    # ---- main loop -----------------------------------------------------------------------------
    def run(self):
        acc, last = 0.0, time.perf_counter()
        running = True
        while running:
            now = time.perf_counter()
            dt = min(now - last, 0.25)
            last = now
            single = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif ev.key == pygame.K_p:
                        self.paused = not self.paused
                    elif ev.key == pygame.K_PERIOD:
                        single, self.paused = True, True
                    elif ev.key == pygame.K_r:
                        self.restart()
                    elif ev.key == pygame.K_TAB and self.params is not None:
                        self.human = not self.human
                    elif ev.key in (pygame.K_SPACE, pygame.K_UP, pygame.K_w):
                        self.jump_latch = True

            if now >= self.hold_until:
                if self.banner:                                    # banner over: continue in the fresh episode
                    self.banner = ""
                    self.prev = self.cur
                    self.cam.snap(*V.to_px(self.cur.x, self.cur.y))
                if not self.paused or single:
                    acc += DT_SIM if single else dt
                    while acc >= DT_SIM and now >= self.hold_until:
                        self.sim_tick(now)
                        acc -= DT_SIM
                        if self.banner:                            # episode ended: stop simulating until the banner is over
                            acc = 0.0
            alpha = 1.0 if (self.paused or now < self.hold_until) else acc / DT_SIM
            self.draw(now, alpha, dt)
            self.clock.tick(60)
        pygame.quit()


DT_SIM = E.DT


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default="runs/latest/params.msgpack")
    ap.add_argument("--assets", default="assets", help="folder with sprites/ and fonts/ (Brackeys bundle)")
    ap.add_argument("--scale", type=int, default=3, help="initial window = 480x270 times this (resizable)")
    ap.add_argument("--human", action="store_true", help="start with keyboard control")
    ap.add_argument("--sample", action="store_true", help="agent samples actions instead of argmax")
    ap.add_argument("--random-phases", action="store_true", help="random platform/enemy phases and jittered spawn")
    ap.add_argument("--theme", default="grass", choices=sorted(terrain.THEMES), help="ground material")
    ap.add_argument("--no-decor", action="store_true", help="no bushes / mushrooms on the ground")
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    v = Viewer(args)
    print("compiling the env on the CPU (a few seconds) ...")
    v.step(v.state, 0)                                    # warm-up so the window does not stutter at the start
    if v.params is not None:
        v.act(v.params, v.obs, v.key)
    v.run()


if __name__ == "__main__":
    main()
