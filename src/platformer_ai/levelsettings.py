"""Settings of the level generator: one model, filled from a TOML file and/or command-line options.

Every setting has a name.  In a settings file it is a TOML key; on the command line the common ones have their own
option (`--gap 2-3`) and `--set NAME=VALUE` reaches every one of them, including the per-tier and test-set ones
(`--set tier.3.riders=2`, `--set test.tiers=1,2,3`).  So a file and a command line can say exactly the same things.

    out, seed, train, test, test_seed, verify      what to write (run settings, top level only)
    tiers                                          which tiers, in which shares: {1 = 1, 2 = 1, 3 = 1} or [3]
    width, height, gap, stone_gap                  ranges [lo, hi] (a single number = exactly that)
    max_step, max_enemies                          numbers
    weight.KIND, limit.KIND, min.KIND              per segment kind (KIND: see KINDS)
    only, without                                  lists of segment kinds
    coins, riders, pairs, narrow_stones            how often (see TUNING)

Generation settings (everything but the run settings) can be given at the top (all tiers), in [tier.N] (one tier),
in [test_set] (the held-out set only; also [test_set.tier.N]).  The more specific place wins: built-in tier defaults <
top < [tier.N] < [test_set] < [test_set.tier.N].  The command line wins over the file for the same name.

weight.KIND multiplies the tier's built-in weight (2 = twice as often, 0 = never).  A kind a tier does not have
(tier 1 has no stones / upper) gets a base weight of 10 when you name it in weight or only.
"""
from __future__ import annotations

import copy
import difflib
import tomllib
from dataclasses import dataclass, field

KINDS = ("flat", "step", "gap", "enemy", "ledge", "ferry", "stones", "upper")
KIND_HELP = {
    "flat": "flat ground, often with coins",
    "step": "the ground goes up (at most max_step) or down (up to 3)",
    "gap": "a pit to jump over (gap tiles wide)",
    "enemy": "a stretch with a patrolling enemy (two of them with chance `pairs`)",
    "ledge": "a one-way ledge 2 above the ground with coins (an enemy on it with chance from `riders`)",
    "ferry": "a moving platform over a wide pit (an enemy riding it with chance from `riders`)",
    "stones": "one-way stepping stones over a pit (narrow 1-tile stones with chance `narrow_stones`)",
    "upper": "a moving platform 2 above the ground with a coin (an enemy riding it with chance from `riders`)",
}
TIER_IDS = (1, 2, 3)

# The built-in tiers.  These are exactly the generator's behaviour without any settings.
TIER_DEFAULTS = {
    1: dict(width=(40, 55), height=(10, 12),
            weights=dict(flat=25, step=20, gap=30, enemy=15, ledge=10, ferry=10, stones=0, upper=0),
            limit=dict(enemy=1, ferry=1), min={}, gap=(1, 2), max_step=1, max_enemies=None,
            coins=1.0, riders=0.0, pairs=0.0, narrow_stones=0.0, stone_gap=(2, 2)),
    2: dict(width=(50, 70), height=(10, 12),
            weights=dict(flat=10, step=15, gap=22, enemy=25, ledge=10, ferry=10, stones=8, upper=7),
            limit=dict(ferry=2), min={}, gap=(1, 3), max_step=2, max_enemies=None,
            coins=1.0, riders=0.0, pairs=0.0, narrow_stones=0.0, stone_gap=(2, 2)),
    3: dict(width=(60, 80), height=(10, 12),
            weights=dict(flat=5, step=10, gap=20, enemy=20, ledge=10, ferry=15, stones=10, upper=10),
            limit=dict(ferry=3), min={}, gap=(2, 3), max_step=2, max_enemies=None,
            coins=1.0, riders=1.0, pairs=0.4, narrow_stones=0.3, stone_gap=(2, 3)),
}
RUN_DEFAULTS = dict(out="gen", seed=0, train=300, test=60, test_seed=None, verify=True, tiers={1: 1, 2: 1, 3: 1})
TEST_SEED_OFFSET = 1_000_003

# What each tuning number means (also the --help text)
TUNING = {
    "coins": "coin amount: multiplies every coin chance (0 = no coins, 1 = normal, 2 = about twice as many)",
    "riders": "enemies on ledges / riding platforms: 0 = never, 1 = the tier-3 amount, 2 = twice that",
    "pairs": "chance that an enemy stretch has two enemies (0..1)",
    "narrow_stones": "chance that stepping stones are 1 tile wide instead of 2 (0..1)",
}

# Physics limits (env.py: the feet rise ~2.2 tiles, a jump covers ~4 tiles; see levelgen.py)
LIMITS = dict(gap=(1, 4), stone_gap=(1, 3), max_step=(0, 2), width=(20, 400), height=(8, 40))


class SettingsError(ValueError):
    """A setting that does not exist or has a value that cannot work; the message says which and why."""


# ---- value types ------------------------------------------------------------------------------------------------
def _num(v, what):
    if isinstance(v, bool):
        raise SettingsError(f"{what}: expected a number, got {v!r}")
    if isinstance(v, (int, float)):
        return v
    try:
        return int(v)
    except (TypeError, ValueError):
        pass
    try:
        return float(v)
    except (TypeError, ValueError):
        raise SettingsError(f"{what}: expected a number, got {v!r}") from None


def _int(v, what):
    x = _num(v, what)
    if float(x) != int(x):
        raise SettingsError(f"{what}: expected a whole number, got {v!r}")
    return int(x)


def _range(v, what):
    """'2-3', '3', 3, [2, 3] -> (2, 3)."""
    if isinstance(v, str) and "-" in v.strip()[1:]:
        lo, hi = v.split("-", 1)
        v = [lo, hi]
    if isinstance(v, (list, tuple)):
        if len(v) != 2:
            raise SettingsError(f"{what}: a range needs two numbers [lo, hi], got {v!r}")
        lo, hi = _int(v[0], what), _int(v[1], what)
    else:
        lo = hi = _int(v, what)
    if lo > hi:
        raise SettingsError(f"{what}: the range {lo}-{hi} goes backwards")
    return (lo, hi)


def _kind(k, what):
    if k not in KINDS:
        raise SettingsError(f"{what}: no segment kind {k!r}{_did_you_mean(k, KINDS)}.  Kinds: {', '.join(KINDS)}")
    return k


def _kinds(v, what):
    if isinstance(v, str):
        v = [x for x in v.replace(" ", "").split(",") if x]
    if not isinstance(v, (list, tuple)):
        raise SettingsError(f"{what}: expected a list of segment kinds, got {v!r}")
    return [_kind(str(k), what) for k in v]


def _kv(v, what, value):
    """'enemy=2,gap=0.5' or a table -> {kind: value}."""
    if isinstance(v, str):
        out = {}
        for part in [p for p in v.replace(" ", "").split(",") if p]:
            if "=" not in part:
                raise SettingsError(f"{what}: expected KIND=VALUE, got {part!r}")
            k, x = part.split("=", 1)
            out[k] = x
        v = out
    if not isinstance(v, dict):
        raise SettingsError(f"{what}: expected KIND=VALUE pairs, got {v!r}")
    return {_kind(str(k), what): value(x, f"{what}.{k}") for k, x in v.items()}


def _tiers(v, what):
    """'3', '2,3', '1=1,3=2', [3], {1 = 1, 3 = 2} -> {tier: share}."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = [v]
    if isinstance(v, str):
        parts = [p for p in v.replace(" ", "").split(",") if p]
        v = dict(p.split("=", 1) if "=" in p else (p, 1) for p in parts)
    if isinstance(v, (list, tuple)):
        v = {t: 1 for t in v}
    if not isinstance(v, dict) or not v:
        raise SettingsError(f"{what}: expected tiers like [3], '2,3' or {{1 = 1, 3 = 2}}, got {v!r}")
    out = {}
    for t, share in v.items():
        t = _int(t, what)
        if t not in TIER_IDS:
            raise SettingsError(f"{what}: there is no tier {t} (tiers: {', '.join(map(str, TIER_IDS))})")
        share = _num(share, f"{what}.{t}")
        if share < 0:
            raise SettingsError(f"{what}.{t}: a share cannot be negative")
        out[t] = share
    if sum(out.values()) <= 0:
        raise SettingsError(f"{what}: all shares are 0")
    return dict(sorted(out.items()))


def _bool(v, what):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    raise SettingsError(f"{what}: expected true or false, got {v!r}")


def _str(v, what):
    if not isinstance(v, str):
        raise SettingsError(f"{what}: expected text, got {v!r}")
    return v


def _opt_int(v, what):
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "none", "off")):
        return None
    return _int(v, what)


def _nonneg(v, what):
    x = float(_num(v, what))
    if x < 0:
        raise SettingsError(f"{what}: cannot be negative ({x:g})")
    return x


def _chance(v, what):
    x = _nonneg(v, what)
    if x > 1:
        raise SettingsError(f"{what}: a chance is at most 1 ({x:g})")
    return x


GEN_TYPES = {
    "width": _range, "height": _range, "gap": _range, "stone_gap": _range,
    "max_step": _int, "max_enemies": _opt_int,
    "weight": lambda v, w: _kv(v, w, _nonneg), "limit": lambda v, w: _kv(v, w, _int),
    "min": lambda v, w: _kv(v, w, _int),
    "only": _kinds, "without": _kinds,
    "coins": _nonneg, "riders": _nonneg, "pairs": _chance, "narrow_stones": _chance,
}
RUN_TYPES = {"out": _str, "seed": _int, "train": _int, "test": _int, "test_seed": _opt_int, "verify": _bool}
MAP_KEYS = ("weight", "limit", "min")          # settings that are tables of kind -> value (merged per kind)


def _did_you_mean(name, options) -> str:
    close = difflib.get_close_matches(str(name), [str(o) for o in options], n=1)
    return f" (did you mean {close[0]!r}?)" if close else ""


# ---- normalising a raw settings table (from a file or the command line) ----------------------------------------
def _norm_gen(raw: dict, where: str, allow: tuple = ()) -> dict:
    out = {}
    for k, v in raw.items():
        name = f"{where}{k}"
        if k in GEN_TYPES:
            out[k] = GEN_TYPES[k](v, name)
        elif k == "tiers" and "tiers" in allow:
            out[k] = _tiers(v, name)
        elif k == "tier" and "tier" in allow:
            if not isinstance(v, dict):
                raise SettingsError(f"{name}: expected sections [{name}.N]")
            out[k] = {}
            for t, sub in v.items():
                t_int = _int(t, f"{name}.{t}")
                if t_int not in TIER_IDS:
                    raise SettingsError(f"{name}.{t}: there is no tier {t} (tiers: {', '.join(map(str, TIER_IDS))})")
                if not isinstance(sub, dict):
                    raise SettingsError(f"{name}.{t}: expected a section of settings")
                out[k][t_int] = _norm_gen(sub, f"{name}.{t}.")
        else:
            valid = list(GEN_TYPES) + list(allow)
            raise SettingsError(f"unknown setting {name!r}{_did_you_mean(k, valid)}.  Here: {', '.join(valid)}")
    return out


def normalize(raw: dict) -> dict:
    """Check and convert a raw settings table (TOML or from options).  Raises SettingsError."""
    out = {}
    for k, v in raw.items():
        if k in RUN_TYPES:
            out[k] = RUN_TYPES[k](v, k)
        elif k == "test_set":
            if not isinstance(v, dict):
                raise SettingsError("test_set: expected a section [test_set] of generation settings")
            out[k] = _norm_gen(v, "test_set.", allow=("tiers", "tier"))
        else:
            out.update(_norm_gen({k: v}, "", allow=("tiers", "tier")))
    return out


def merge(base: dict, over: dict) -> dict:
    """over wins; tables of kinds (weight / limit / min), [tier.N] and [test] merge key by key."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in MAP_KEYS and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        elif k in ("tier",) and isinstance(out.get(k), dict):
            for t, sub in v.items():
                out[k][t] = merge(out[k].get(t, {}), sub)
        elif k == "test_set" and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_file(path: str) -> dict:
    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except OSError as e:
        raise SettingsError(f"cannot read the settings file {path}: {e.strerror}") from None
    except tomllib.TOMLDecodeError as e:
        raise SettingsError(f"the settings file {path} is not valid TOML: {e}") from None
    try:
        return normalize(raw)
    except SettingsError as e:
        raise SettingsError(f"{path}: {e}") from None


def parse_set(item: str) -> dict:
    """'--set NAME=VALUE' -> a raw nested table, e.g. 'tier.3.weight.enemy=2' -> {'tier': {'3': {'weight': ...}}}."""
    if "=" not in item:
        raise SettingsError(f"--set {item!r}: expected NAME=VALUE (e.g. --set tier.3.gap=2-3)")
    name, value = item.split("=", 1)
    keys = [k for k in name.strip().split(".") if k]
    if not keys:
        raise SettingsError(f"--set {item!r}: no name before '='")
    value = value.strip()
    try:                                     # a TOML value if it is one ([2, 3], true, "text", 2.5), else the text
        value = tomllib.loads(f"v = {value}")["v"]
    except tomllib.TOMLDecodeError:
        pass
    raw: dict = {}
    node = raw
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value
    return raw


# ---- resolving: settings -> what one tier of one set looks like --------------------------------------------------
@dataclass
class TierConfig:
    tier: int
    width: tuple
    height: tuple
    weights: dict
    limit: dict = field(default_factory=dict)
    min: dict = field(default_factory=dict)
    gap: tuple = (1, 3)
    max_step: int = 2
    max_enemies: int | None = None
    coins: float = 1.0
    riders: float = 0.0
    pairs: float = 0.0
    narrow_stones: float = 0.0
    stone_gap: tuple = (2, 2)

    def describe(self) -> str:
        w = ", ".join(f"{k} {v:g}" for k, v in self.weights.items() if v > 0)
        extra = [f"limit {self.limit}" if self.limit else "", f"min {self.min}" if self.min else "",
                 f"max_enemies {self.max_enemies}" if self.max_enemies is not None else ""]
        return (f"tier {self.tier}: width {self.width[0]}-{self.width[1]}, height {self.height[0]}-{self.height[1]}, "
                f"gap {self.gap[0]}-{self.gap[1]}, max_step {self.max_step}, coins {self.coins:g}, "
                f"riders {self.riders:g}, pairs {self.pairs:g}, narrow_stones {self.narrow_stones:g}, "
                f"stone_gap {self.stone_gap[0]}-{self.stone_gap[1]}; weights: {w}"
                + "".join(f"; {x}" for x in extra if x))


def _layers(s: dict, part: str, tier: int) -> list[dict]:
    gen = {k: v for k, v in s.items() if k in GEN_TYPES}
    out = [gen, s.get("tier", {}).get(tier, {})]
    if part == "test":
        t = s.get("test_set", {})
        out += [{k: v for k, v in t.items() if k in GEN_TYPES}, t.get("tier", {}).get(tier, {})]
    return out


def resolve(s: dict, part: str, tier: int) -> TierConfig:
    """The full configuration of `tier` in the 'train' or 'test' set.  Raises SettingsError if it cannot work."""
    d = copy.deepcopy(TIER_DEFAULTS[tier])
    mult, only, without = {}, None, None
    for layer in _layers(s, part, tier):
        for k, v in layer.items():
            if k == "weight":
                mult.update(v)
            elif k == "only":
                only = v
            elif k == "without":
                without = v
            elif k in ("limit", "min"):
                d[k] = {**d[k], **v}
            else:
                d[k] = v
    named = set(mult) | set(only or ())
    weights = {}
    for k in KINDS:
        base = d["weights"].get(k, 0) or (10 if k in named else 0)   # named but not in this tier: a normal amount
        w = base * mult.get(k, 1.0)
        if (only is not None and k not in only) or (without and k in without):
            w = 0
        weights[k] = w
    cfg = TierConfig(tier=tier, width=tuple(d["width"]), height=tuple(d["height"]), weights=weights,
                     limit=d["limit"], min=d["min"], gap=tuple(d["gap"]), max_step=d["max_step"],
                     max_enemies=d["max_enemies"], coins=d["coins"], riders=d["riders"], pairs=d["pairs"],
                     narrow_stones=d["narrow_stones"], stone_gap=tuple(d["stone_gap"]))
    _check(cfg, part)
    return cfg


def _check(c: TierConfig, part: str) -> None:
    where = f"{part} set, tier {c.tier}"
    for name in ("gap", "stone_gap", "width", "height"):
        lo, hi = LIMITS[name]
        v = getattr(c, name)
        if v[0] < lo or v[1] > hi:
            why = {"gap": "a jump covers about 4 tiles (4 is already at the very edge)",
                   "stone_gap": "gaps between stepping stones: 1 to 3",
                   "width": "room for the spawn, one segment and the goal",
                   "height": "room above the ground for ledges, riders and coins"}[name]
            raise SettingsError(f"{where}: {name} {v[0]}-{v[1]} is outside {lo}-{hi} ({why})")
    lo, hi = LIMITS["max_step"]
    if not lo <= c.max_step <= hi:
        raise SettingsError(f"{where}: max_step {c.max_step} is outside {lo}-{hi} (the jump rises about 2.2 tiles)")
    if c.max_enemies is not None and c.max_enemies < 0:
        raise SettingsError(f"{where}: max_enemies cannot be negative")
    if sum(c.weights.values()) <= 0:
        raise SettingsError(f"{where}: every segment kind is switched off (only / without / weight)")
    for k, n in c.min.items():
        if n < 0:
            raise SettingsError(f"{where}: min.{k} cannot be negative")
        if n > 0 and c.weights[k] <= 0:
            raise SettingsError(f"{where}: min.{k} = {n}, but {k} is switched off")
        if n > c.limit.get(k, 99):
            raise SettingsError(f"{where}: min.{k} = {n} is more than limit.{k} = {c.limit[k]}")
    if c.max_enemies is not None and c.max_enemies < c.min.get("enemy", 0):
        raise SettingsError(f"{where}: max_enemies {c.max_enemies} is less than min.enemy {c.min['enemy']}")


def run_settings(s: dict) -> dict:
    """out / seed / train / test / test_seed / verify / tiers (train and test), with the defaults filled in."""
    r = {**RUN_DEFAULTS, **{k: v for k, v in s.items() if k in RUN_TYPES or k == "tiers"}}
    if r["test_seed"] is None:
        r["test_seed"] = r["seed"] + TEST_SEED_OFFSET
    r["test_tiers"] = s.get("test_set", {}).get("tiers", r["tiers"])
    for k in ("train", "test"):
        if r[k] < 0:
            raise SettingsError(f"{k} cannot be negative")
    return r


def tier_plan(n: int, shares: dict) -> list[int]:
    """n tiers in the given shares (largest remainder), interleaved: {1: 1, 2: 1, 3: 1} -> 1, 2, 3, 1, 2, 3, ..."""
    tiers = [t for t, s in shares.items() if s > 0]
    total = sum(shares[t] for t in tiers)
    exact = {t: n * shares[t] / total for t in tiers}
    count = {t: int(exact[t]) for t in tiers}
    for t in sorted(tiers, key=lambda t: (-(exact[t] - count[t]), t))[: n - sum(count.values())]:
        count[t] += 1
    slots = [((k + 0.5) / count[t], t) for t in tiers for k in range(count[t])]
    return [t for _, t in sorted(slots)]


# ---- writing a settings file --------------------------------------------------------------------------------------
def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{k} = {_toml_value(x)}" for k, x in v.items()) + " }" if v else "{}"
    raise TypeError(v)


def _gen_lines(g: dict) -> list[str]:
    return [f"{k} = {_toml_value(list(v) if isinstance(v, tuple) else v)}" for k, v in g.items()
            if (k in GEN_TYPES or k == "tiers") and v is not None]


def dump(s: dict, resolved: list[TierConfig] | None = None) -> str:
    """A settings file that gives exactly these settings (run settings filled in), optionally with the resolved
    tiers as comments."""
    r = run_settings(s)
    lines = ["# level generator settings (python -m platformer_ai.levelgen --config THIS_FILE)"]
    for k in ("out", "seed", "train", "test", "test_seed", "verify"):
        lines.append(f"{k} = {_toml_value(r[k])}")
    lines.append(f"tiers = {_toml_value(r['tiers'])}")
    lines += _gen_lines({k: v for k, v in s.items() if k in GEN_TYPES})
    for t, sub in sorted(s.get("tier", {}).items()):
        lines += ["", f"[tier.{t}]"] + _gen_lines(sub)
    test = s.get("test_set", {})
    top = {k: v for k, v in test.items() if k != "tier"}
    if top or test.get("tier"):
        lines += ["", "[test_set]"] + _gen_lines(top)
        for t, sub in sorted(test.get("tier", {}).items()):
            lines += ["", f"[test_set.tier.{t}]"] + _gen_lines(sub)
    if resolved:
        lines += ["", "# resolved (built-in tier defaults + the settings above):"]
        lines += [f"#   {c.describe()}" for c in resolved]
    return "\n".join(lines) + "\n"
