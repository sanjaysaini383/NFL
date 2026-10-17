"""SYNTHETIC data with the same schema as the Big Data Bowl 2027 files.

Used only for unit tests, CI and the app's demo mode. None of these numbers are
real players or real measurements. The competition data (CC BY-NC 4.0) is never
stored in this repository.

Each fake player has a latent "engine" (top speed S0, acceleration time
constant tau, braking capacity) and a latent "expression" fraction that controls
how much of that engine shows up in games. Career outcomes depend on both, so the
pipeline has a known signal to recover.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

POS_PARAMS = {
    #       S0 mean (yd/s), tau mean (s), weight (lb), height (in)
    "WR": (10.2, 1.05, 200, 72),
    "DB": (10.1, 1.05, 195, 71),
    "TE": (9.5, 1.15, 250, 76),
    "DL": (8.9, 1.20, 295, 75),
    "OL": (8.2, 1.30, 315, 77),
}
NFL_POS = {"WR": ["WR"], "DB": ["CB", "SS", "FS"], "TE": ["TE"], "DL": ["DE", "DT", "OLB"], "OL": ["T", "G", "C"]}
T0 = pd.Timestamp("2025-03-01 10:00:00", tz="UTC")


def _iso(seconds: np.ndarray) -> list[str]:
    ts = T0 + pd.to_timedelta(seconds, unit="s")
    return [t.strftime("%Y-%m-%dT%H:%M:%S.%f") for t in ts]


def _sprint(s0, tau, dist=40.0, pre=0.8, post=1.5, rng=None):
    """Mono-exponential sprint over `dist` yards, plus standing and slowing phases."""
    t_run = np.arange(0, 8, config.DT)
    v = s0 * (1 - np.exp(-t_run / tau))
    x = s0 * (t_run - tau * (1 - np.exp(-t_run / tau)))
    end = int(np.searchsorted(x, dist)) + 1
    v, x = v[:end], x[:end]
    # slow down after the line
    n_post = int(post / config.DT)
    v_post = np.maximum(v[-1] - 6.0 * config.DT * np.arange(1, n_post + 1), 0.5)
    x_post = x[-1] + np.cumsum(v_post) * config.DT
    n_pre = int(pre / config.DT)
    v_all = np.concatenate([np.zeros(n_pre), v, v_post])
    x_all = np.concatenate([np.zeros(n_pre), x, x_post])
    noise = rng.normal(0, 0.03, x_all.size) if rng is not None else 0
    return x_all + noise, v_all


def _waypoint_run(points, s0, a0, dmax, rng):
    """Point-to-point run with an acceleration cap that falls with speed and a braking cap."""
    pos = np.array(points[0], float)
    vel = np.zeros(2)
    out = [pos.copy()]
    for target in points[1:]:
        target = np.asarray(target, float)
        for _ in range(200):
            d = target - pos
            dist = np.linalg.norm(d)
            if dist < 0.15:
                break
            u = d / dist
            speed = np.linalg.norm(vel)
            v_des = min(s0, np.sqrt(2 * dmax * max(dist - 0.1, 0)))
            v_cur_along = vel @ u
            if v_des > v_cur_along:
                acc = a0 * max(1 - speed / s0, 0.05)
            else:
                acc = -dmax
            new_speed_along = np.clip(v_cur_along + acc * config.DT, 0, s0)
            # lateral velocity is damped by braking capacity
            lat = vel - v_cur_along * u
            lat_mag = np.linalg.norm(lat)
            lat = lat * max(lat_mag - dmax * config.DT, 0) / lat_mag if lat_mag > 1e-9 else lat
            vel = new_speed_along * u + lat
            pos = pos + vel * config.DT
            out.append(pos.copy())
    out = np.array(out)
    return out + rng.normal(0, 0.02, out.shape)


def _speed_acc(xy: np.ndarray):
    v = np.gradient(xy, config.DT, axis=0)
    s = np.linalg.norm(v, axis=1)
    a = np.abs(np.gradient(s, config.DT))
    return s, a


def generate(n_per_pos: int = 24, plays_per_player: int = 60, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    players, results, career, ct_rows, gt_rows, pp_rows = [], [], [], [], [], []
    nfl_id = 60000
    play_counter = 0
    games = pd.DataFrame({
        "game_id": np.arange(2023090700, 2023090700 + 30),
        "season": np.repeat([2023, 2024, 2025], 10),
        "week": np.tile(np.arange(1, 11), 3),
        "game_key": np.arange(30),
        "home_team_abbr": "HOM", "visitor_team_abbr": "VIS", "season_type": "REG",
    })
    event_seq = 0
    for pos, (s0_mu, tau_mu, w_mu, h_mu) in POS_PARAMS.items():
        for _ in range(n_per_pos):
            nfl_id += 1
            draft_year = int(rng.choice([2023, 2024, 2025]))
            s0 = rng.normal(s0_mu, 0.35)
            tau = max(rng.normal(tau_mu, 0.08), 0.75)
            a0 = s0 / tau
            dmax = max(rng.normal(7.0, 0.9), 4.0)
            expression = np.clip(rng.beta(6, 2.5), 0.45, 0.99)
            talent = 0.6 * (s0 - s0_mu) / 0.35 + 0.4 * (expression - 0.7) / 0.12 + rng.normal(0, 0.7)
            pick_score = 0.5 * (s0 - s0_mu) / 0.35 + rng.normal(0, 1.0)  # teams see the engine only
            pick = int(np.clip(130 - 55 * pick_score, 1, 300))
            drafted = pick <= 257
            seasons = 2025 - draft_year + 1
            snaps = max(0, int(seasons * (350 + 260 * talent + rng.normal(0, 120))))
            off = pos in ("WR", "TE", "OL")
            players.append({
                "nfl_id": nfl_id, "display_name": f"Synthetic {pos} {nfl_id}",
                "birth_date": "2001-01-01", "college_name": "Synthetic U", "college_conference": "SYN",
                "draft_year": draft_year,
                "draft_round": (pick - 1) // 32 + 1 if drafted else np.nan,
                "draft_pick_within_round": (pick - 1) % 32 + 1 if drafted else np.nan,
                "draft_overall_pick": pick if drafted else np.nan,
                "nfl_position": rng.choice(NFL_POS[pos]),
            })
            forty_t = _sprint(s0, tau)[1]
            x40, _ = _sprint(s0, tau)
            forty = (np.searchsorted(x40[8:], 40) * config.DT) + rng.normal(0.05, 0.02)
            ten = (np.searchsorted(x40[8:], 10) * config.DT) + rng.normal(0.05, 0.02)
            results.append({
                "nfl_id": nfl_id, "draft_year": draft_year, "combine_position": pos,
                "combine_height": rng.normal(h_mu, 1.5), "combine_weight": rng.normal(w_mu, 10),
                "hand_size": rng.normal(9.6, 0.4), "arm_length": rng.normal(32.5, 1.2),
                "wing_span": rng.normal(78, 2.5), "forty": round(forty, 2), "ten_yd_split": round(ten, 2),
                "vertical": rng.normal(33, 3), "broad_jump": rng.normal(118, 6),
                "three_cone": rng.normal(7.1, 0.2), "short_shuttle": rng.normal(4.4, 0.12),
                "bench_reps": rng.normal(20, 5),
                "ngs_athleticism_score": int(np.clip(rng.normal(72, 10), 40, 99)),
                "ngs_college_production_score": int(np.clip(rng.normal(70, 10), 40, 99)),
                "ngs_final_score": int(np.clip(rng.normal(70, 10), 40, 99)),
            })
            career.append({
                "nfl_id": nfl_id,
                "career_offensive_snaps": snaps if off else 0,
                "career_defensive_snaps": 0 if off else snaps,
                "career_special_teams_snaps": int(rng.integers(0, 300)),
                "career_games_active": int(min(17 * seasons, max(0, 8 * seasons + 4 * talent * seasons))),
                "career_games_started": int(min(17 * seasons, max(0, 4 * seasons + 4 * talent * seasons))),
                "pro_bowl_original_ballot": int(talent > 2.0),
                "ap_all_pro_1st_team": int(talent > 2.6),
                "ap_all_pro_2nd_team": 0,
            })
            # ---- combine tracking: two 40s and two of each COD drill
            t_clock = float(event_seq * 100)
            for attempt in (1, 2):
                event_seq += 1
                x, _ = _sprint(s0 * rng.normal(1, 0.01), tau * rng.normal(1, 0.03), rng=rng)
                xy = np.column_stack([20 + x, 25 + rng.normal(0, 0.03, x.size)])
                ct_rows.append(_ct_frame(nfl_id, draft_year, config.DRILL_FORTY, "forty_yard_dash",
                                         attempt, event_seq, t_clock, xy))
                t_clock += 30
            shuttle = [(0, 0), (0, 5), (0, -5), (0, 0)]
            cone = [(0, 0), (5, 0), (0, 0), (5, 0), (5, 5), (0, 5.2), (5, 5), (5.2, 0), (0, 0)]
            for drill, dtype, pts in ((config.DRILL_SHUTTLE, "short_shuttle", shuttle),
                                      (config.DRILL_THREE_CONE, "three_cone_drill", cone)):
                for attempt in (1, 2):
                    event_seq += 1
                    xy = _waypoint_run(pts, s0 * 0.85, a0, dmax * rng.normal(1, 0.03), rng)
                    xy = np.vstack([np.repeat(xy[:1], 6, axis=0), xy]) + [40, 20]
                    ct_rows.append(_ct_frame(nfl_id, draft_year, drill, dtype, attempt, event_seq, t_clock, xy))
                    t_clock += 30
            # ---- game tracking: sprints at intensity <= expression
            season_games = games[games["season"] >= draft_year]["game_id"].to_numpy()
            for k in range(plays_per_player):
                play_counter += 1
                gid = int(rng.choice(season_games))
                intensity = expression * rng.uniform(0.55, 1.0)
                v0 = rng.uniform(0, 2.5)
                t = np.arange(0, 4.5, config.DT)
                vt = s0 * intensity
                v = vt - (vt - v0) * np.exp(-t / (tau * rng.uniform(1.0, 1.25)))
                v[t > 3.3] = np.maximum(v[t > 3.3][0] - 5 * (t[t > 3.3] - 3.3), 0)
                ang = rng.uniform(0, 2 * np.pi)
                d = np.cumsum(v) * config.DT
                xy = np.column_stack([60 + d * np.cos(ang), 26 + d * np.sin(ang)])
                s, a = _speed_acc(xy)
                gt_rows.append(pd.DataFrame({
                    "game_id": gid, "play_id": play_counter, "nfl_id": nfl_id,
                    "time": _iso(10_000 + play_counter * 10 + t), "x": xy[:, 0], "y": xy[:, 1],
                    "s": s, "a": a, "dis": s * config.DT, "o": 0.0, "dir": np.degrees(ang) % 360,
                    "event": ["ball_snap"] + [np.nan] * (t.size - 1),
                }))
                pp_rows.append(_pp_row(pos, gid, play_counter, nfl_id, talent, rng))
    data = {
        "players": pd.DataFrame(players),
        "combine_results": pd.DataFrame(results),
        "player_career_successes": pd.DataFrame(career),
        "games": games,
        "combine_tracking": pd.concat(ct_rows, ignore_index=True),
        "player_play": pd.DataFrame(pp_rows),
    }
    gt = pd.concat(gt_rows, ignore_index=True)
    season_of = games.set_index("game_id")["season"]
    for season in config.SEASONS:
        data[f"game_tracking_{season}"] = gt[gt["game_id"].map(season_of) == season].reset_index(drop=True)
    return data


def _ct_frame(nfl_id, draft_year, drill, dtype, attempt, event_seq, t_clock, xy):
    s, a = _speed_acc(xy)
    t = t_clock + np.arange(len(xy)) * config.DT
    return pd.DataFrame({
        "nfl_id": nfl_id, "draft_year": draft_year, "drill_name": drill.upper(),
        "drill_type": dtype.upper(), "entity_type": "PLAYER", "attempt": attempt, "event_id": f"SYN_{event_seq:05d}",
        "time": _iso(t), "x": xy[:, 0], "y": xy[:, 1], "s": s, "a": a,
        "dis": s * config.DT, "dir": 90.0,
    })


def _pp_row(pos, gid, play_id, nfl_id, talent, rng):
    row = {"game_id": gid, "play_id": play_id, "nfl_id": nfl_id, "team_abbr": "SYN",
           "route_ran": np.nan, "separation_at_pass_forward": np.nan, "target": False,
           "pressure_allowed": np.nan, "time_to_pressure": np.nan, "player_get_off": np.nan,
           "sack": 0, "tackle": np.nan, "assist": np.nan, "pass_rushers_encountered": np.nan}
    if pos in ("WR", "TE"):
        row["route_ran"] = rng.choice(["go", "hitch", "slant", "out", "post"])
        row["separation_at_pass_forward"] = max(0.3, rng.normal(3.0 + 0.35 * talent, 1.2))
        row["target"] = bool(rng.random() < 0.18 + 0.03 * talent)
    elif pos == "OL":
        row["pressure_allowed"] = bool(rng.random() < max(0.02, 0.07 - 0.015 * talent))
        row["pass_rushers_encountered"] = 1
    elif pos == "DL":
        pressured = rng.random() < max(0.02, 0.10 + 0.025 * talent)
        row["time_to_pressure"] = rng.uniform(1.8, 3.5) if pressured else np.nan
        row["player_get_off"] = max(0.4, rng.normal(0.95 - 0.04 * talent, 0.1))
    else:
        if rng.random() < 0.08 + 0.01 * talent:  # real data: blank unless involved
            solo = rng.random() < 0.6
            row["tackle"], row["assist"] = (1, 0) if solo else (0, 1)
    return row


def write(out_dir: Path, **kw) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in generate(**kw).items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "SYNTHETIC_DATA_README.txt").write_text(
        "These files are SYNTHETIC. They copy the schema of the Big Data Bowl 2027 data but "
        "contain no real measurements.\n"
    )
    return out_dir
