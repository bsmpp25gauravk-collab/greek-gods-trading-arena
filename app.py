from flask import Flask, render_template, request, jsonify
import math, random

app = Flask(__name__)

LIMITS = {"delta": 100, "gamma": 20, "vega": 500, "theta": 1000}

# ── Pure-Python BSM (no scipy needed) ─────────────────────────────────────────
def _norm_cdf(x):
    return math.erfc(-x / math.sqrt(2)) / 2

def _norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)

def bsm(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
        delta = _norm_cdf(d1)
        rho   = K * T * math.exp(-r * T) * _norm_cdf(d2) / 100
    else:
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
        delta = _norm_cdf(d1) - 1
        rho   = -K * T * math.exp(-r * T) * _norm_cdf(-d2) / 100
    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega  = S * math.sqrt(T) * _norm_pdf(d1) / 100
    theta = (-(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * _norm_cdf(d2 if option_type == "call" else -d2)) / 365
    return {
        "price": round(price, 2), "d1": round(d1, 4), "d2": round(d2, 4),
        "delta": round(delta, 4), "gamma": round(gamma, 6),
        "theta": round(theta, 2), "vega": round(vega, 4), "rho": round(rho, 4),
    }

def risk_pct(value, limit):
    return round(min(abs(value) / limit * 100, 120), 1)

def risk_zone(pct):
    if pct >= 100: return "breach"
    if pct >= 70:  return "caution"
    return "safe"

def get_recommendations(port):
    recs = []
    d, dv, dt, dg = port["delta"], port["vega"], port["theta"], port["gamma"]
    if abs(d) > LIMITS["delta"]:
        hedge = "sell futures / add short calls" if d > 0 else "buy futures / add long calls"
        recs.append(f"🔴 Delta BREACH ({d:+.1f}) — {'too long' if d>0 else 'too short'}. Action: {hedge}.")
    elif abs(d) > LIMITS["delta"] * 0.7:
        recs.append(f"🟡 Delta nearing limit ({d:+.1f} / ±{LIMITS['delta']}). Monitor closely.")
    if abs(dg * 1000) > LIMITS["gamma"]:
        recs.append(f"🔴 Gamma BREACH — close short ATM positions immediately.")
    elif abs(dg * 1000) > LIMITS["gamma"] * 0.7:
        recs.append(f"🟡 Gamma caution — reduce short near-ATM exposure.")
    if abs(dv) > LIMITS["vega"]:
        recs.append(f"🔴 Vega BREACH (₹{dv:+.0f}) — {'long' if dv>0 else 'short'} vol too high.")
    elif abs(dv) > LIMITS["vega"] * 0.7:
        recs.append(f"🟡 Vega approaching limit. Watch for upcoming events.")
    if abs(dt) > LIMITS["theta"]:
        recs.append(f"🔴 Theta BREACH (₹{dt:+.0f}/day) — rebalance premium exposure.")
    if not recs:
        recs.append("✅ All Greeks within limits. Portfolio is well-managed.")
    return recs

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/study")
def study():
    return render_template("study.html")

# ── CALCULATOR ────────────────────────────────────────────────────────────────
@app.route("/calculator", methods=["GET", "POST"])
def calculator():
    result_call = result_put = None
    # Always store display values (percentages as user sees them)
    form = {"spot": 0, "strike": 0, "days": 0, "rate": 0, "iv": 0,
            "lot_size": 65, "contracts": 1}
    if request.method == "POST":
        try:
            spot      = float(request.form["spot"])
            strike    = float(request.form["strike"])
            days      = float(request.form["days"])
            rate_pct  = float(request.form["rate"])   # user enters 5.28
            iv_pct    = float(request.form["iv"])      # user enters 22.49
            lot_size  = int(request.form.get("lot_size", 65))
            contracts = int(request.form.get("contracts", 1))
            # Store display values back — this prevents the "second click" bug
            form = {"spot": spot, "strike": strike, "days": days,
                    "rate": rate_pct, "iv": iv_pct,
                    "lot_size": lot_size, "contracts": contracts}
            rate = rate_pct / 100
            iv   = iv_pct   / 100
            T    = days / 365
            ps   = lot_size * contracts
            gc = bsm(spot, strike, T, rate, iv, "call")
            gp = bsm(spot, strike, T, rate, iv, "put")
            if gc and gp:
                for g, sign in [(gc, 1), (gp, -1)]:
                    g["port_delta"]   = round(g["delta"] * ps, 2)
                    g["dollar_delta"] = round(g["delta"] * spot * ps, 0)
                    g["daily_theta"]  = round(g["theta"] * ps, 2)
                    g["port_vega"]    = round(g["vega"] * ps, 2)
                result_call, result_put = gc, gp
        except Exception as e:
            result_call = {"error": str(e)}
    return render_template("calculator.html",
                           result_call=result_call, result_put=result_put, form=form)

# ── SCENARIOS ─────────────────────────────────────────────────────────────────
@app.route("/scenarios", methods=["GET", "POST"])
def scenarios():
    results, form = [], {}
    SCENARIOS = [
        {"label": "Base",         "price_shock":  0, "iv_shock":  0},
        {"label": "Mild Drop",    "price_shock": -2, "iv_shock":  2},
        {"label": "Crash −5%",   "price_shock": -5, "iv_shock": 10},
        {"label": "Rally +5%",   "price_shock":  5, "iv_shock": -3},
        {"label": "Vol Spike",    "price_shock":  0, "iv_shock": 15},
        {"label": "Black Swan",  "price_shock":-10, "iv_shock": 20},
    ]
    if request.method == "POST":
        try:
            spot      = float(request.form["spot"])
            strike    = float(request.form["strike"])
            days      = float(request.form["days"])
            rate_pct  = float(request.form["rate"])
            iv_pct    = float(request.form["iv"])
            option_type = request.form.get("option_type", "call")
            lot_size  = int(request.form.get("lot_size", 65))
            contracts = int(request.form.get("contracts", 1))
            form = {"spot": spot, "strike": strike, "days": days,
                    "rate": rate_pct, "iv": iv_pct, "option_type": option_type,
                    "lot_size": lot_size, "contracts": contracts}
            rate = rate_pct / 100
            iv   = iv_pct   / 100
            ps   = lot_size * contracts
            base_g = bsm(spot, strike, max(days/365, 0.001), rate, iv, option_type)
            for sc in SCENARIOS:
                new_spot = spot * (1 + sc["price_shock"] / 100)
                new_iv   = max(iv + sc["iv_shock"] / 100, 0.01)
                new_T    = max(days / 365, 0.001)
                g = bsm(new_spot, strike, new_T, rate, new_iv, option_type)
                if g and base_g:
                    results.append({
                        "label":       sc["label"],
                        "price_shock": sc["price_shock"],
                        "iv_shock":    sc["iv_shock"],
                        "new_spot":    round(new_spot, 0),
                        "new_iv":      round(new_iv * 100, 2),
                        "price":       g["price"],
                        "delta":       g["delta"],
                        "gamma":       g["gamma"],
                        "theta":       g["theta"],
                        "vega":        g["vega"],
                        "port_delta":  round(g["delta"] * ps, 2),
                        "pnl":         round((g["price"] - base_g["price"]) * ps, 2),
                    })
        except:
            results = []
    if not form:
        form = {"spot": 0, "strike": 0, "days": 0, "rate": 0, "iv": 0,
                "option_type": "call", "lot_size": 65, "contracts": 1}
    return render_template("scenarios.html", results=results, form=form, scenarios=SCENARIOS)

# ── PORTFOLIO ─────────────────────────────────────────────────────────────────
@app.route("/portfolio", methods=["GET", "POST"])
def portfolio():
    positions, portfolio_greeks, gauges, recs = [], {}, {}, []
    if request.method == "POST":
        spot = float(request.form.get("spot", 0) or 0)
        rate = float(request.form.get("rate", 5.28) or 5.28) / 100
        port_d = port_g = port_t = port_v = 0.0
        i = 0
        while True:
            name = request.form.get(f"name_{i}", "").strip()
            if not name:
                break
            try:
                strike   = float(request.form.get(f"strike_{i}", 0))
                iv_pct   = float(request.form.get(f"iv_{i}", 20))
                days     = float(request.form.get(f"days_{i}", 5))
                lots     = int(request.form.get(f"lots_{i}", 1))
                lot_size = int(request.form.get(f"lot_size_{i}", 65))
                opt_type = request.form.get(f"type_{i}", "call")
                g = bsm(spot, strike, max(days/365, 0.001), rate, iv_pct/100, opt_type)
                if g:
                    ps = lot_size * lots
                    pd = round(g["delta"] * ps, 2)
                    pg = round(g["gamma"] * ps, 6)
                    pt = round(g["theta"] * ps, 2)
                    pv = round(g["vega"]  * ps, 2)
                    port_d += pd; port_g += pg; port_t += pt; port_v += pv
                    positions.append({**p, **g, "pos_size": ps,
                        "p_delta": pd, "p_gamma": pg, "p_theta": pt, "p_vega": pv}
                        if False else {
                        "name": name, "type": opt_type, "strike": strike,
                        "iv": iv_pct, "days": days, "lots": lots, "lot_size": lot_size,
                        **g, "pos_size": ps,
                        "p_delta": pd, "p_gamma": pg, "p_theta": pt, "p_vega": pv
                    })
            except:
                pass
            i += 1
        portfolio_greeks = {
            "delta": round(port_d, 2), "gamma": round(port_g, 6),
            "theta": round(port_t, 2), "vega":  round(port_v, 2),
            "dollar_delta": round(port_d * spot, 0) if spot else 0,
        }
        gd = risk_pct(port_d, LIMITS["delta"])
        gg = risk_pct(port_g * 1000, LIMITS["gamma"])
        gv = risk_pct(port_v, LIMITS["vega"])
        gt = risk_pct(port_t, LIMITS["theta"])
        gauges = {
            "delta": {"pct": gd, "zone": risk_zone(gd), "limit": LIMITS["delta"], "value": round(port_d,2)},
            "gamma": {"pct": gg, "zone": risk_zone(gg), "limit": LIMITS["gamma"], "value": round(port_g*1000,2)},
            "vega":  {"pct": gv, "zone": risk_zone(gv), "limit": LIMITS["vega"],  "value": round(port_v,2)},
            "theta": {"pct": gt, "zone": risk_zone(gt), "limit": LIMITS["theta"], "value": round(port_t,2)},
        }
        recs = get_recommendations(portfolio_greeks)
    return render_template("portfolio.html",
        positions=positions, portfolio=portfolio_greeks,
        gauges=gauges, recs=recs, limits=LIMITS)

# ── ARENA ─────────────────────────────────────────────────────────────────────
@app.route("/arena")
def arena():
    return render_template("arena.html", levels=LEVEL_CONFIGS)

LEVEL_CONFIGS = {
    1: {
        "name": "Delta Rookie", "greek_focus": "Δ Delta",
        "badge": "Delta Initiate", "xp_reward": 100,
        "mission": "Keep portfolio delta between −20 and +20 for 8 moves.",
        "tip": "Buy futures to ADD delta. Sell futures to REDUCE delta.",
        "max_moves": 8, "win_score": 100,
        "spot_vol": 0.03, "iv_drift": 0.01,
        "delta_safe": 20, "delta_caution": 50,
        "pts_safe": 15, "pts_caution": 5, "pts_breach": -15,
        "start": {"spot": 23750, "strike": 23750, "iv": 20, "days": 5,
                  "lot_size": 65, "lots": 1, "type": "call"},
        "color": "#4cba76",
    },
    2: {
        "name": "Delta Apprentice", "greek_focus": "Δ Delta",
        "badge": "Delta Master", "xp_reward": 200,
        "mission": "Manage 2 positions. Keep net delta between −50 and +50 for 10 moves.",
        "tip": "You have a long call AND short put. Net delta can swing fast — stay alert.",
        "max_moves": 10, "win_score": 120,
        "spot_vol": 0.04, "iv_drift": 0.015,
        "delta_safe": 50, "delta_caution": 80,
        "pts_safe": 12, "pts_caution": 4, "pts_breach": -18,
        "start": {"spot": 23750, "strike_call": 24000, "strike_put": 23500,
                  "iv": 22, "days": 7, "lot_size": 65, "lots_call": 2, "lots_put": -2},
        "color": "#3ec9c1",
    },
    3: {
        "name": "Gamma Guardian", "greek_focus": "Γ Gamma",
        "badge": "Gamma Guardian", "xp_reward": 300,
        "mission": "Near-expiry ATM options. Keep delta in ±30 AND don't let gamma spike breach.",
        "tip": "With 2 DTE, gamma is EXTREME. Even small spot moves cause huge delta swings. Rebalance faster.",
        "max_moves": 8, "win_score": 100,
        "spot_vol": 0.02, "iv_drift": 0.005,
        "delta_safe": 30, "delta_caution": 55,
        "pts_safe": 15, "pts_caution": 5, "pts_breach": -20,
        "start": {"spot": 23750, "strike": 23750, "iv": 20, "days": 2,
                  "lot_size": 65, "lots": 1, "type": "call"},
        "color": "#d4a017",
    },
    4: {
        "name": "Theta Harvester", "greek_focus": "Θ Theta",
        "badge": "Time Collector", "xp_reward": 400,
        "mission": "You sold a strangle. Collect theta daily. Keep delta in ±60. Survive 10 moves.",
        "tip": "Short strangle = positive theta income every move. But a big spike will hurt fast.",
        "max_moves": 10, "win_score": 80,
        "spot_vol": 0.02, "iv_drift": 0.005,
        "delta_safe": 60, "delta_caution": 85,
        "pts_safe": 10, "pts_caution": 3, "pts_breach": -12,
        "start": {"spot": 23750, "strike_call": 24200, "strike_put": 23300,
                  "iv": 18, "days": 7, "lot_size": 65, "lots_call": -3, "lots_put": -3},
        "color": "#e08c3c",
    },
    5: {
        "name": "Vega Voyager", "greek_focus": "ν Vega",
        "badge": "Volatility Navigator", "xp_reward": 500,
        "mission": "IV will crush 2% per move. Manage your long vega before it destroys your P&L.",
        "tip": "You're long vega. IV is high (35%) and will collapse. Sell calls to reduce vega exposure.",
        "max_moves": 10, "win_score": 90,
        "spot_vol": 0.025, "iv_drift": -0.02,
        "delta_safe": 40, "delta_caution": 70,
        "pts_safe": 12, "pts_caution": 4, "pts_breach": -15,
        "start": {"spot": 23750, "strike": 23750, "iv": 35, "days": 10,
                  "lot_size": 65, "lots": 2, "type": "call"},
        "color": "#7c5cbf",
    },
    6: {
        "name": "Greek Grandmaster", "greek_focus": "All Greeks",
        "badge": "Greek God", "xp_reward": 1000,
        "mission": "CRISIS MODE. Spot gapped −5%, IV spiked +20%, 3 DTE. All Greeks are red. Survive 12 moves.",
        "tip": "Reduce gross exposure FIRST before trying to hedge individual Greeks. Survive above all else.",
        "max_moves": 12, "win_score": 130,
        "spot_vol": 0.045, "iv_drift": 0.025,
        "delta_safe": 35, "delta_caution": 65,
        "pts_safe": 15, "pts_caution": 5, "pts_breach": -20,
        "start": {"spot": 22563, "strike_call": 23750, "strike_put": 23000,
                  "iv": 40, "days": 3, "lot_size": 65,
                  "lots_call": 2, "lots_put": -3},
        "color": "#e05252",
    },
}

def calc_multi_position(spot, rate, positions_cfg):
    """Calculate aggregated Greeks for multi-position start state."""
    port_delta = 0.0
    pos_list = []
    for p in positions_cfg:
        g = bsm(spot, p["strike"], max(p["days"]/365, 0.001), rate, p["iv"]/100, p["type"])
        if g:
            ps = p["lot_size"] * p["lots"]
            port_delta += g["delta"] * ps
            pos_list.append({**p, **g, "ps": ps})
    return round(port_delta, 2), pos_list

@app.route("/api/arena/start", methods=["POST"])
def arena_start():
    data  = request.json or {}
    level = int(data.get("level", 1))
    cfg   = LEVEL_CONFIGS.get(level, LEVEL_CONFIGS[1])
    s     = cfg["start"]
    rate  = 0.0528

    # Build starting positions
    if "strike_call" in s and "strike_put" in s:
        # Two-position start
        positions = [
            {"strike": s["strike_call"], "iv": s["iv"], "days": s["days"],
             "lot_size": s["lot_size"], "lots": s.get("lots_call", 1), "type": "call"},
            {"strike": s["strike_put"],  "iv": s["iv"], "days": s["days"],
             "lot_size": s["lot_size"], "lots": s.get("lots_put", -1),  "type": "put"},
        ]
    else:
        positions = [
            {"strike": s["strike"], "iv": s["iv"], "days": s["days"],
             "lot_size": s["lot_size"], "lots": s.get("lots", 1), "type": s.get("type","call")},
        ]

    spot = s["spot"]
    port_delta, pos_list = calc_multi_position(spot, rate, positions)

    # Compute primary option Greeks for display
    primary = pos_list[0] if pos_list else {}

    return jsonify({
        "level": level, "spot": spot, "rate": rate * 100, "iv": s["iv"],
        "days": s["days"], "futures_held": 0,
        "positions": positions,
        "port_delta": port_delta,
        "delta": primary.get("delta", 0.5),
        "gamma": primary.get("gamma", 0.001),
        "theta": primary.get("theta", -10),
        "vega":  primary.get("vega", 10),
        "option_price": primary.get("price", 200),
        "score": 100, "xp": 0, "move": 0,
        "max_moves": cfg["max_moves"], "win_score": cfg["win_score"],
        "log": [
            f"🎮 Level {level}: {cfg['name']} — {cfg['mission']}",
            f"💡 {cfg['tip']}",
            f"📊 Starting portfolio delta: {port_delta:+.1f}",
        ],
        "game_over": False, "won": False,
        "lot_size": s["lot_size"],
        "cfg": {k: cfg[k] for k in ["delta_safe","delta_caution","pts_safe","pts_caution","pts_breach","greek_focus","badge","xp_reward","color","mission","name","win_score","max_moves"]},
    })

@app.route("/api/arena/move", methods=["POST"])
def arena_move():
    data         = request.json
    level        = int(data.get("level", 1))
    cfg          = LEVEL_CONFIGS.get(level, LEVEL_CONFIGS[1])
    spot         = float(data["spot"])
    rate         = float(data["rate"]) / 100
    iv           = float(data["iv"]) / 100
    days         = max(float(data["days"]) - 1, 0.5)
    futures_held = int(data["futures_held"])
    action       = data.get("action", "hold")
    score        = int(data["score"])
    xp           = int(data["xp"])
    move         = int(data["move"]) + 1
    max_moves    = int(data["max_moves"])
    log          = list(data.get("log", []))
    lot_size     = int(data.get("lot_size", 65))
    positions    = data.get("positions", [])

    # Market moves
    # For level 4 (theta), keep spot mostly sideways but occasional spike
    if level == 4 and random.random() > 0.25:
        move_pct = random.uniform(-0.015, 0.015)
    else:
        move_pct = random.uniform(-cfg["spot_vol"], cfg["spot_vol"])

    # Level 5: IV drift downward (IV crush)
    iv_change = cfg["iv_drift"] + random.uniform(-0.005, 0.005)
    new_iv    = max(iv + iv_change, 0.05)
    new_spot  = round(spot * (1 + move_pct), 2)

    # Action
    if action == "buy_future":
        futures_held += lot_size
        action_desc = f"📈 Bought futures (+{lot_size} Δ)"
        score -= 2
    elif action == "sell_future":
        futures_held -= lot_size
        action_desc = f"📉 Sold futures (−{lot_size} Δ)"
        score -= 2
    else:
        action_desc = "⏸ Held"

    # Recalculate portfolio delta at new spot
    port_delta = futures_held
    primary = None
    for p in positions:
        g = bsm(new_spot, p["strike"], max(days/365, 0.001), rate, new_iv, p["type"])
        if g:
            ps = p["lot_size"] * p["lots"]
            port_delta += g["delta"] * ps
            if primary is None:
                primary = g

    port_delta = round(port_delta, 2)
    g_display  = primary or {"delta":0.5,"gamma":0.001,"theta":-10,"vega":10,"price":200}

    # Level 4 (Theta): add theta income to score
    if level == 4 and primary:
        theta_income = abs(sum(
            bsm(new_spot, p["strike"], max(days/365,0.001), rate, new_iv, p["type"])["theta"]
            * p["lot_size"] * p["lots"]
            for p in positions
            if bsm(new_spot, p["strike"], max(days/365,0.001), rate, new_iv, p["type"])
        ))
        score += int(theta_income * 0.5)  # bonus from theta

    # Scoring
    direction = "▲" if move_pct > 0 else "▼"
    chg = round(abs(new_spot - spot), 0)
    entry = f"Move {move}: Nifty {direction}₹{chg:.0f} → ₹{new_spot:,.0f}. {action_desc}."
    if abs(port_delta) <= cfg["delta_safe"]:
        score += cfg["pts_safe"]; xp += 20
        entry += f" ✅ Δ={port_delta:+.1f} — safe! +{cfg['pts_safe']}pts"
    elif abs(port_delta) <= cfg["delta_caution"]:
        score += cfg["pts_caution"]; xp += 8
        entry += f" 🟡 Δ={port_delta:+.1f} — caution. +{cfg['pts_caution']}pts"
    else:
        score -= abs(cfg["pts_breach"]); 
        entry += f" 🔴 Δ={port_delta:+.1f} — BREACH! {cfg['pts_breach']}pts"

    score = max(0, score)
    log.append(entry)

    game_over = move >= max_moves
    won       = game_over and score >= cfg["win_score"]
    if game_over:
        if won:
            xp += cfg["xp_reward"]
            log.append(f"🏆 LEVEL {level} COMPLETE! Score: {score}. {cfg['badge']} badge earned! +{cfg['xp_reward']} XP")
        else:
            log.append(f"⚠️ Level ended. Score: {score}/{cfg['max_moves']*cfg['pts_safe']+100}. Need {cfg['win_score']}+ to pass.")

    return jsonify({
        "level": level, "spot": new_spot, "rate": rate*100,
        "iv": round(new_iv*100, 2), "days": days,
        "positions": positions, "futures_held": futures_held,
        "port_delta": port_delta, "lot_size": lot_size,
        "delta": g_display.get("delta",0.5), "gamma": g_display.get("gamma",0.001),
        "theta": g_display.get("theta",-10), "vega": g_display.get("vega",10),
        "option_price": g_display.get("price",200),
        "score": score, "xp": xp, "move": move, "max_moves": max_moves,
        "win_score": cfg["win_score"],
        "log": log, "game_over": game_over, "won": won,
        "cfg": {k: cfg[k] for k in ["delta_safe","delta_caution","pts_safe","pts_caution","pts_breach","greek_focus","badge","xp_reward","color","mission","name","win_score","max_moves"]},
    })

if __name__ == "__main__":
    app.run(debug=True)
