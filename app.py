from flask import Flask, render_template, request, jsonify
import math, random
from scipy.stats import norm

app = Flask(__name__)

LIMITS = {"delta": 100, "gamma": 20, "vega": 500, "theta": 1000}

def risk_pct(value, limit):
    return round(min(abs(value) / limit * 100, 120), 1)

def risk_zone(pct):
    if pct >= 100: return "breach"
    if pct >= 70:  return "caution"
    return "safe"

def recommendation(port):
    recs = []
    d, dv, dt, dg = port["delta"], port["vega"], port["theta"], port["gamma"]
    if abs(d) > LIMITS["delta"]:
        hedge = "sell futures / add short calls" if d > 0 else "buy futures / add long calls"
        recs.append(f"🔴 Delta BREACH ({d:+.1f}) — book is {'too long' if d>0 else 'too short'}. Action: {hedge}.")
    elif abs(d) > LIMITS["delta"] * 0.7:
        recs.append(f"🟡 Delta approaching limit ({d:+.1f}/{LIMITS['delta']}). Monitor closely.")
    if abs(dg * 1000) > LIMITS["gamma"]:
        recs.append(f"🔴 Gamma BREACH — negative gamma too high near expiry. Close short ATM positions immediately.")
    elif abs(dg * 1000) > LIMITS["gamma"] * 0.7:
        recs.append(f"🟡 Gamma caution — reduce short near-ATM exposure before next expiry.")
    if abs(dv) > LIMITS["vega"]:
        side = "long vol" if dv > 0 else "short vol"
        recs.append(f"🔴 Vega BREACH (₹{dv:+.0f}) — {side} exposure too high. Reduce before any event.")
    elif abs(dv) > LIMITS["vega"] * 0.7:
        recs.append(f"🟡 Vega approaching limit. Watch for upcoming RBI / earnings events.")
    if abs(dt) > LIMITS["theta"]:
        side = "paying too much theta" if dt < 0 else "collecting theta aggressively"
        recs.append(f"🔴 Theta BREACH (₹{dt:+.0f}/day) — {side}. Rebalance premium exposure.")
    if not recs:
        recs.append("✅ All Greeks within limits. Portfolio is well-managed.")
    return recs

def bsm_calculate(S, K, T, r, sigma, option_type="call"):
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho   = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho   = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega  = S * math.sqrt(T) * norm.pdf(d1) / 100
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)) / 365
    return {
        "price": round(price, 2), "d1": round(d1, 4), "d2": round(d2, 4),
        "delta": round(delta, 4), "gamma": round(gamma, 6),
        "theta": round(theta, 2), "vega": round(vega, 4), "rho": round(rho, 4),
    }

def run_scenario(S, K, T_days, r, sigma, option_type, lot_size, contracts, price_shock_pct, iv_shock_pct):
    new_S   = S * (1 + price_shock_pct / 100)
    new_sig = max(sigma + iv_shock_pct / 100, 0.01)
    new_T   = max(T_days / 365, 0.001)
    base    = bsm_calculate(S, K, T_days / 365, r, sigma, option_type)
    result  = bsm_calculate(new_S, K, new_T, r, new_sig, option_type)
    if not result or not base:
        return None
    pos = lot_size * contracts
    return {
        "new_spot": round(new_S, 2), "new_iv": round(new_sig * 100, 2),
        "new_price": result["price"], "base_price": base["price"],
        "delta": result["delta"], "gamma": result["gamma"],
        "vega": result["vega"], "theta": result["theta"],
        "port_delta": round(result["delta"] * pos, 2),
        "pnl": round((result["price"] - base["price"]) * pos, 2),
    }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/study")
def study():
    return render_template("study.html")

@app.route("/calculator", methods=["GET", "POST"])
def calculator():
    result, form = None, {}
    if request.method == "POST":
        try:
            form = {
                "spot": float(request.form["spot"]), "strike": float(request.form["strike"]),
                "days": float(request.form["days"]), "rate": float(request.form["rate"]) / 100,
                "iv": float(request.form["iv"]) / 100, "option_type": request.form["option_type"],
                "lot_size": int(request.form.get("lot_size", 65)),
                "contracts": int(request.form.get("contracts", 1)),
            }
            g = bsm_calculate(form["spot"], form["strike"], form["days"] / 365,
                               form["rate"], form["iv"], form["option_type"])
            if g:
                ps = form["lot_size"] * form["contracts"]
                g["port_delta"]   = round(g["delta"] * ps, 4)
                g["port_vega"]    = round(g["vega"]  * ps * 100, 4)
                g["dollar_delta"] = round(g["delta"] * form["spot"] * ps, 2)
                result = g
        except Exception as e:
            result = {"error": str(e)}
    if not form:
        form = {"spot": 23741, "strike": 23750, "days": 5, "rate": 5.28,
                "iv": 22.49, "option_type": "call", "lot_size": 65, "contracts": 1}
    return render_template("calculator.html", result=result, form=form)

@app.route("/scenarios", methods=["GET", "POST"])
def scenarios():
    results, form = [], {}
    SCENARIOS = [
        {"label": "Base (No Change)",      "price_shock":  0, "iv_shock":  0},
        {"label": "Mild Drop",             "price_shock": -2, "iv_shock":  2},
        {"label": "Crash (-5%, +10% vol)", "price_shock": -5, "iv_shock": 10},
        {"label": "Big Rally (+5%)",       "price_shock":  5, "iv_shock": -3},
        {"label": "Vol Spike Only",        "price_shock":  0, "iv_shock": 15},
        {"label": "Black Swan (-10%)",     "price_shock":-10, "iv_shock": 20},
    ]
    if request.method == "POST":
        try:
            form = {
                "spot": float(request.form["spot"]), "strike": float(request.form["strike"]),
                "days": float(request.form["days"]), "rate": float(request.form["rate"]) / 100,
                "iv": float(request.form["iv"]) / 100, "option_type": request.form["option_type"],
                "lot_size": int(request.form.get("lot_size", 65)),
                "contracts": int(request.form.get("contracts", 1)),
            }
            for sc in SCENARIOS:
                r = run_scenario(form["spot"], form["strike"], form["days"],
                                 form["rate"], form["iv"], form["option_type"],
                                 form["lot_size"], form["contracts"],
                                 sc["price_shock"], sc["iv_shock"])
                if r:
                    r["label"] = sc["label"]; r["price_shock"] = sc["price_shock"]; r["iv_shock"] = sc["iv_shock"]
                    results.append(r)
        except:
            results = []
    if not form:
        form = {"spot": 23741, "strike": 23750, "days": 5, "rate": 5.28,
                "iv": 22.49, "option_type": "call", "lot_size": 65, "contracts": 1}
    return render_template("scenarios.html", results=results, form=form)

@app.route("/portfolio", methods=["GET", "POST"])
def portfolio():
    positions, portfolio_greeks, gauges, recs = [], {}, {}, []
    DEFAULT_SPOT = 23741
    PRELOADED = [
        {"name": "Nifty 24500 CE", "type": "call", "strike": 24500, "iv": 20, "days": 4, "lots":  10, "lot_size": 65},
        {"name": "Nifty 24000 PE", "type": "put",  "strike": 24000, "iv": 22, "days": 4, "lots":  -5, "lot_size": 65},
        {"name": "Nifty 24200 CE", "type": "call", "strike": 24200, "iv": 21, "days": 4, "lots":  -8, "lot_size": 65},
        {"name": "Nifty 23800 PE", "type": "put",  "strike": 23800, "iv": 23, "days": 4, "lots":   6, "lot_size": 65},
    ]
    if request.method == "POST":
        spot = float(request.form.get("spot", DEFAULT_SPOT))
        rate = float(request.form.get("rate", 5.28)) / 100
        raw_positions = []
        for i in range(6):
            name = request.form.get(f"name_{i}", "").strip()
            if not name:
                continue
            try:
                raw_positions.append({
                    "name": name, "type": request.form.get(f"type_{i}", "call"),
                    "strike": float(request.form.get(f"strike_{i}", 0)),
                    "iv": float(request.form.get(f"iv_{i}", 20)),
                    "days": float(request.form.get(f"days_{i}", 5)),
                    "lots": int(request.form.get(f"lots_{i}", 1)),
                    "lot_size": int(request.form.get(f"lot_size_{i}", 65)),
                })
            except:
                continue
        port_d = port_g = port_t = port_v = 0.0
        for p in raw_positions:
            g = bsm_calculate(spot, p["strike"], max(p["days"]/365, 0.001),
                              rate, p["iv"]/100, p["type"])
            if not g:
                continue
            pos_size = p["lot_size"] * p["lots"]
            p_delta  = round(g["delta"] * pos_size, 2)
            p_gamma  = round(g["gamma"] * pos_size, 6)
            p_theta  = round(g["theta"] * pos_size, 2)
            p_vega   = round(g["vega"]  * pos_size, 2)
            port_d  += p_delta; port_g += p_gamma
            port_t  += p_theta; port_v += p_vega
            positions.append({**p, **g, "pos_size": pos_size,
                "p_delta": p_delta, "p_gamma": p_gamma,
                "p_theta": p_theta, "p_vega":  p_vega})
        portfolio_greeks = {
            "delta": round(port_d, 2), "gamma": round(port_g, 6),
            "theta": round(port_t, 2), "vega":  round(port_v, 2),
            "dollar_delta": round(port_d * spot, 0),
        }
        g_delta = risk_pct(port_d, LIMITS["delta"])
        g_gamma = risk_pct(port_g * 1000, LIMITS["gamma"])
        g_vega  = risk_pct(port_v, LIMITS["vega"])
        g_theta = risk_pct(port_t, LIMITS["theta"])
        gauges = {
            "delta": {"pct": g_delta, "zone": risk_zone(g_delta), "limit": LIMITS["delta"], "value": round(port_d, 2)},
            "gamma": {"pct": g_gamma, "zone": risk_zone(g_gamma), "limit": LIMITS["gamma"], "value": round(port_g*1000, 2)},
            "vega":  {"pct": g_vega,  "zone": risk_zone(g_vega),  "limit": LIMITS["vega"],  "value": round(port_v, 2)},
            "theta": {"pct": g_theta, "zone": risk_zone(g_theta), "limit": LIMITS["theta"], "value": round(port_t, 2)},
        }
        recs = recommendation(portfolio_greeks)
    return render_template("portfolio.html",
        positions=positions, portfolio=portfolio_greeks,
        gauges=gauges, recs=recs, preloaded=PRELOADED,
        limits=LIMITS, spot_default=DEFAULT_SPOT, rate_default=5.28)

@app.route("/arena")
def arena():
    return render_template("arena.html")

@app.route("/api/arena/start", methods=["POST"])
def arena_start():
    spot = 23750.0; strike = 23750.0; iv = 0.20; days = 5; rate = 0.0528
    g = bsm_calculate(spot, strike, days/365, rate, iv, "call")
    lot_size = 65
    return jsonify({
        "spot": spot, "strike": strike, "iv": iv*100, "days": days, "rate": rate*100,
        "option_price": g["price"], "delta": g["delta"], "gamma": g["gamma"],
        "theta": g["theta"], "vega": g["vega"],
        "port_delta": round(g["delta"] * lot_size, 2),
        "lot_size": lot_size, "futures_held": 0,
        "score": 100, "xp": 0, "move": 0, "max_moves": 8,
        "log": [
            "🎮 Game started! Nifty at ₹23,750. You hold 1 lot (65 units) of the ATM call.",
            "📋 Mission: Keep portfolio delta between −20 and +20 for all 8 moves.",
            "💡 Tip: BUY futures to add delta. SELL futures to reduce delta. HOLD to do nothing."
        ],
        "game_over": False, "won": False,
    })

@app.route("/api/arena/move", methods=["POST"])
def arena_move():
    data = request.json
    spot         = float(data["spot"])
    strike       = float(data["strike"])
    iv           = float(data["iv"]) / 100
    days         = max(float(data["days"]) - 1, 0.5)
    rate         = float(data["rate"]) / 100
    futures_held = int(data["futures_held"])
    action       = data.get("action", "hold")
    score        = int(data["score"])
    xp           = int(data["xp"])
    move         = int(data["move"]) + 1
    max_moves    = int(data["max_moves"])
    log          = list(data.get("log", []))
    lot_size     = int(data.get("lot_size", 65))

    move_pct  = random.uniform(-0.03, 0.03)
    new_spot  = round(spot * (1 + move_pct), 2)
    new_iv    = max(iv + random.uniform(-0.01, 0.02), 0.10)

    action_desc = ""
    if action == "buy_future":
        futures_held += lot_size
        action_desc   = f"📈 Bought 1 futures lot (+{lot_size} delta)."
        score -= 2
    elif action == "sell_future":
        futures_held -= lot_size
        action_desc   = f"📉 Sold 1 futures lot (−{lot_size} delta)."
        score -= 2
    else:
        action_desc = "⏸ Held — no hedge."

    g = bsm_calculate(new_spot, strike, days/365, rate, new_iv, "call")
    if not g:
        g = {"delta": 0.5, "gamma": 0.0006, "theta": -10, "vega": 10, "price": 200}

    net_delta = round(g["delta"] * lot_size + futures_held, 2)
    direction = "▲" if move_pct > 0 else "▼"
    chg = round(abs(new_spot - spot), 0)

    entry = f"Move {move}: Nifty {direction} ₹{chg:.0f} → ₹{new_spot:,.0f}. {action_desc}"
    if abs(net_delta) <= 20:
        score += 15; xp += 20
        entry += f" ✅ Δ={net_delta:+.1f} — in range! +15 pts"
    elif abs(net_delta) <= 50:
        score += 5; xp += 8
        entry += f" 🟡 Δ={net_delta:+.1f} — caution. +5 pts"
    else:
        score -= 15
        entry += f" 🔴 Δ={net_delta:+.1f} — BREACH! −15 pts"

    score = max(0, score); log.append(entry)
    game_over = move >= max_moves
    won = game_over and score >= 100
    if game_over:
        if won:
            xp += 50
            log.append(f"🏆 LEVEL COMPLETE! Score: {score}. Delta Initiate badge earned! +50 bonus XP")
        else:
            log.append(f"⚠️ Level ended. Final score: {score}/190. Try again to earn the badge!")

    return jsonify({
        "spot": new_spot, "strike": strike, "iv": round(new_iv*100,2),
        "days": days, "rate": rate*100,
        "option_price": g["price"], "delta": g["delta"], "gamma": g["gamma"],
        "theta": g["theta"], "vega": g["vega"],
        "port_delta": net_delta, "futures_held": futures_held,
        "lot_size": lot_size, "score": score, "xp": xp,
        "move": move, "max_moves": max_moves,
        "log": log, "game_over": game_over, "won": won,
    })

if __name__ == "__main__":
    app.run(debug=True)
