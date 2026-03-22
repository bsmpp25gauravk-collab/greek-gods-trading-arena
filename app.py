from flask import Flask, render_template, request, jsonify
import math

# ─────────────────────────────────────────────
#  NORMAL DISTRIBUTION — Pure Python (no scipy)
# ─────────────────────────────────────────────

def norm_cdf(x):
    """Cumulative normal distribution — replaces scipy.stats.norm.cdf"""
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

def norm_pdf(x):
    """Normal probability density — replaces scipy.stats.norm.pdf"""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

class norm:
    """Drop-in replacement for scipy.stats.norm"""
    @staticmethod
    def cdf(x): return norm_cdf(x)
    @staticmethod
    def pdf(x): return norm_pdf(x)

app = Flask(__name__)

# ─────────────────────────────────────────────
#  BLACK-SCHOLES-MERTON CALCULATOR
# ─────────────────────────────────────────────

def bsm_calculate(S, K, T, r, sigma, option_type="call"):
    """
    S     = Spot price
    K     = Strike price
    T     = Time to expiry in YEARS
    r     = Risk-free rate (e.g. 0.05 for 5%)
    sigma = Implied volatility (e.g. 0.20 for 20%)
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None

    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # Option prices
    if option_type == "call":
        price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
        rho   = K * T * math.exp(-r * T) * norm.cdf(d2) / 100
    else:
        price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1
        rho   = -K * T * math.exp(-r * T) * norm.cdf(-d2) / 100

    gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T))
    vega  = S * math.sqrt(T) * norm.pdf(d1) / 100          # per 1% vol move
    theta = (-(S * norm.pdf(d1) * sigma) / (2 * math.sqrt(T))
             - r * K * math.exp(-r * T) * norm.cdf(d2 if option_type == "call" else -d2)) / 365

    return {
        "price" : round(price, 4),
        "d1"    : round(d1, 6),
        "d2"    : round(d2, 6),
        "delta" : round(delta, 6),
        "gamma" : round(gamma, 6),
        "theta" : round(theta, 6),
        "vega"  : round(vega, 6),
        "rho"   : round(rho, 6),
    }


def run_scenario(S, K, T_days, r, sigma, option_type, lot_size, contracts,
                 price_shock_pct, iv_shock_pct):
    """Run a stress scenario by shifting spot and vol."""
    new_S   = S * (1 + price_shock_pct / 100)
    new_sig = sigma + iv_shock_pct / 100
    new_T   = max(T_days / 365, 0.001)

    base   = bsm_calculate(S, K, T_days / 365, r, sigma, option_type)
    result = bsm_calculate(new_S, K, new_T, r, new_sig, option_type)

    if not result or not base:
        return None

    position_size = lot_size * contracts
    pnl = (result["price"] - base["price"]) * position_size

    return {
        "new_spot"    : round(new_S, 2),
        "new_iv"      : round(new_sig * 100, 2),
        "new_price"   : result["price"],
        "base_price"  : base["price"],
        "delta"       : result["delta"],
        "gamma"       : result["gamma"],
        "vega"        : result["vega"],
        "theta"       : result["theta"],
        "port_delta"  : round(result["delta"] * position_size, 4),
        "pnl"         : round(pnl, 2),
    }


# ─────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/study")
def study():
    return render_template("study.html")


@app.route("/calculator", methods=["GET", "POST"])
def calculator():
    result = None
    form   = {}

    if request.method == "POST":
        try:
            form = {
                "spot"        : float(request.form["spot"]),
                "strike"      : float(request.form["strike"]),
                "days"        : float(request.form["days"]),
                "rate"        : float(request.form["rate"]) / 100,
                "iv"          : float(request.form["iv"]) / 100,
                "option_type" : request.form["option_type"],
                "lot_size"    : int(request.form.get("lot_size", 65)),
                "contracts"   : int(request.form.get("contracts", 1)),
            }
            greeks = bsm_calculate(
                form["spot"], form["strike"],
                form["days"] / 365,
                form["rate"], form["iv"],
                form["option_type"]
            )
            if greeks:
                pos_size         = form["lot_size"] * form["contracts"]
                greeks["port_delta"] = round(greeks["delta"] * pos_size, 4)
                greeks["port_vega"]  = round(greeks["vega"]  * pos_size * 100, 4)
                greeks["dollar_delta"] = round(greeks["delta"] * form["spot"] * pos_size, 2)
                result = greeks
        except Exception as e:
            result = {"error": str(e)}

    # Default values shown on page load
    if not form:
        form = {
            "spot": 23741, "strike": 23750, "days": 5,
            "rate": 5.28, "iv": 22.49,
            "option_type": "call", "lot_size": 65, "contracts": 1
        }

    return render_template("calculator.html", result=result, form=form)


@app.route("/scenarios", methods=["GET", "POST"])
def scenarios():
    results = []
    form    = {}

    SCENARIOS = [
        {"label": "Base (No Change)",   "price_shock":  0,   "iv_shock":  0},
        {"label": "Mild Drop",          "price_shock": -2,   "iv_shock":  2},
        {"label": "Crash (-5%, +10% vol)", "price_shock": -5, "iv_shock": 10},
        {"label": "Big Rally (+5%)",    "price_shock":  5,   "iv_shock": -3},
        {"label": "Vol Spike Only",     "price_shock":  0,   "iv_shock": 15},
        {"label": "Black Swan (-10%)",  "price_shock": -10,  "iv_shock": 20},
    ]

    if request.method == "POST":
        try:
            form = {
                "spot"        : float(request.form["spot"]),
                "strike"      : float(request.form["strike"]),
                "days"        : float(request.form["days"]),
                "rate"        : float(request.form["rate"]) / 100,
                "iv"          : float(request.form["iv"]) / 100,
                "option_type" : request.form["option_type"],
                "lot_size"    : int(request.form.get("lot_size", 65)),
                "contracts"   : int(request.form.get("contracts", 1)),
            }
            for sc in SCENARIOS:
                r = run_scenario(
                    form["spot"], form["strike"], form["days"],
                    form["rate"], form["iv"], form["option_type"],
                    form["lot_size"], form["contracts"],
                    sc["price_shock"], sc["iv_shock"]
                )
                if r:
                    r["label"]        = sc["label"]
                    r["price_shock"]  = sc["price_shock"]
                    r["iv_shock"]     = sc["iv_shock"]
                    results.append(r)
        except Exception as e:
            results = []

    if not form:
        form = {
            "spot": 23741, "strike": 23750, "days": 5,
            "rate": 5.28, "iv": 22.49,
            "option_type": "call", "lot_size": 65, "contracts": 1
        }

    return render_template("scenarios.html", results=results, form=form)


if __name__ == "__main__":
    app.run(debug=True)
