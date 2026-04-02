import math
from flask import Flask, render_template, request

app = Flask(__name__)

# ── Pure-Python normal distribution helpers ────────────────────────────────
def _erf(x):
    """Abramowitz & Stegun approximation for erf."""
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
                 - 0.284496736) * t + 0.254829592) * t * math.exp(-x * x)
    return sign * y

def norm_cdf(x):
    return 0.5 * (1.0 + _erf(x / math.sqrt(2)))

def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


# ── BSM (with continuous dividend yield q) ─────────────────────────────────
def bsm_greeks(S, K, T, r, sigma, q=0.0):
    """
    Returns dict with price, delta, gamma, theta, vega, rho for both call & put.
    r, sigma, q are in decimal (0.05, 0.22, 0.015 etc.)
    """
    if T <= 0 or sigma <= 0:
        raise ValueError("T and sigma must be positive.")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    Nd1  = norm_cdf(d1)
    Nd2  = norm_cdf(d2)
    Nd1n = norm_cdf(-d1)
    Nd2n = norm_cdf(-d2)
    nd1  = norm_pdf(d1)

    disc  = math.exp(-r * T)
    divsc = math.exp(-q * T)

    call_price = S * divsc * Nd1 - K * disc * Nd2
    put_price  = K * disc * Nd2n - S * divsc * Nd1n

    delta_call =  divsc * Nd1
    delta_put  =  divsc * (Nd1 - 1)

    gamma = divsc * nd1 / (S * sigma * math.sqrt(T))

    theta_call = (
        -S * divsc * nd1 * sigma / (2 * math.sqrt(T))
        - r * K * disc * Nd2
        + q * S * divsc * Nd1
    ) / 365
    theta_put = (
        -S * divsc * nd1 * sigma / (2 * math.sqrt(T))
        + r * K * disc * Nd2n
        - q * S * divsc * Nd1n
    ) / 365

    vega = S * divsc * math.sqrt(T) * nd1 / 100  # per 1% IV

    rho_call =  K * T * disc * Nd2  / 100
    rho_put  = -K * T * disc * Nd2n / 100

    return {
        'd1': round(d1, 4), 'd2': round(d2, 4),
        'call': {
            'price': round(call_price, 2),
            'delta': round(delta_call, 4),
            'gamma': round(gamma, 6),
            'theta': round(theta_call, 2),
            'vega':  round(vega, 2),
            'rho':   round(rho_call, 4),
        },
        'put': {
            'price': round(put_price, 2),
            'delta': round(delta_put, 4),
            'gamma': round(gamma, 6),
            'theta': round(theta_put, 2),
            'vega':  round(vega, 2),
            'rho':   round(rho_put, 4),
        }
    }


# ── Binomial (CRR) with dividends ─────────────────────────────────────────
def binomial_greeks(S, K, T, r, sigma, q=0.0, steps=200, option_type='call'):
    """
    Cox-Ross-Rubinstein binomial tree.
    Returns price + approximate greeks via finite difference on the tree.
    """
    dt = T / steps
    u  = math.exp(sigma * math.sqrt(dt))
    d  = 1.0 / u
    p  = (math.exp((r - q) * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    is_call = (option_type == 'call')

    # Terminal payoffs
    prices = [S * (u ** (steps - 2 * j)) for j in range(steps + 1)]
    values = [max(px - K, 0) if is_call else max(K - px, 0) for px in prices]

    # Backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            values[j] = disc * (p * values[j] + (1 - p) * values[j + 1])

    price = values[0]

    # Greeks via small perturbations on tree
    dS = S * 0.01
    def _price(S_):
        px_ = [S_ * (u ** (i - 2*j)) for j in range(steps+1)]
        v_ = [max(px_[j]-K,0) if is_call else max(K-px_[j],0) for j in range(steps+1)]
        for i in range(steps-1,-1,-1):
            for j in range(i+1):
                v_[j] = disc*(p*v_[j] + (1-p)*v_[j+1])
        return v_[0]

    p_up   = _price(S + dS)
    p_down = _price(S - dS)

    delta = (p_up - p_down) / (2 * dS)
    gamma = (p_up - 2 * price + p_down) / (dS ** 2)

    dt_small = 1/365
    T2 = T - dt_small
    if T2 > 0:
        steps2 = max(10, int(steps * T2 / T))
        dt2 = T2 / steps2
        u2 = math.exp(sigma * math.sqrt(dt2))
        d2_ = 1.0 / u2
        p2 = (math.exp((r - q) * dt2) - d2_) / (u2 - d2_)
        disc2 = math.exp(-r * dt2)
        px2 = [S * (u2 ** (steps2 - 2*j)) for j in range(steps2+1)]
        v2  = [max(x-K,0) if is_call else max(K-x,0) for x in px2]
        for i in range(steps2-1,-1,-1):
            for j in range(i+1):
                v2[j] = disc2*(p2*v2[j]+(1-p2)*v2[j+1])
        theta = (v2[0] - price) / dt_small / 365  # approximate daily
    else:
        theta = 0.0

    # Vega: bump IV by 1%
    sigma2 = sigma + 0.01
    u2v = math.exp(sigma2 * math.sqrt(dt))
    d2v = 1.0 / u2v
    p2v = (math.exp((r-q)*dt) - d2v) / (u2v - d2v)
    disc2v = math.exp(-r*dt)
    px2v = [S*(u2v**(steps-2*j)) for j in range(steps+1)]
    v2v  = [max(x-K,0) if is_call else max(K-x,0) for x in px2v]
    for i in range(steps-1,-1,-1):
        for j in range(i+1):
            v2v[j] = disc2v*(p2v*v2v[j]+(1-p2v)*v2v[j+1])
    vega = v2v[0] - price  # change per 1% IV bump

    return {
        'price': round(price, 2),
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta, 2),
        'vega':  round(vega, 2),
    }


# ── Put-Call Parity ────────────────────────────────────────────────────────
def put_call_parity(S, K, T, r, q=0.0, call_price=None, put_price=None):
    """
    C - P = S*e^(-qT) - K*e^(-rT)
    Given one, solve for the other.
    """
    lhs = S * math.exp(-q * T) - K * math.exp(-r * T)  # forward value
    if call_price is not None:
        implied_put = call_price - lhs
        return {'call': round(call_price, 2), 'put': round(implied_put, 2), 'parity_lhs': round(lhs, 4)}
    elif put_price is not None:
        implied_call = put_price + lhs
        return {'call': round(implied_call, 2), 'put': round(put_price, 2), 'parity_lhs': round(lhs, 4)}
    else:
        return {'parity_lhs': round(lhs, 4)}


# ── Shared form-parsing helper ─────────────────────────────────────────────
def _flt(form, key, default=None):
    try:
        val = form.get(key, '').strip()
        return float(val) if val else default
    except (ValueError, AttributeError):
        return default

def _int(form, key, default=None):
    try:
        val = form.get(key, '').strip()
        return int(val) if val else default
    except (ValueError, AttributeError):
        return default


# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/study')
def study():
    return render_template('study.html')


@app.route('/calculator', methods=['GET', 'POST'])
def calculator():
    form = {}
    result_call = result_put = None
    binom_call = binom_put = None
    pcp_result = None
    model = 'bsm'
    error = None

    if request.method == 'POST':
        model = request.form.get('model', 'bsm')
        S   = _flt(request.form, 'spot')
        K   = _flt(request.form, 'strike')
        days = _flt(request.form, 'days')
        r_pct = _flt(request.form, 'rate', 5.28)
        iv_pct = _flt(request.form, 'iv')
        q_pct = _flt(request.form, 'dividend', 0.0)
        lot_size  = _flt(request.form, 'lot_size', 65)
        contracts = _flt(request.form, 'contracts', 1)

        form = dict(
            spot=S, strike=K, days=days, rate=r_pct, iv=iv_pct,
            dividend=q_pct, lot_size=lot_size, contracts=contracts, model=model
        )

        if model in ('bsm', 'binomial'):
            if not all([S, K, days, iv_pct]):
                error = 'Please fill Spot, Strike, Days, and IV.'
            else:
                T = days / 365
                r = r_pct / 100
                sigma = iv_pct / 100
                q = (q_pct or 0.0) / 100
                lots = (lot_size or 65) * (contracts or 1)

                try:
                    if model == 'bsm':
                        res = bsm_greeks(S, K, T, r, sigma, q)
                        d1v, d2v = res['d1'], res['d2']
                        c, p_ = res['call'], res['put']

                        result_call = {
                            'price':       c['price'],
                            'delta':       c['delta'],
                            'gamma':       c['gamma'],
                            'theta':       c['theta'],
                            'vega':        c['vega'],
                            'rho':         c['rho'],
                            'port_delta':  round(c['delta'] * lots, 2),
                            'daily_theta': round(c['theta'] * lots, 2),
                            'port_vega':   round(c['vega']  * lots, 2),
                            'dollar_delta':round(c['delta'] * S * lots, 0),
                            'd1': d1v, 'd2': d2v,
                        }
                        result_put = {
                            'price':       p_['price'],
                            'delta':       p_['delta'],
                            'gamma':       p_['gamma'],
                            'theta':       p_['theta'],
                            'vega':        p_['vega'],
                            'rho':         p_['rho'],
                            'port_delta':  round(p_['delta'] * lots, 2),
                            'daily_theta': round(p_['theta'] * lots, 2),
                            'port_vega':   round(p_['vega']  * lots, 2),
                            'dollar_delta':round(p_['delta'] * S * lots, 0),
                            'd1': d1v, 'd2': d2v,
                        }

                    else:  # binomial
                        bc = binomial_greeks(S, K, T, r, sigma, q, steps=200, option_type='call')
                        bp = binomial_greeks(S, K, T, r, sigma, q, steps=200, option_type='put')

                        def _wrap(b):
                            return {
                                'price':       b['price'],
                                'delta':       b['delta'],
                                'gamma':       b['gamma'],
                                'theta':       b['theta'],
                                'vega':        b['vega'],
                                'rho':         '—',
                                'port_delta':  round(b['delta'] * lots, 2),
                                'daily_theta': round(b['theta'] * lots, 2),
                                'port_vega':   round(b['vega']  * lots, 2),
                                'dollar_delta':round(b['delta'] * S * lots, 0),
                                'd1': '—', 'd2': '—',
                            }
                        result_call = _wrap(bc)
                        result_put  = _wrap(bp)

                except Exception as ex:
                    result_call = {'error': str(ex)}

        elif model == 'pcp':
            # Put-Call Parity
            S   = _flt(request.form, 'spot')
            K   = _flt(request.form, 'strike')
            days = _flt(request.form, 'days')
            r_pct = _flt(request.form, 'rate', 5.28)
            q_pct = _flt(request.form, 'dividend', 0.0)
            call_inp = _flt(request.form, 'pcp_call')
            put_inp  = _flt(request.form, 'pcp_put')

            if not all([S, K, days]):
                error = 'Please fill Spot, Strike, and Days.'
            elif call_inp is None and put_inp is None:
                error = 'Please provide either a Call price or a Put price.'
            else:
                T = days / 365
                r = r_pct / 100
                q = (q_pct or 0.0) / 100
                try:
                    pcp_result = put_call_parity(S, K, T, r, q,
                                                 call_price=call_inp,
                                                 put_price=put_inp)
                    # Also attach forward & arbitrage check
                    pcp_result['S']   = S
                    pcp_result['K']   = K
                    pcp_result['T']   = round(T, 4)
                    pcp_result['r']   = r_pct
                    pcp_result['q']   = q_pct or 0
                    pcp_result['disc_K'] = round(K * math.exp(-r * T), 2)
                    pcp_result['fwd_S']  = round(S * math.exp(-q * T), 2)
                    diff = round(pcp_result['call'] - pcp_result['put'] - pcp_result['parity_lhs'], 4)
                    pcp_result['arb_diff'] = diff
                except Exception as ex:
                    error = str(ex)

    return render_template(
        'calculator.html',
        form=form,
        model=model,
        result_call=result_call,
        result_put=result_put,
        pcp_result=pcp_result,
        error=error,
    )


@app.route('/scenarios', methods=['GET', 'POST'])
def scenarios():
    form = {}
    results = []

    if request.method == 'POST':
        S      = _flt(request.form, 'spot')
        K      = _flt(request.form, 'strike')
        days   = _int(request.form, 'days')   # INTEGER
        r_pct  = _flt(request.form, 'rate', 5.28)
        iv_pct = _flt(request.form, 'iv')
        q_pct  = _flt(request.form, 'dividend', 0.0)
        opt    = request.form.get('option_type', 'call')
        lot_size  = _int(request.form, 'lot_size', 65)  # INTEGER
        contracts = _int(request.form, 'contracts', 1)  # INTEGER

        form = dict(spot=S, strike=K, days=days, rate=r_pct, iv=iv_pct,
                    dividend=q_pct, option_type=opt, lot_size=lot_size, contracts=contracts)

        if S and K and days and iv_pct:
            SCENARIOS = [
                {'label': 'Base',       'price_shock': 0,    'iv_shock': 0},
                {'label': 'Mild Drop',  'price_shock': -2,   'iv_shock': 2},
                {'label': 'Crash',      'price_shock': -6,   'iv_shock': 10},
                {'label': 'Big Rally',  'price_shock': +5,   'iv_shock': -3},
                {'label': 'Vol Spike',  'price_shock': 0,    'iv_shock': 15},
                {'label': 'Black Swan', 'price_shock': -10,  'iv_shock': 20},
            ]
            lots = (lot_size or 65) * (contracts or 1)
            r = r_pct / 100
            q = (q_pct or 0.0) / 100

            base_res = None
            for sc in SCENARIOS:
                new_S  = S * (1 + sc['price_shock'] / 100)
                new_iv = max(iv_pct + sc['iv_shock'], 0.5)
                T      = days / 365
                sigma  = new_iv / 100

                try:
                    res = bsm_greeks(new_S, K, T, r, sigma, q)
                    g   = res['call'] if opt == 'call' else res['put']
                    price = g['price']

                    if base_res is None:
                        base_res = price
                    pnl = round((price - base_res) * lots, 0)

                    results.append({
                        'label':       sc['label'],
                        'price_shock': sc['price_shock'],
                        'iv_shock':    sc['iv_shock'],
                        'new_spot':    round(new_S, 0),
                        'new_iv':      round(new_iv, 1),
                        'price':       price,
                        'pnl':         pnl,
                        'delta':       g['delta'],
                        'gamma':       g['gamma'],
                        'theta':       g['theta'],
                        'vega':        g['vega'],
                        'port_delta':  round(g['delta'] * lots, 2),
                    })
                except Exception:
                    pass

    return render_template('scenarios.html', form=form, results=results)


@app.route('/portfolio', methods=['GET', 'POST'])
def portfolio():
    LIMITS = {'delta': 500, 'gamma': 50, 'vega': 200000, 'theta': 50000}
    positions = []
    portfolio_totals = None
    gauges = {}
    recs = []

    if request.method == 'POST':
        S     = _flt(request.form, 'spot')
        r_pct = _flt(request.form, 'rate', 5.28)
        q_pct = _flt(request.form, 'dividend', 0.0)
        r     = r_pct / 100
        q     = (q_pct or 0.0) / 100

        i = 0
        while True:
            name = request.form.get(f'name_{i}')
            if name is None:
                break
            strike   = _flt(request.form, f'strike_{i}')
            iv_pct   = _flt(request.form, f'iv_{i}')
            days     = _int(request.form, f'days_{i}')    # INTEGER
            lots     = _int(request.form, f'lots_{i}')    # INTEGER
            lot_size = _int(request.form, f'lot_size_{i}', 65)  # INTEGER
            opt_type = request.form.get(f'type_{i}', 'call')

            if strike and iv_pct and days and lots and lot_size and S:
                T      = days / 365
                sigma  = iv_pct / 100
                mult   = lots * lot_size
                try:
                    res = bsm_greeks(S, strike, T, r, sigma, q)
                    g   = res['call'] if opt_type == 'call' else res['put']
                    positions.append({
                        'name':    name or f'Pos {i+1}',
                        'type':    opt_type,
                        'strike':  strike,
                        'iv':      iv_pct,
                        'days':    days,
                        'lots':    lots,
                        'lot_size':lot_size,
                        'price':   g['price'],
                        'p_delta': round(g['delta'] * mult, 2),
                        'p_gamma': round(g['gamma'] * mult, 4),
                        'p_theta': round(g['theta'] * mult, 2),
                        'p_vega':  round(g['vega']  * mult, 2),
                    })
                except Exception:
                    pass
            i += 1

        if positions:
            tot_delta = sum(p['p_delta'] for p in positions)
            tot_gamma = sum(p['p_gamma'] for p in positions)
            tot_theta = sum(p['p_theta'] for p in positions)
            tot_vega  = sum(p['p_vega']  for p in positions)
            portfolio_totals = {
                'delta':       round(tot_delta, 2),
                'gamma':       round(tot_gamma, 4),
                'theta':       round(tot_theta, 2),
                'vega':        round(tot_vega,  2),
                'dollar_delta':round(tot_delta  * (S or 0), 0),
            }

            def _gauge(key, value, limit):
                pct = min(round(abs(value) / limit * 100, 1), 999)
                zone = 'breach' if pct >= 100 else 'caution' if pct >= 70 else 'safe'
                return {'value': round(value, 2), 'limit': limit, 'pct': pct, 'zone': zone}

            gauges = {
                'delta': _gauge('delta', tot_delta, LIMITS['delta']),
                'gamma': _gauge('gamma', tot_gamma * 1000, LIMITS['gamma']),
                'vega':  _gauge('vega',  tot_vega,  LIMITS['vega']),
                'theta': _gauge('theta', tot_theta, LIMITS['theta']),
            }

            # Recommendations
            if abs(tot_delta) > LIMITS['delta']:
                recs.append(f"🔴 Delta {round(tot_delta,1)} exceeds ±{LIMITS['delta']} limit — hedge with futures or offsetting options.")
            elif abs(tot_delta) > LIMITS['delta'] * 0.7:
                recs.append(f"🟡 Delta {round(tot_delta,1)} approaching limit — consider delta hedge.")
            else:
                recs.append(f"🟢 Delta {round(tot_delta,1)} within safe range.")

            if abs(tot_gamma * 1000) > LIMITS['gamma']:
                recs.append(f"🔴 Gamma exposure high — large spot moves will cause rapid delta change.")
            elif abs(tot_gamma * 1000) > LIMITS['gamma'] * 0.7:
                recs.append(f"🟡 Gamma approaching limit — be alert near expiry and large moves.")

            if abs(tot_vega) > LIMITS['vega']:
                recs.append(f"🔴 Vega ₹{round(tot_vega,0)} breached — portfolio highly sensitive to IV changes.")
            elif abs(tot_vega) > LIMITS['vega'] * 0.7:
                recs.append(f"🟡 Vega elevated — an IV spike could materially impact P&L.")
            else:
                recs.append(f"🟢 Vega ₹{round(tot_vega,0)} within limits.")

            if tot_theta < -LIMITS['theta']:
                recs.append(f"🔴 Theta ₹{round(tot_theta,0)}/day — heavy time decay. Reconsider net long options.")
            elif tot_theta < -LIMITS['theta'] * 0.7:
                recs.append(f"🟡 Theta ₹{round(tot_theta,0)}/day — meaningful daily bleed.")
            elif tot_theta > 0:
                recs.append(f"🟢 Theta ₹{round(tot_theta,0)}/day — net theta positive (premium seller).")

    return render_template(
        'portfolio.html',
        positions=positions,
        portfolio=portfolio_totals,
        gauges=gauges,
        recs=recs,
        limits=LIMITS,
    )


@app.route('/arena')
def arena():
    return render_template('arena.html')

@app.route('/strategy-lab')
def strategy_lab():
    return render_template('strategy_lab.html')

# ══════════════════════════════════════════════════════════════════════════
#  ADD THESE ROUTES TO app.py
#  Place them alongside your existing @app.route entries
# ══════════════════════════════════════════════════════════════════════════

@app.route('/volatility-engine')
def volatility_engine():
    """Module 3: IV Rank & Percentile Trading System"""
    return render_template('volatility_engine.html')


# ── Optional: IV Rank API endpoint (for future calculator integrations) ──
@app.route('/api/iv-rank', methods=['POST'])
def iv_rank_api():
    """
    POST JSON: { current_iv, high_iv, low_iv, hv, days_below }
    Returns:   { iv_rank, iv_percentile, iv_hv_spread, regime, signal }
    """
    import json
    from flask import request, jsonify

    data = request.get_json(force=True)
    current_iv  = _float(data.get('current_iv', 0))
    high_iv     = _float(data.get('high_iv', 1))
    low_iv      = _float(data.get('low_iv', 0))
    hv          = _float(data.get('hv', 0))
    days_below  = _float(data.get('days_below', 0))

    if high_iv <= low_iv:
        return jsonify({'error': '52-week high must exceed low'}), 400

    iv_rank      = round((current_iv - low_iv) / (high_iv - low_iv) * 100, 1)
    iv_percentile = round(days_below / 252 * 100, 1)
    iv_hv_spread  = round(current_iv - hv, 2)

    if iv_rank >= 50 or iv_percentile >= 50:
        regime = 'high'
        signal = 'sell_premium'
    elif iv_rank < 30 and iv_percentile < 40:
        regime = 'low'
        signal = 'buy_premium'
    else:
        regime = 'neutral'
        signal = 'neutral'

    return jsonify({
        'iv_rank':        iv_rank,
        'iv_percentile':  iv_percentile,
        'iv_hv_spread':   iv_hv_spread,
        'regime':         regime,
        'signal':         signal,
        'current_iv':     current_iv,
        'hv':             hv,
    })


# ══════════════════════════════════════════════════════════════════════════
#  ADD THIS LINK TO YOUR nav in base.html
#  (find your existing nav items and add alongside them)
# ══════════════════════════════════════════════════════════════════════════
#
#  <a href="/volatility-engine" class="nav-link">⚡ Volatility Engine</a>
#
# ══════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════
#  ADD THESE TO app.py — Module 4: Weekly Decay Optimizer
# ══════════════════════════════════════════════════════════════════════════

@app.route('/decay-optimizer')
def decay_optimizer():
    """Module 4: Weekly Options Decay Optimizer — Theta Timing Engine"""
    return render_template('decay_optimizer.html')


# ── Optional: Theta calculation API endpoint ──
@app.route('/api/theta-timing', methods=['POST'])
def theta_timing_api():
    """
    POST JSON: { dte, premium, iv_rank, event_flag }
    Returns:   { signal, gamma_score, daily_theta, tv_remaining, 
                 target_exit, stop_loss, recommended_strategy }
    """
    from flask import request, jsonify
    import math

    data = request.get_json(force=True)
    dte       = max(0, min(7, int(data.get('dte', 4))))
    premium   = _float(data.get('premium', 180))
    iv_rank   = _float(data.get('iv_rank', 50))
    event     = data.get('event_flag', 'none')  # none|minor|major|earnings

    tv_remaining  = round(premium * math.sqrt(dte / 7))
    daily_theta   = round(premium * (math.sqrt(dte/7) - math.sqrt(max(0, dte-1)/7))) if dte > 0 else premium
    gamma_score   = 95 if dte <= 1 else 72 if dte <= 2 else 48 if dte <= 3 else 25 if dte <= 4 else 12
    target_exit   = round(premium * 0.5)
    stop_loss     = round(premium * 2)  # 200% of credit = max loss threshold

    # Signal logic
    if event in ('major', 'earnings'):
        signal = 'avoid_event'
        strategy = 'Skip week — wait for post-event IV crush'
    elif dte >= 6:
        signal = 'too_early'
        strategy = 'Observe — wait for 4-5 DTE window'
    elif dte >= 4:
        signal = 'optimal'
        strategy = 'Iron Condor (defined risk) — enter now'
    elif dte == 3:
        signal = 'aggressive'
        strategy = 'Iron Condor at reduced size — aggressive entry'
    else:
        signal = 'avoid_gamma'
        strategy = 'Avoid fresh entry — gamma risk too high'

    return jsonify({
        'dte':                dte,
        'signal':             signal,
        'gamma_score':        gamma_score,
        'daily_theta':        daily_theta,
        'tv_remaining':       tv_remaining,
        'target_exit':        target_exit,
        'stop_loss_trigger':  stop_loss,
        'recommended_strategy': strategy,
        'iv_rank':            iv_rank,
        'event_flag':         event,
    })


# ══════════════════════════════════════════════════════════════════════════
#  ADD THIS LINK TO base.html nav (alongside existing nav items):
#  <a href="/decay-optimizer" class="nav-link">⏱ Decay Optimizer</a>
#
#  RECOMMENDED NAV ORDER (left → right):
#  Home | ⚡ Volatility Engine | ⏱ Decay Optimizer | 📐 Strategy Lab |
#  📊 Portfolio | 🔢 Calculator | 📈 Scenarios | 🏟 Arena
# ══════════════════════════════════════════════════════════════════════════



if __name__ == '__main__':
    app.run(debug=True)
