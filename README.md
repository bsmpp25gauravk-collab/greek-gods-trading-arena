# ⚡ Greek Gods Trading Arena

**Options Greeks Portfolio Dashboard** — ZeTheta Algorithms Research Project

A gamified educational simulation for options portfolio risk management. Built with Python (Flask) and Black-Scholes-Merton calculations.

---

## 📁 Project Structure

```
greek_gods_arena/
├── app.py                  ← Main Flask app + BSM calculator logic
├── requirements.txt        ← Python packages needed
├── templates/
│   ├── base.html           ← Navigation + footer (shared layout)
│   ├── index.html          ← Home / Landing page
│   ├── study.html          ← Greeks study section (all 5 Greeks explained)
│   ├── calculator.html     ← BSM Calculator
│   └── scenarios.html      ← Scenario stress test engine
└── static/
    ├── css/style.css       ← All styling
    └── js/main.js          ← Small JavaScript enhancements
```

---

## 🚀 How to Run Locally (Step by Step)

### Step 1: Make sure Python is installed
Open your terminal (Command Prompt on Windows, Terminal on Mac/Linux) and type:
```bash
python --version
```
You need Python 3.8 or higher.

### Step 2: Install the required packages
Navigate to this project folder and run:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Flask app
```bash
python app.py
```

### Step 4: Open in browser
Go to: **http://127.0.0.1:5000**

You'll see the Greek Gods Trading Arena homepage. That's it!

---

## 🌐 How to Deploy on GitHub + Render (Free Hosting)

### Part A: Push to GitHub

1. Go to [github.com](https://github.com) → Sign in → Click **New repository**
2. Name it `greek-gods-trading-arena` → Make it **Public** → Click **Create repository**
3. In your terminal, navigate to this project folder:
   ```bash
   cd greek_gods_arena
   git init
   git add .
   git commit -m "Initial commit - Greek Gods Trading Arena"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/greek-gods-trading-arena.git
   git push -u origin main
   ```
   *(Replace YOUR_USERNAME with your actual GitHub username)*

### Part B: Deploy on Render (Free)

Render hosts Flask apps for free. No credit card needed.

1. Go to [render.com](https://render.com) → Sign up with GitHub
2. Click **New +** → **Web Service**
3. Connect your GitHub repo → Select `greek-gods-trading-arena`
4. Fill in these settings:
   - **Name**: greek-gods-arena (or anything)
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
5. Click **Create Web Service**
6. Wait 2-3 minutes → Render gives you a live URL like: `https://greek-gods-arena.onrender.com`

**That's your live website! Share the link in your project submission.**

---

## 📊 What Each Page Does

| Page | URL | Description |
|------|-----|-------------|
| Home | `/` | Landing page with Greek cards and game levels overview |
| Study | `/study` | Full explanations of all 5 Greeks with formulas, examples, case studies |
| Calculator | `/calculator` | Enter any Nifty option → get all Greeks calculated using BSM |
| Scenarios | `/scenarios` | Stress test across 6 market scenarios (Base, Crash, Rally, Black Swan, etc.) |

---

## 🧮 BSM Formula Reference

```
d1 = [ln(S/K) + (r + σ²/2) × T] / (σ × √T)
d2 = d1 - σ × √T

Call Price = S × N(d1) - K × e^(-rT) × N(d2)
Put  Price = K × e^(-rT) × N(-d2) - S × N(-d1)

Delta (Call) = N(d1)
Delta (Put)  = N(d1) - 1
Gamma        = N'(d1) / (S × σ × √T)
Theta        = -[S × N'(d1) × σ / (2√T)] - r × K × e^(-rT) × N(d2)  (÷365 for daily)
Vega         = S × √T × N'(d1)  (÷100 for per 1% IV move)
Rho (Call)   = K × T × e^(-rT) × N(d2)  (÷100 for per 1% rate move)
```

Where:
- S = Spot Price (Nifty level)
- K = Strike Price
- T = Time to expiry in years (days ÷ 365)
- r = Risk-free rate (RBI repo rate)
- σ = Implied Volatility
- N() = Cumulative Normal Distribution
- N'() = Standard Normal PDF

---

## 🎮 Game Levels (Campaign Mode Design)

| Level | Name | Focus Greek | Challenge |
|-------|------|-------------|-----------|
| 1 | Delta Rookie | Δ Delta | Hedge 1 lot to delta-neutral |
| 2 | Delta Apprentice | Δ Delta | Keep portfolio delta in [-50, +50] through ±5% moves |
| 3 | Gamma Guardian | Γ Gamma | Navigate volatility spike, manage gamma near expiry |
| 4 | Theta Harvester | Θ Theta | Maximize theta income on Bank Nifty weeklies |
| 5 | Vega Voyager | ν Vega | Navigate earnings IV expansion + IV crush |
| 6 | Greek Grandmaster | All | Full crisis scenario: manage all Greeks simultaneously |

---

## 📚 Key Case Studies Covered

- **GameStop Gamma Squeeze (Jan 2021)** — Gamma amplification, short option risk
- **XIV Volatility Collapse (Feb 2018)** — Vega blow-up, short volatility dangers
- **Bank Nifty Weekly Options (India)** — Theta acceleration, pin risk at expiry
- **LTCM Collapse (1998)** — Portfolio Greeks failure, correlation breakdown

---

*ZeTheta Algorithms Private Limited · Options Greeks Portfolio Dashboard · 15-Day Research Project*
*For educational purposes only. Not financial advice.*
