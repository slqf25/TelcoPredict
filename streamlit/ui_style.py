"""
ui_style.py — visual design system for the Streamlit prototype: colours, icon
maps, the global CSS injector, and small HTML/markdown helper functions used
to build the click-driven cards/tower.

Split out of app.py so the page-flow/business logic there (data wiring,
prediction pipeline, report sections) isn't buried under ~250 lines of CSS —
keeps app.py focused and easier to review.
"""

import re

import streamlit as st

# Report palette (Section colours) — shared with eda_plots.py / evaluation.py
NAVY, BLUE, LIGHT, RED = "#1F4E79", "#2E75B6", "#9DC3E6", "#C00000"
GREEN, AMBER = "#2E8B57", "#E8A317"

MODEL_ICON = {
    "Logistic Regression": "show_chart",
    "Decision Tree": "account_tree",
    "Random Forest": "forest",
    "XGBoost": "rocket_launch",
}

ADDON_ICONS = {"OnlineSecurity": "shield", "OnlineBackup": "cloud_upload",
               "DeviceProtection": "phonelink_lock", "TechSupport": "support_agent",
               "StreamingTV": "live_tv", "StreamingMovies": "movie"}

# Material Symbols (not emoji) for the icon click-cards — icon name -> display text
CONTRACT_ICON = {"Month-to-month": "autorenew", "One year": "calendar_month", "Two year": "lock"}
PAYMENT_ICON = {
    "Electronic check": ("receipt_long", "E-check"),
    "Mailed check": ("local_post_office", "Mailed"),
    "Bank transfer (automatic)": ("account_balance", "Bank auto"),
    "Credit card (automatic)": ("credit_card", "Card auto"),
}
INTERNET_ICON = {"DSL": "cable", "Fiber optic": "speed", "No": "wifi_off"}
LOCK_ICON = {"Month-to-month": "lock_open", "One year": "lock_clock", "Two year": "lock"}
# Tenure band (1-5) -> membership tier label + tower mast gradient
TIER_INFO = {
    1: ("NEW", "linear-gradient(135deg,#8e9aaf,#5c6370)"),
    2: ("BRONZE", "linear-gradient(135deg,#c98a4b,#8a5a2b)"),
    3: ("SILVER", "linear-gradient(135deg,#c9ccd1,#8b8f96)"),
    4: ("GOLD", "linear-gradient(135deg,#E8A317,#a97a10)"),
    5: ("PLATINUM", "linear-gradient(135deg,#1F4E79,#2E75B6)"),
}


def inject_css():
    """Injects the app's global stylesheet. Call once, near the top of app.py,
    after st.set_page_config()."""
    st.markdown(f"""
<style>
  /* Reduce top padding / space */
  .block-container {{
      padding-top: 1.5rem !important;
      padding-bottom: 0rem !important;
  }}
  [data-testid="stHeader"] {{
      height: 2rem !important;
      background: transparent !important;
  }}

  /* Dynamic Background */
  .stApp {{
      background: radial-gradient(circle at 15% 50%, rgba(64, 156, 255, 0.08), transparent 40%),
                  radial-gradient(circle at 85% 30%, rgba(255, 105, 180, 0.08), transparent 40%);
      background-attachment: fixed;
  }}

  /* Minimal hero — no card, no border, no blur. Oversized bold type sitting
     directly on the page background (the .stApp radial gradients behind it are
     enough colour); a muted single-line subtitle underneath. */
  .app-banner {{ padding: 4px 2px 22px; margin-bottom: 4px; text-align:center; }}
  .app-banner .t {{ font-size: 3.1rem; font-weight: 800; letter-spacing:-1.5px;
    line-height:1.05; color: var(--text-color); }}
  .app-banner .s {{ opacity:.55; font-size:1.05rem; margin:10px auto 0; font-weight:400;
    max-width: 620px; }}

  /* Buttons */
  .stButton > button {{
      border-radius: 16px !important; /* Squircle */
      backdrop-filter: blur(10px);
      -webkit-backdrop-filter: blur(10px);
      transition: all 0.2s ease-in-out;
  }}
  .stButton > button:hover {{
      transform: scale(1.02);
  }}

  /* Inputs (Squircles) */
  .stSelectbox > div > div,
  .stTextInput > div > div,
  .stNumberInput > div > div,
  .stSlider > div > div {{
      border-radius: 16px !important;
  }}

  /* Metric Cards */
  [data-testid="stMetric"] {{
      background: color-mix(in srgb, var(--background-color) 40%, transparent);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid color-mix(in srgb, var(--text-color) 8%, transparent);
      border-radius: 20px;
      padding: 16px 20px;
      box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
  }}
  div[data-testid="stMetricValue"] {{ color: var(--text-color); font-weight: 700; letter-spacing: -0.5px; }}

  /* DataFrames */
  [data-testid="stDataFrame"] {{
      border-radius: 16px !important;
      overflow: hidden;
      border: 1px solid color-mix(in srgb, var(--text-color) 8%, transparent);
  }}

  /* Badges & Signal bars (Telco motif — a phone-reception read, not a generic progress bar) */
  .risk-badge {{ display:inline-block; padding:10px 24px; border-radius:16px;
    font-weight:700; font-size:1.15rem; color:#fff; letter-spacing:.2px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.1); }}
  .signal-bars {{ display:flex; align-items:flex-end; gap:5px; height:44px; margin:10px 0 4px; }}
  .signal-bar {{ width:11px; border-radius:3px 3px 0 0;
    background: color-mix(in srgb, var(--text-color) 12%, transparent);
    transition: background 0.4s ease, box-shadow 0.4s ease; }}
  .signal-bar:nth-child(1) {{ height:20%; }}
  .signal-bar:nth-child(2) {{ height:40%; }}
  .signal-bar:nth-child(3) {{ height:60%; }}
  .signal-bar:nth-child(4) {{ height:80%; }}
  .signal-bar:nth-child(5) {{ height:100%; }}
  .signal-bar.on {{ box-shadow: 0 2px 10px rgba(0,0,0,0.15); }}
  .signal-bar.flicker {{ animation: barFlicker 1.2s ease-in-out infinite; }}
  @keyframes barFlicker {{ 0%, 100% {{ opacity:1; }} 50% {{ opacity:.3; }} }}

  /* Telecom interaction map: the tower is visual navigation, while the selected
     module opens a real Streamlit control dock immediately below it. */
  div.st-key-customer_network {{ width:100%; }}
  .tower-title {{ text-align:center; font-weight:750; font-size:1.05rem;
    margin:2px 0 4px; letter-spacing:-.1px; }}
  div.st-key-tower_wrap {{ position:relative; flex:0 0 460px !important;
    width:min(100%, 860px) !important; max-width:860px !important;
    align-self:center !important; height:460px !important; margin:4px auto 16px !important;
    border-radius:24px;
    background:linear-gradient(180deg, rgba(46,117,182,.045), transparent 52%),
      radial-gradient(ellipse at 50% 80%, rgba(46,117,182,.10), transparent 48%);
    border:1px solid color-mix(in srgb, var(--text-color) 8%, transparent); overflow:hidden; }}
  .tower-svg {{ position:absolute; top:82px; left:0; width:100%; height:330px; z-index:1; }}
  .tower-base {{ position:absolute; left:50%; bottom:6%; width:180px; height:6px;
    transform:translateX(-50%); background:var(--text-color); opacity:.20; border-radius:999px; }}
  .tower-beacon {{ position:absolute; left:50%; top:88px; width:8px; height:8px;
    transform:translateX(-50%); border-radius:50%; background:{RED}; z-index:2;
    box-shadow:0 0 9px 2px rgba(192,0,0,.45); animation:pulseGlow 1.8s ease-in-out infinite; }}

  /* Six consistent modules surround the tower. Shape and behaviour stay stable;
     only the selected module changes colour. */
  div[class*="st-key-hub_"] {{ position:absolute; width:142px; height:58px;
    z-index:4; text-align:center; }}
  div[class*="st-key-hub_"] [data-testid="stElementContainer"] {{ margin:0 !important; }}
  div[class*="st-key-hub_"] .stButton button {{
    width:142px !important; height:58px !important; border-radius:16px !important;
    display:flex !important; align-items:center !important; justify-content:center !important;
    padding:8px 12px !important; gap:5px !important;
    background:color-mix(in srgb, var(--background-color) 94%, transparent) !important;
    border:1px solid color-mix(in srgb, {NAVY} 32%, transparent) !important;
    box-shadow:0 6px 18px rgba(31,78,121,.10) !important;
    transition:transform .18s ease, box-shadow .18s ease, background .18s ease; }}
  div[class*="st-key-hub_"] .stButton button:hover {{
    transform:translateY(-2px) !important; box-shadow:0 9px 22px rgba(31,78,121,.17) !important; }}
  div[class*="st-key-hub_"] .stButton button[data-testid="stBaseButton-primary"] {{
    background:linear-gradient(145deg, {BLUE}, {NAVY}) !important;
    border-color:rgba(255,255,255,.38) !important;
    box-shadow:0 0 0 3px rgba(46,117,182,.13), 0 8px 22px rgba(31,78,121,.25) !important; }}
  div[class*="st-key-hub_"] .stButton button p {{
    white-space:nowrap !important; font-size:.72rem !important; font-weight:750 !important; }}
  div[class*="st-key-hub_"] .stButton [data-testid="stIconMaterial"] {{ font-size:18px !important; }}
  div.st-key-hub_demographics {{ top:4%; left:50%; transform:translateX(-50%); }}
  div.st-key-hub_charges {{ top:32%; left:7%; }}
  div.st-key-hub_contract {{ top:32%; right:7%; }}
  div.st-key-hub_billing {{ top:57%; right:4%; }}
  div.st-key-hub_connection {{ bottom:7%; left:7%; }}
  div.st-key-hub_addons {{ bottom:7%; right:7%; }}

  div.st-key-tower_control_dock {{ width:min(100%, 860px) !important; max-width:860px !important;
    align-self:center !important; margin:0 auto 18px !important; padding:18px 22px 12px;
    border-radius:20px; background:color-mix(in srgb, var(--background-color) 94%, {BLUE} 6%);
    border:1px solid color-mix(in srgb, {BLUE} 24%, transparent);
    box-shadow:0 10px 28px rgba(31,78,121,.08); animation:dockReveal .25s ease-out; }}
  div.st-key-tower_control_dock h4 {{ margin:0 !important; }}
  @keyframes dockReveal {{ from {{ opacity:0; transform:translateY(-5px); }}
    to {{ opacity:1; transform:translateY(0); }} }}

  div.st-key-bill_card .stButton button {{
    background: transparent !important; border: none !important; color:#fff !important;
    box-shadow: none !important; justify-content: flex-start !important; gap:6px !important;
    padding: 6px 4px !important; font-size: .78rem !important; font-weight: 600 !important;
    border-radius: 8px !important; width: 100%; line-height:1.2 !important; }}
  div.st-key-bill_card .stButton button:hover {{
    background: rgba(255,255,255,0.14) !important; transform: none !important; }}
  div.st-key-bill_card .stButton button p {{
    color:#fff !important; font-size:.82rem !important; line-height:1.25 !important; }}
  div.st-key-bill_card .stButton button p strong {{ font-weight:700 !important; }}
  div.st-key-bill_card .stButton [data-testid="stIconMaterial"] {{
    color:#fff !important; font-size:16px !important; opacity:.85; flex-shrink:0; }}

  /* Charges -> "Bill" card and Services -> "Coverage" card — same gradient-card
     mechanics as the old Contract Card (dynamic background set via a per-rerun
     inline <style> tag, since the colour depends on widget values known only
     at runtime). Bill's gradient is data-driven: it shifts cool-to-hot with
     the actual Monthly Charges value (a top churn predictor per the report),
     not an arbitrary decorative choice. Coverage has no single "intensity"
     scalar the way Charges does, so it stays a neutral glass card instead. */
  div.st-key-bill_card {{ border-radius:18px; padding:16px 16px 12px; margin-bottom:14px;
    box-shadow: 0 10px 26px rgba(0,0,0,0.20); position:relative; overflow:hidden;
    transition: background 0.5s ease, box-shadow .45s ease; gap:0 !important; }}
  div.st-key-bill_card:hover {{ box-shadow: 0 14px 32px rgba(0,0,0,0.26); }}
  div.st-key-bill_card::before {{ content:""; position:absolute; top:0; left:-130%;
    width:55%; height:100%; background: linear-gradient(115deg, transparent,
    rgba(255,255,255,0.30), transparent); transform: skewX(-20deg);
    transition: left .75s ease; pointer-events:none; z-index:1; }}
  div.st-key-bill_card:hover::before {{ left:130%; }}
  div.st-key-bill_card [data-testid="stElementContainer"] {{ margin:0 !important; }}
  .bc-label {{ font-size:.68rem; letter-spacing:1.3px; opacity:.75; font-weight:700;
    text-transform:uppercase; color:#fff; }}
  .bc-amount {{ font-size:2rem; font-weight:800; color:#fff; line-height:1.15; margin-top:2px; }}
  .bc-line {{ font-size:.78rem; opacity:.7; color:#fff; margin:12px 0 2px; }}
  .bc-total {{ font-size:1.25rem; font-weight:700; color:#fff; margin-bottom:10px; }}
  .bc-total-label {{ font-size:.68rem; font-weight:400; opacity:.65; margin-left:5px; }}

  div.st-key-coverage_card {{ border-radius:18px; padding:16px 18px; margin-bottom:14px;
    background: color-mix(in srgb, var(--background-color) 55%, transparent);
    backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
    border: 1px solid color-mix(in srgb, var(--text-color) 10%, transparent);
    box-shadow: 0 4px 20px rgba(0,0,0,0.04); }}
  .cov-label {{ font-size:.65rem; letter-spacing:1.3px; opacity:.55; font-weight:700;
    text-transform:uppercase; color: var(--text-color); margin:4px 2px 8px; }}
  .cov-count {{ font-size:.8rem; font-weight:600; opacity:.75; color: var(--text-color); }}
  /* Raw HTML injected via unsafe_allow_html doesn't get the `:material/x:` shortcode
     substitution Streamlit applies inside its own widgets — draw icons manually using
     the same "Material Symbols Rounded" webfont Streamlit already loads site-wide. */
  .mi {{ font-family:'Material Symbols Rounded'; font-weight:400; font-style:normal;
    vertical-align:middle; line-height:1; display:inline-block; }}

  /* Glass Findings */
  .finding {{ border-left: 6px solid {RED}; background:{RED}15; border-radius: 0 20px 20px 0;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    padding:16px 20px; margin:12px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.02); }}
  .finding.blue {{ border-left-color:{BLUE}; background:{BLUE}15; }}
  .finding.green {{ border-left-color:{GREEN}; background:{GREEN}15; }}

  .secnote {{ color: var(--text-color); opacity: 0.6; font-size:.95rem; margin:-4px 0 12px; }}

  /* Tab bar — simple segmented control (light track, thin dividers between
     inactive items, solid pill on the active one) instead of Streamlit's thin
     underline indicator. Plain hex colours, not var()/color-mix — those don't
     resolve inside this particular React-rendered node, so var()-based values
     silently computed to nothing here. Only the outermost tablist is centred
     (selector excludes anything nested inside a stTabPanel) so the wider 7-8
     item report sub-tab bars keep their natural left alignment. */
  .react-aria-SelectionIndicator {{ display:none !important; }}
  [role="tablist"] {{
    gap:0 !important; background: rgba(120,130,150,0.10); border-radius:12px;
    padding:4px; border-bottom:none !important; }}
  [role="tablist"]:not([data-testid="stTabPanel"] [role="tablist"]) {{
    width:fit-content; margin:0 auto; }}
  [data-testid="stTab"] {{
    border-radius:9px !important; padding:8px 18px !important; font-weight:600 !important;
    opacity:.65; border-right:1px solid rgba(120,130,150,0.22); transition: all .15s ease; }}
  [data-testid="stTab"]:last-child {{ border-right:none !important; }}
  [data-testid="stTab"]:hover {{ opacity:1; }}
  [data-testid="stTab"][aria-selected="true"] {{
    background:#2E75B6 !important; opacity:1; border-right-color:transparent !important; }}
  [data-testid="stTab"][aria-selected="true"] p {{ color:#fff !important; }}

  /* Chart entrance animation — every Plotly AND matplotlib figure fades/slides in on
     first render, so switching tabs or triggering a recompute feels alive rather than
     a static image just appearing. */
  [data-testid="stPlotlyChart"], [data-testid="stImage"] {{
      animation: fadeSlideIn 0.6s ease-out;
  }}
  @keyframes fadeSlideIn {{
      from {{ opacity: 0; transform: translateY(14px); }}
      to   {{ opacity: 1; transform: translateY(0); }}
  }}

  /* High-risk badge pulses to draw the eye; low/medium stay static (calmer states) */
  .risk-badge.pulse {{ animation: pulseGlow 1.6s ease-in-out infinite; }}
  @keyframes pulseGlow {{
      0%, 100% {{ box-shadow: 0 4px 16px rgba(192,0,0,0.25); transform: scale(1); }}
      50%      {{ box-shadow: 0 4px 28px rgba(192,0,0,0.55); transform: scale(1.03); }}
  }}

  /* Finding callouts slide in from the left */
  .finding {{ animation: slideInLeft 0.5s ease-out; }}
  @keyframes slideInLeft {{
      from {{ opacity: 0; transform: translateX(-10px); }}
      to   {{ opacity: 1; transform: translateX(0); }}
  }}
</style>
""", unsafe_allow_html=True)


def sec(number, title, note=None):
    """Section header matching the report's numbering, so the app reads as a live
    companion to the write-up rather than a disconnected demo."""
    st.markdown(f"### {number}  {title}")
    if note:
        st.markdown(f'<p class="secnote">{note}</p>', unsafe_allow_html=True)


def show_fig(fig):
    import matplotlib.pyplot as plt
    st.pyplot(fig, width="stretch")
    plt.close(fig)


def signal_bars_html(active, color, flicker_last=False):
    """5-bar phone-signal-strength visual (Telco motif, mirrors the risk-badge
    colour). `active` bars (0-5) are filled with `color`; the rest stay dim.
    flicker_last pulses the top active bar, reading as a call about to drop."""
    active = max(0, min(5, int(round(active))))
    bars = []
    for i in range(1, 6):
        cls = "signal-bar"
        style = ""
        if i <= active:
            cls += " on"
            style = f' style="background:{color}"'
            if flicker_last and i == active:
                cls += " flicker"
        bars.append(f'<div class="{cls}"{style}></div>')
    return f'<div class="signal-bars">{"".join(bars)}</div>'


def mi(name, size=16):
    """Inline Material Symbols glyph — for raw HTML (unsafe_allow_html) contexts,
    where the `:material/x:` shortcode isn't parsed. Native widget labels
    (st.button, st.segmented_control) support the shortcode directly instead."""
    return f'<span class="mi" style="font-size:{size}px">{name}</span>'


def _gradient_stops(css_gradient):
    """Pull the hex colour stops out of a `linear-gradient(...)` string (as
    stored in TIER_INFO) so they can be reused as SVG <stop> colours — avoids
    keeping two parallel colour representations for the same tier."""
    stops = re.findall(r"#[0-9a-fA-F]{6}", css_gradient)
    return stops or ["#2E75B6", "#1F4E79"]


def tower_background_html(mast_gradient, height=850, segments=18, width=240,
                          active_category="Demographics"):
    """Clean lattice-tower backdrop for the category navigation canvas.

    ``active_category`` remains in the signature for callers created by the
    earlier prototype; selection state is now communicated by the real module
    buttons rather than decorative SVG callout lines.
    """
    c1, c2 = _gradient_stops(mast_gradient)[0], _gradient_stops(mast_gradient)[-1]
    bottom_half_w, top_half_w = width * 0.38, width * 0.085
    cx = width / 2
    rows = []
    for i in range(segments + 1):
        t = i / segments
        y = 78 + (height - 88) * t
        half_w = top_half_w + (bottom_half_w - top_half_w) * t
        rows.append((cx - half_w, cx + half_w, y))

    parts = []
    # Three sector antennas make the silhouette immediately recognisable while
    # keeping the background quiet enough for the six real controls around it.
    parts.append(f'<line x1="{width * .33:.1f}" y1="62" x2="{width * .67:.1f}" y2="62" '
                 'stroke="#647080" stroke-width="4" stroke-linecap="round"/>')
    for ax in (width * .38, width * .50, width * .62):
        parts.append(f'<rect x="{ax - 8:.1f}" y="10" width="16" height="50" rx="6" '
                     'fill="#f7f9fc" stroke="#647080" stroke-width="2.5"/>')
        parts.append(f'<line x1="{ax:.1f}" y1="62" x2="{cx:.1f}" y2="78" '
                     'stroke="#8993a0" stroke-width="1.8"/>')
    left_pts = " ".join(f"{lx:.1f},{y:.1f}" for lx, rx, y in rows)
    right_pts = " ".join(f"{rx:.1f},{y:.1f}" for lx, rx, y in rows)
    parts.append(f'<polyline points="{left_pts}" fill="none" stroke="url(#towerGrad)" '
                 'stroke-width="5" stroke-linejoin="round"/>')
    parts.append(f'<polyline points="{right_pts}" fill="none" stroke="url(#towerGrad)" '
                 'stroke-width="5" stroke-linejoin="round"/>')
    for i in range(segments):
        lx0, rx0, y0 = rows[i]
        lx1, rx1, y1 = rows[i + 1]
        parts.append(f'<line x1="{lx0:.1f}" y1="{y0:.1f}" x2="{rx0:.1f}" y2="{y0:.1f}" '
                     'stroke="#7b8795" stroke-width="1.5" opacity=".72"/>')
        parts.append(f'<line x1="{lx0:.1f}" y1="{y0:.1f}" x2="{rx1:.1f}" y2="{y1:.1f}" '
                     f'stroke="{c1}" stroke-width="2" opacity=".78"/>')
        parts.append(f'<line x1="{rx0:.1f}" y1="{y0:.1f}" x2="{lx1:.1f}" y2="{y1:.1f}" '
                     f'stroke="{c2}" stroke-width="2" opacity=".78"/>')
    lx_b, rx_b, y_b = rows[-1]
    parts.append(f'<line x1="{lx_b:.1f}" y1="{y_b:.1f}" x2="{rx_b:.1f}" y2="{y_b:.1f}" '
                 'stroke="#647080" stroke-width="5" stroke-linecap="round"/>')
    body = "".join(parts)

    return f'''
    <svg viewBox="0 0 {width} {height}" width="100%" height="100%"
         preserveAspectRatio="xMidYMin meet" class="tower-svg">
      <defs>
        <linearGradient id="towerGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="{c1}"/>
          <stop offset="100%" stop-color="{c2}"/>
        </linearGradient>
      </defs>
      {body}
    </svg>
    <div class="tower-beacon"></div>
    <div class="tower-base"></div>
    '''


def lerp_color(c1_hex, c2_hex, t):
    """Linear-interpolate between two hex colors by t in [0,1] — drives the Bill
    card's background so it shifts cool-to-hot with the actual Monthly Charges
    value (a top churn predictor), not an arbitrary decorative gradient."""
    t = max(0.0, min(1.0, t))
    c1 = tuple(int(c1_hex[i:i + 2], 16) for i in (1, 3, 5))
    c2 = tuple(int(c2_hex[i:i + 2], 16) for i in (1, 3, 5))
    mixed = tuple(round(c1[j] + (c2[j] - c1[j]) * t) for j in range(3))
    return f"#{mixed[0]:02x}{mixed[1]:02x}{mixed[2]:02x}"

