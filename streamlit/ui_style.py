"""
ui_style.py — visual design system for the Streamlit prototype: colours, icon
maps, the global CSS injector, and small HTML/markdown helper functions used
to build the click-driven cards/tower.

Split out of app.py so the page-flow/business logic there (data wiring,
prediction pipeline, report sections) isn't buried under ~250 lines of CSS —
keeps app.py focused and easier to review.
"""

import streamlit as st

# Report palette (Section colours) — shared with eda_plots.py / evaluation.py
NAVY, BLUE, LIGHT, RED = "#1F4E79", "#2E75B6", "#9DC3E6", "#C00000"
GREEN, AMBER = "#2E8B57", "#E8A317"
MAIN_NAV_STYLE_VERSION = 2

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
  /* Blue is the interaction colour. Red is reserved for churn/risk evidence. */
  button[data-testid="stBaseButton-primary"] {{
      background:{BLUE} !important; border-color:{BLUE} !important; color:#fff !important;
  }}
  button[data-testid="stBaseButton-primary"]:hover {{
      background:{NAVY} !important; border-color:{NAVY} !important;
  }}
  button[data-variant="segmented_control"][aria-checked="true"] {{
      background:color-mix(in srgb, {BLUE} 12%, transparent) !important;
      border-color:{BLUE} !important; color:{BLUE} !important;
  }}
  button[data-variant="segmented_control"][aria-checked="true"] p {{
      color:{BLUE} !important; font-weight:700 !important;
  }}
  button[data-variant="segmented_control"][aria-checked="false"]:hover {{
      border-color:color-mix(in srgb, {BLUE} 55%, transparent) !important;
      color:{BLUE} !important;
  }}
  button[data-variant="segmented_control"]:focus-visible {{
      outline:3px solid color-mix(in srgb, {BLUE} 28%, transparent) !important;
      outline-offset:2px;
  }}

  /* Performance workspace: keep view and metric controls together as one
     compact analysis toolbar instead of two disconnected form rows. */
  div.st-key-performance_controls {{
      background:color-mix(in srgb, var(--background-color) 70%, transparent);
      border:1px solid color-mix(in srgb, {BLUE} 14%, transparent) !important;
      border-radius:18px !important;
      padding:12px 16px 10px !important;
      margin:2px 0 14px;
      box-shadow:0 5px 20px rgba(31,78,121,.04);
  }}
  div.st-key-performance_controls [data-testid="stWidgetLabel"] p {{
      font-weight:650 !important;
      color:color-mix(in srgb, var(--text-color) 78%, transparent) !important;
  }}
  .control-context {{
      min-height:50px;
      display:flex;
      flex-direction:column;
      justify-content:center;
      padding:3px 4px 5px;
      color:color-mix(in srgb, var(--text-color) 68%, transparent);
      font-size:.88rem;
      line-height:1.35;
  }}
  @media (max-width: 720px) {{
    div.st-key-performance_controls {{ padding:10px 12px 8px !important; }}
  }}

  /* Primary workspace navigation only. Keep analysis-level segmented controls
     compact, while making Predict / Data Analysis / Models read as the app's
     centred top-level navigation. */
  div.st-key-main_workspace_view {{
      display:flex !important;
      justify-content:center !important;
      width:100% !important;
      margin:4px auto 24px !important;
  }}
  div.st-key-main_workspace_view [data-testid="stButtonGroup"] {{
      display:flex !important;
      justify-content:center !important;
      width:100% !important;
  }}
  div.st-key-main_workspace_view [role="radiogroup"] {{
      width:fit-content !important;
      max-width:100% !important;
      margin:0 auto !important;
      border-radius:15px !important;
      box-shadow:0 5px 18px rgba(31,78,121,.08);
  }}
  div.st-key-main_workspace_view button[data-variant="segmented_control"] {{
      min-height:54px !important;
      min-width:132px !important;
      padding:0 30px !important;
  }}
  div.st-key-main_workspace_view button[data-variant="segmented_control"] p {{
      font-size:1.06rem !important;
      font-weight:650 !important;
      line-height:1.15 !important;
  }}
  @media (max-width: 720px) {{
    div.st-key-main_workspace_view button[data-variant="segmented_control"] {{
      min-height:48px !important;
      min-width:auto !important;
      padding:0 16px !important;
    }}
    div.st-key-main_workspace_view button[data-variant="segmented_control"] p {{
      font-size:.94rem !important;
    }}
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

  /* Telecom profile heading above the frameless 3D component. */
  div.st-key-customer_network {{ width:100%; }}
  .tower-title {{ text-align:center; font-weight:750; font-size:1.05rem;
    margin:2px 0 4px; letter-spacing:-.1px; }}

  /* Applied synchronously by the non-isolated Predict bridge on click. */
  div.st-key-predict_button button.predict-is-loading {{
    cursor:wait; color:#102f36 !important;
    background:linear-gradient(100deg,#19b8c5,#71dce0,#f2a23b,#ffd078,#19b8c5) !important;
    background-size:260% 100% !important; border-color:transparent !important;
    box-shadow:0 6px 20px rgba(25,184,197,.20) !important;
    animation:predictionFlow 1.65s linear infinite; }}
  div.st-key-predict_button button.predict-is-loading p {{ font-size:0 !important; }}
  div.st-key-predict_button button.predict-is-loading p::before {{
    content:"wifi"; display:inline-block; margin-right:10px;
    font-family:'Material Symbols Rounded'; font-size:20px; vertical-align:-4px;
    animation:wifiPulse 1.05s ease-in-out infinite; }}
  div.st-key-predict_button button.predict-is-loading p::after {{
    content:"Transmitting customer profile…"; font-size:.92rem; font-weight:760; }}
  @keyframes predictionFlow {{ to {{ background-position:260% 0; }} }}
  @keyframes wifiPulse {{ 0%,100% {{ opacity:.38; transform:scale(.88); }}
    50% {{ opacity:1; transform:scale(1.08); text-shadow:0 0 9px rgba(255,255,255,.72); }} }}

  /* Raw HTML injected via unsafe_allow_html doesn't get the `:material/x:` shortcode
     substitution Streamlit applies inside its own widgets — draw icons manually using
     the same "Material Symbols Rounded" webfont Streamlit already loads site-wide. */
  .mi {{ font-family:'Material Symbols Rounded'; font-weight:400; font-style:normal;
    vertical-align:middle; line-height:1; display:inline-block; }}

  /* Glass Findings */
  .finding {{ border-left: 6px solid {BLUE}; background:{BLUE}15; border-radius: 0 20px 20px 0;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    padding:16px 20px; margin:12px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.02); }}
  .finding.blue {{ border-left-color:{BLUE}; background:{BLUE}15; }}
  .finding.green {{ border-left-color:{GREEN}; background:{GREEN}15; }}
  .finding.amber {{ border-left-color:{AMBER}; background:{AMBER}15; }}
  .finding.red {{ border-left-color:{RED}; background:{RED}15; }}

  .secnote {{ color: var(--text-color); opacity: 0.6; font-size:.95rem; margin:-4px 0 12px; }}

  /* Tab bar — compact website navigation rather than a full-width report index.
     Every level stays content-width and centred; long bars scroll on narrow
     screens instead of stretching an empty track across the page. */
  .react-aria-SelectionIndicator {{ display:none !important; }}
  [role="tablist"] {{
    gap:0 !important; background: rgba(120,130,150,0.10); border-radius:12px;
    padding:4px; border-bottom:none !important; width:fit-content !important;
    max-width:100%; margin-left:auto !important; margin-right:auto !important;
    overflow-x:auto; scrollbar-width:thin; }}
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
    """Render a web-first section header while retaining report mapping in code."""
    st.markdown(f"### {title}")
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


_PREDICT_BRIDGE_JS = r"""
export default function(component) {
  const {parentElement}=component;
  parentElement.style.display='none';
  const doc=parentElement.ownerDocument;
  let button=null, fallbackTimer=null, disposed=false;
  const clear=()=>{
    button?.classList.remove('predict-is-loading');
    button?.removeAttribute('aria-busy');
    if (fallbackTimer) { clearTimeout(fallbackTimer); fallbackTimer=null; }
  };
  const activate=()=>{
    button?.classList.add('predict-is-loading');
    button?.setAttribute('aria-busy','true');
    if (fallbackTimer) clearTimeout(fallbackTimer);
    fallbackTimer=setTimeout(clear,15000);
  };
  const onClick=()=>activate();
  const bindButton=()=>{
    const current=doc.querySelector('div.st-key-predict_button button');
    if (!current || current===button) return Boolean(current);
    button?.removeEventListener('click',onClick,true);
    button=current;
    button.addEventListener('click',onClick,true);
    return true;
  };
  const start=()=>{
    if (disposed) return;
    if (!bindButton()) { requestAnimationFrame(start); return; }
  };
  start();
  return ()=>{
    disposed=true;
    button?.removeEventListener('click',onClick,true);
  };
}
"""

_predict_button_bridge = st.components.v2.component(
    "predict_button_bridge_v3",
    html='<span aria-hidden="true"></span>',
    js=_PREDICT_BRIDGE_JS,
    isolate_styles=False,
)


def attach_predict_button_loader():
    """Attach the immediate browser-side loader to the native Predict button."""
    _predict_button_bridge(key="predict_button_bridge_v3", height=1)
