"""Interactive, telecom-themed visuals for a single churn prediction."""

from html import escape

import plotly.graph_objects as go


LOW = "#5F8F78"
AMBER = "#E8A317"
RED = "#C84B45"
INK = "#27313D"
MUTED = "#7B8491"
PREDICTION_VISUALS_VERSION = 3


def _risk_color(value: float) -> str:
    return LOW if value < 40 else (AMBER if value < 70 else RED)


def hero_html(probability: float, risk_label: str, risk_color: str,
              model_name: str, base_rate: float = 26.54,
              animation_token: int | str = "") -> str:
    """Border-light result hero; all values are display-only."""
    probability = max(0.0, min(100.0, float(probability)))
    signal_bars = max(0, min(5, round((100 - probability) / 20)))
    bars = "".join(
        f'<i style="height:{7 + index * 5}px;opacity:{1 if index <= signal_bars else .16}"></i>'
        for index in range(1, 6)
    )
    ribbon_specs = (
        (7, 92, -86, 128, 42, -18, "#5F8F78"),
        (8, 138, -58, 205, 74, 22, "#E8A317"),
        (9, 184, -78, 252, 54, -36, "#D9E0E5"),
        (6, 112, -42, 172, 106, 58, "#8EAAA0"),
        (10, 218, -66, 292, 92, -8, "#F2C45C"),
        (7, 154, -94, 224, 24, 34, "#EEF1F3"),
        (8, 72, -54, 126, 82, -48, "#6E9C87"),
        (9, 196, -38, 264, 118, 16, "#D8A43A"),
        (93, -92, -86, -128, 42, 18, "#5F8F78"),
        (92, -138, -58, -205, 74, -22, "#E8A317"),
        (91, -184, -78, -252, 54, 36, "#D9E0E5"),
        (94, -112, -42, -172, 106, -58, "#8EAAA0"),
        (90, -218, -66, -292, 92, 8, "#F2C45C"),
        (93, -154, -94, -224, 24, -34, "#EEF1F3"),
        (92, -72, -54, -126, 82, 48, "#6E9C87"),
        (91, -196, -38, -264, 118, -16, "#D8A43A"),
    )
    ribbons = ""
    if risk_label == "LOW RISK":
        pieces = "".join(
            '<i style="--x:{}%;--x1:{}px;--y1:{}px;--x2:{}px;--y2:{}px;'
            '--turn:{}deg;--ribbon:{};--delay:{}ms"></i>'.format(
                x, x1, y1, x2, y2, turn, color, index * 18
            )
            for index, (x, x1, y1, x2, y2, turn, color) in enumerate(ribbon_specs)
        )
        ribbons = f'<div class="result-ribbons" aria-hidden="true">{pieces}</div>'
    return f"""
<style>
  .prediction-hero {{
    position:relative;
    display:grid; grid-template-columns:minmax(210px,.8fr) minmax(280px,1.2fr);
    align-items:center; gap:28px; padding:24px 12px 22px; margin:5px 0 2px;
    border-top:1px solid rgba(104,115,128,.20);
    border-bottom:1px solid rgba(104,115,128,.20);
  }}
  .prediction-reading {{ display:flex; align-items:center; gap:18px; }}
  .prediction-orbit {{
    width:104px; height:104px; flex:0 0 104px; border-radius:50%; display:grid;
    place-items:center; color:{risk_color}; position:relative;
    background:radial-gradient(circle,rgba(255,255,255,.96) 35%,transparent 37%),
      conic-gradient({risk_color} {probability:.1f}%,rgba(126,137,149,.13) 0);
    box-shadow:0 0 28px color-mix(in srgb,{risk_color} 22%,transparent);
  }}
  .prediction-orbit::after {{ content:'cell_tower'; font-family:'Material Symbols Rounded';
    font-size:34px; color:{risk_color}; }}
  .prediction-value {{ font-size:clamp(2rem,3.3vw,3.3rem); line-height:.95;
    font-weight:820; letter-spacing:-2px; color:#27313d; }}
  .prediction-kicker {{ color:{risk_color}; font-size:.76rem; letter-spacing:.13em;
    font-weight:820; text-transform:uppercase; margin-bottom:8px; }}
  .prediction-copy h3 {{ margin:0 0 7px; font-size:1.22rem; color:#27313d; }}
  .prediction-copy p {{ margin:0; color:#78828f; line-height:1.55; }}
  .prediction-meta {{ display:flex; gap:9px; flex-wrap:wrap; margin-top:14px; }}
  .prediction-meta span {{ border-left:2px solid {risk_color}; padding:3px 10px;
    color:#535e6b; font-size:.82rem; background:rgba(255,255,255,.42); }}
  .prediction-signal {{ height:34px; display:flex; align-items:flex-end; gap:4px; margin-top:13px; }}
  .prediction-signal i {{ display:block; width:7px; background:{risk_color};
    border-radius:2px 2px 0 0; }}
  .result-ribbons {{ position:fixed; inset:0; z-index:999999; overflow:hidden; pointer-events:none; }}
  .result-ribbons i {{
    position:absolute; left:var(--x); top:47%; width:5px; height:17px; border-radius:2px;
    background:var(--ribbon); opacity:0; transform-origin:center;
    animation:result-ribbon-burst 1.35s cubic-bezier(.18,.72,.24,1) var(--delay) both;
  }}
  @keyframes result-ribbon-burst {{
    0% {{ opacity:0; transform:translate(0,0) rotate(var(--turn)); }}
    12% {{ opacity:1; }}
    52% {{ opacity:.95; transform:translate(var(--x1),var(--y1)) rotate(calc(var(--turn) + 170deg)); }}
    100% {{ opacity:0; transform:translate(var(--x2),var(--y2)) rotate(calc(var(--turn) + 360deg)); }}
  }}
  @media(max-width:760px) {{ .prediction-hero {{ grid-template-columns:1fr; }} }}
  @media(prefers-reduced-motion:reduce) {{ .result-ribbons {{ display:none; }} }}
</style>
<section class="prediction-hero" aria-label="Prediction result" data-animation="{escape(str(animation_token))}">
  {ribbons}
  <div class="prediction-reading">
    <div class="prediction-orbit" aria-hidden="true"></div>
    <div><div class="prediction-kicker">{escape(risk_label)}</div>
      <div class="prediction-value">{probability:.1f}%</div></div>
  </div>
  <div class="prediction-copy">
    <h3>Customer churn signal</h3>
    <p>The selected model places this customer in the <b>{escape(risk_label.lower())}</b>
      band. The visuals below locate the score against decision thresholds and the
      other deployed models.</p>
    <div class="prediction-meta"><span>{escape(model_name)}</span>
      <span>Dataset base rate&nbsp; {base_rate:.1f}%</span>
      <span>{signal_bars}/5 retention signal</span></div>
    <div class="prediction-signal" aria-label="{signal_bars} of 5 retention signal bars">{bars}</div>
  </div>
</section>"""


def signal_spectrum(probability: float, base_rate: float = 26.54) -> go.Figure:
    """Compact signal route with explicit thresholds and base-rate context."""
    probability = float(probability)
    fig = go.Figure()

    # A single telecom-style route is easier to scan than three large background
    # rectangles. Small endpoint markers visually round Plotly's line caps.
    bands = (
        (0, 40, LOW, "Stable"),
        (40, 70, AMBER, "At risk"),
        (70, 100, RED, "Critical"),
    )
    for start, end, color, label in bands:
        fig.add_trace(go.Scatter(
            x=[start, end], y=[0, 0], mode="lines", showlegend=False,
            line=dict(color=color, width=15),
            hovertemplate=f"{label}: {start}–{end}%<extra></extra>",
        ))
        fig.add_annotation(
            x=(start + end) / 2, y=.31, text=f"<b>{label.upper()}</b>",
            showarrow=False, font=dict(size=11, color=color),
        )
    fig.add_trace(go.Scatter(
        x=[0, 40, 40, 70, 70, 100], y=[0] * 6,
        mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(size=15, color=[LOW, LOW, AMBER, AMBER, RED, RED]),
    ))

    for threshold, label in ((40, "40%"), (70, "70%")):
        fig.add_shape(
            type="line", x0=threshold, x1=threshold, y0=-.17, y1=.18,
            line=dict(color="rgba(74,84,96,.50)", width=2, dash="dot"),
        )
        fig.add_annotation(
            x=threshold, y=-.29, text=f"<b>{label}</b><br>threshold",
            showarrow=False, align="center", font=dict(size=10, color=MUTED),
        )

    fig.add_trace(go.Scatter(
        x=[base_rate], y=[0], mode="markers+text", name="Dataset base rate",
        marker=dict(symbol="line-ns", size=25, color=MUTED,
                    line=dict(width=3, color=MUTED)),
        text=[f"Base {base_rate:.1f}%"], textposition="bottom center",
        textfont=dict(size=11, color=MUTED),
        hovertemplate="Dataset churn base rate: %{x:.1f}%<extra></extra>",
    ))

    # Layered translucent markers create a restrained signal glow without turning
    # the result into a conventional speedometer gauge.
    prediction_color = _risk_color(probability)
    for size, opacity in ((42, .10), (31, .18)):
        fig.add_trace(go.Scatter(
            x=[probability], y=[0], mode="markers", showlegend=False,
            marker=dict(symbol="diamond", size=size, color=prediction_color,
                        opacity=opacity), hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=[probability], y=[0], mode="markers", name="This customer",
        marker=dict(symbol="diamond", size=18, color=prediction_color,
                    line=dict(width=2, color="white")),
        hovertemplate="Predicted churn probability: %{x:.1f}%<extra></extra>",
    ))
    delta = probability - base_rate
    fig.add_annotation(
        x=probability, y=.18,
        text=(f"<b>THIS CUSTOMER&nbsp; {probability:.1f}%</b>"
              f"<br><span style='font-size:10px'>{delta:+.1f} pp vs base rate</span>"),
        showarrow=True, arrowhead=0, arrowwidth=1.5, arrowcolor=prediction_color,
        ax=0, ay=-53, bgcolor="rgba(255,255,255,.96)",
        bordercolor=prediction_color, borderwidth=1, borderpad=7,
        font=dict(size=12, color=INK), align="center",
    )
    fig.update_layout(
        height=225, template="plotly_white", showlegend=False,
        margin=dict(l=26, r=26, t=67, b=51),
        font=dict(family="Segoe UI, sans-serif", color=INK),
        hoverlabel=dict(bgcolor="white", font_size=12),
        xaxis=dict(range=[-1, 101], tickvals=[0, 20, 40, 60, 70, 80, 100],
                   ticksuffix="%", fixedrange=True, title=None, showgrid=False,
                   zeroline=False, tickfont=dict(size=10, color=MUTED)),
        yaxis=dict(range=[-.43, .56], visible=False, fixedrange=True),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        transition=dict(duration=450, easing="cubic-in-out"),
    )
    return fig


def model_consensus(all_probs: dict, selected_model: str, metrics: dict) -> go.Figure:
    """Four-model dot plot with decision bands and evaluation context on hover."""
    names = list(all_probs)
    values = [float(all_probs[name]) for name in names]
    colors = [_risk_color(value) for value in values]
    symbols = ["star" if name == selected_model else "circle" for name in names]
    sizes = [18 if name == selected_model else 13 for name in names]
    custom = [[metrics[name]["AUC"], metrics[name]["Recall"]] for name in names]

    fig = go.Figure()
    for start, end, color in (
        (0, 40, "rgba(95,143,120,.10)"),
        (40, 70, "rgba(232,163,23,.11)"),
        (70, 100, "rgba(200,75,69,.10)"),
    ):
        fig.add_vrect(x0=start, x1=end, fillcolor=color, line_width=0)
    for index, value in enumerate(values):
        fig.add_shape(type="line", x0=0, x1=value, y0=index, y1=index,
                      line=dict(color="rgba(116,127,139,.28)", width=2))
    fig.add_trace(go.Scatter(
        x=values, y=names, mode="markers+text", text=[f"{value:.1f}%" for value in values],
        textposition="middle right", cliponaxis=False,
        marker=dict(color=colors, symbol=symbols, size=sizes,
                    line=dict(color="white", width=2)), customdata=custom,
        hovertemplate=("<b>%{y}</b><br>Churn probability: %{x:.1f}%"
                       "<br>Test AUC: %{customdata[0]:.3f}"
                       "<br>Test recall: %{customdata[1]:.1f}%<extra></extra>"),
    ))
    spread = max(values) - min(values)
    fig.update_layout(
        height=285, template="plotly_white", showlegend=False,
        margin=dict(l=12, r=58, t=18, b=38),
        font=dict(family="Segoe UI, sans-serif", color=INK),
        hoverlabel=dict(bgcolor="white", font_size=12),
        xaxis=dict(range=[0, 106], ticksuffix="%", fixedrange=True,
                   title=f"Prediction range · {spread:.1f} percentage-point spread",
                   gridcolor="rgba(120,130,140,.12)"),
        yaxis=dict(autorange="reversed", fixedrange=True),
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    return fig


_IMPORTANCE_GROUPS = (
    ("Gender", ("gender",), "gender"),
    ("Senior citizen", ("SeniorCitizen",), "senior_citizen"),
    ("Partner", ("Partner",), "partner"),
    ("Dependents", ("Dependents",), "dependents"),
    ("Tenure", ("tenure",), "tenure"),
    ("Phone service", ("PhoneService",), "phone_service"),
    ("Multiple lines", ("MultipleLines",), "multiple_lines"),
    ("Online security", ("OnlineSecurity",), "OnlineSecurity"),
    ("Online backup", ("OnlineBackup",), "OnlineBackup"),
    ("Device protection", ("DeviceProtection",), "DeviceProtection"),
    ("Tech support", ("TechSupport",), "TechSupport"),
    ("Streaming TV", ("StreamingTV",), "StreamingTV"),
    ("Streaming movies", ("StreamingMovies",), "StreamingMovies"),
    ("Paperless billing", ("PaperlessBilling",), "paperless_billing"),
    ("Monthly charges", ("MonthlyCharges",), "monthly_charges"),
    ("Total charges", ("TotalCharges",), "total_charges"),
    ("Contract type", ("ContractRiskScore",), "contract"),
    ("Charges / tenure", ("ChargesToTenureRatio",), "charges_tenure"),
    ("Internet service", ("InternetService_Fiber optic", "InternetService_No"),
     "internet_service"),
    ("Payment method", ("PaymentMethod_Credit card (automatic)",
                        "PaymentMethod_Electronic check",
                        "PaymentMethod_Mailed check"), "payment_method"),
)


def _customer_value(profile: dict, value_key: str) -> str:
    if value_key in profile:
        value = profile[value_key]
    else:
        value = profile.get("addons", {}).get(value_key, "Not available")
    if value_key == "tenure":
        return f"{int(value)} months"
    if value_key == "monthly_charges":
        return f"${float(value):,.1f}/mo"
    if value_key == "total_charges":
        return f"${float(value):,.2f}"
    if value_key == "charges_tenure":
        tenure = max(float(profile.get("tenure", 0)), 1.0)
        return f"${float(profile.get('total_charges', 0)) / tenure:,.1f}/mo"
    return str(value)


def importance_skyline(raw_importances: dict, profile: dict,
                       model_name: str, top_n: int = 6) -> go.Figure:
    """Vertical, business-level view of model-wide relative importance.

    One-hot columns belonging to the same user-facing field are summed before
    ranking. This remains a global model summary, not a local contribution chart.
    """
    grouped = []
    for label, columns, value_key in _IMPORTANCE_GROUPS:
        score = sum(abs(float(raw_importances.get(column, 0.0))) for column in columns)
        grouped.append({
            "label": label,
            "columns": ", ".join(columns),
            "score": score,
            "customer": _customer_value(profile, value_key),
        })
    total = sum(item["score"] for item in grouped) or 1.0
    for item in grouped:
        item["relative"] = item["score"] / total * 100
    ranked = sorted(grouped, key=lambda item: item["relative"], reverse=True)[:top_n]

    labels = [item["label"] for item in ranked]
    values = [item["relative"] for item in ranked]
    customers = [item["customer"] for item in ranked]
    columns = [item["columns"] for item in ranked]
    tick_labels = [
        f"{label.replace(' ', '<br>', 1)}<br><span style='font-size:10px'>"
        f"Customer: {escape(customer)}</span>"
        for label, customer in zip(labels, customers)
    ]
    colors = [AMBER if index < 2 else "#89949F" for index in range(len(ranked))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=labels, y=values, width=.48, marker=dict(color=colors),
        text=[f"{value:.1f}%" for value in values], textposition="outside",
        textfont=dict(size=12, color=INK), cliponaxis=False,
        customdata=[[customer, column] for customer, column in zip(customers, columns)],
        hovertemplate=(f"<b>%{{x}}</b><br>Relative importance: %{{y:.1f}}%"
                       "<br>This customer: %{customdata[0]}"
                       "<br>Model input(s): %{customdata[1]}"
                       f"<br>Selected model: {escape(model_name)}<extra></extra>"),
    ))
    # Small caps make each column read as a signal mast rather than a generic bar.
    fig.add_trace(go.Scatter(
        x=labels, y=values, mode="markers", showlegend=False, hoverinfo="skip",
        marker=dict(symbol="diamond", size=10, color=colors,
                    line=dict(color="white", width=1.5)),
    ))
    ceiling = max(values) * 1.30 if values else 1
    fig.update_layout(
        height=360, template="plotly_white", showlegend=False, bargap=.43,
        margin=dict(l=44, r=22, t=28, b=92),
        font=dict(family="Segoe UI, sans-serif", color=INK),
        hoverlabel=dict(bgcolor="white", font_size=12),
        xaxis=dict(tickmode="array", tickvals=labels, ticktext=tick_labels,
                   fixedrange=True, tickfont=dict(size=11, color=INK),
                   showgrid=False, title=None),
        yaxis=dict(range=[0, ceiling], ticksuffix="%", fixedrange=True,
                   title="Relative model importance",
                   gridcolor="rgba(120,130,140,.13)", zeroline=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        transition=dict(duration=500, easing="cubic-in-out"),
    )
    return fig
