import sys
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.env import FloodAllocEnv
from src.evaluate import build_policies, run
from src.resources import units_from_records
from src.approval import plan_from_log, validate, log_decision
from src.briefing import build_facts, llm_briefing
from src.impact import upsert_zone

st.set_page_config(page_title="Flood Response Planner", layout="wide")

st.markdown("""<style>
h1, h2, h3, h4 {font-family: Georgia, 'Times New Roman', serif !important; font-weight: 500 !important; color: #2b2620;}
.block-container {padding-top: 2rem; max-width: 1200px;}
.lede {font-size: 1.08rem; color: #5a5148; max-width: 780px; line-height: 1.55; margin-bottom: 1rem;}
.card {background: #fffdf9; border: 1px solid #e6dccd; border-radius: 8px; padding: 14px 16px; height: 100%;}
.card h5 {margin: 0 0 6px 0; font-family: Georgia, serif; font-weight: 500; font-size: 1.02rem; color: #2b2620;}
.card p {margin: 0; font-size: 0.9rem; color: #5a5148; line-height: 1.45;}
.card .tag {display: inline-block; margin-top: 8px; font-size: 0.78rem; color: #9a4a2a;}
.flow {display: flex; gap: 10px; flex-wrap: wrap; align-items: stretch; margin: 8px 0 4px 0;}
.flow .card {flex: 1 1 150px;}
.flow .arrow {align-self: center; color: #9a8f80;}
[data-testid="stMetric"] {background: #fffdf9; border: 1px solid #e6dccd; border-radius: 8px; padding: 12px 14px;}
</style>""", unsafe_allow_html=True)

METRICS = {
    "coverage": ("Need met", "Share of all estimated resource need that the plan covers.", "Higher is better"),
    "severity_weighted_coverage": ("Priority-weighted need met", "Same idea, but zones with more exposed people, poorer roads and farther hospitals count more. The main score.", "Higher is better"),
    "worst_zone_coverage": ("Worst-served zone", "Need met in the least-served zone. A fairness check: it shows whether any area was left behind.", "Higher is better"),
    "mean_travel_km": ("Average travel (km)", "Mean distance resources travel from depot to zone. A proxy for response time.", "Lower is better"),
    "wasted_units": ("Wasted units", "Units sent to a zone that did not need that resource type.", "Lower is better"),
}
STRATEGIES = {
    "ppo": "Reinforcement learning (PPO)",
    "severity_first": "Most severe zone first",
    "nearest_first": "Nearest zone first",
    "largest_need": "Largest remaining need first",
    "random": "Random (reference only)",
}


def card(title, body, tag=""):
    t = f'<div class="tag">{tag}</div>' if tag else ""
    return f'<div class="card"><h5>{title}</h5><p>{body}</p>{t}</div>'


@st.cache_data(show_spinner="Running flood model")
def _predict(data: bytes):
    from src.infer import predict_flood_mask
    return predict_flood_mask(Image.open(io.BytesIO(data)).convert("RGB"))


# ---------- header ----------
st.title("Flood Response Planner")
st.markdown('<p class="lede">Decision support for emergency coordinators. It reads flooded areas from '
            'satellite or aerial imagery, estimates how many boats, medical teams and trucks each area needs, '
            'and recommends where to send limited resources first. You review, edit and approve every plan.</p>',
            unsafe_allow_html=True)

# ---------- sidebar inputs ----------
zones = pd.read_csv("data/zones.csv")
zones["exposed"] = (zones.population * zones.flood_frac).astype(int)

st.sidebar.subheader("Resources available")
res = st.sidebar.data_editor(pd.DataFrame(json.load(open("data/resources.json"))),
                             num_rows="dynamic", use_container_width=True)
st.sidebar.caption("Type must be boat, medical or truck. x and y are the depot position in km.")
st.sidebar.subheader("Strategy")
name = st.sidebar.selectbox("Allocation strategy", list(STRATEGIES), format_func=STRATEGIES.get,
                            label_visibility="collapsed")
st.sidebar.caption("Population, road and hospital values are simulated or entered by hand in this prototype.")

res = res.dropna()
try:
    units = units_from_records(res.to_dict("records"))
except Exception as ex:
    st.error(f"Check the resource table: {ex}")
    st.stop()
if not units:
    st.warning("Add at least one resource in the sidebar.")
    st.stop()

# ---------- compute ----------
pols = build_policies(len(zones))
if "ppo" not in pols:
    st.sidebar.caption("PPO model not found. Train it once: python -m src.train_rl")
if name not in pols:
    st.warning("PPO needs training for the current number of zones, showing 'Most severe zone first' instead.")
    name = "severity_first"
env = FloodAllocEnv(zones, units, randomize=False)
metrics = run(env, pols[name], seed=0)
plan = plan_from_log(env, zones)
nz = len(zones)
zones["need_met"] = (1 - env.rem[:nz].sum(1) / env.need[:nz].sum(1).clip(min=1)) * 100

t_about, t_sit, t_plan, t_brief, t_res, t_img = st.tabs(
    ["About", "Situation", "Plan", "Briefing", "Results", "Add from image"])

# ---------- about ----------
with t_about:
    st.subheader("What this platform does")
    steps = ["Detect flooding|A segmentation model (U-Net) marks flooded pixels in imagery.",
             "Describe each zone|Flooded share, people exposed, road access and hospital distance.",
             "Estimate need|How many boats, medical teams and trucks each zone requires.",
             "Allocate|A reinforcement learning agent or a simple rule assigns each unit to a zone.",
             "Human decision|You approve, edit or reject. Every decision is logged."]
    html = '<div class="flow">'
    for i, s in enumerate(steps):
        a, b = s.split("|")
        html += card(f"{i+1}. {a}", b)
        if i < len(steps) - 1:
            html += '<div class="arrow">&rarr;</div>'
    st.markdown(html + "</div>", unsafe_allow_html=True)

    st.subheader("How to use it")
    cols = st.columns(4)
    howto = [("1. Enter resources", "In the sidebar, list what you have and where it is stored."),
             ("2. Check the situation", "Upload imagery in 'Add from image', or use the Situation tab to see flooded zones."),
             ("3. Review the plan", "Pick a strategy in the sidebar. Edit unit counts in the Plan tab if you disagree."),
             ("4. Decide", "Approve or reject. Then generate a briefing to share.")]
    for c, (t, b) in zip(cols, howto):
        c.markdown(card(t, b), unsafe_allow_html=True)

    st.subheader("What the metrics mean")
    cols = st.columns(len(METRICS))
    for c, (label, desc, direction) in zip(cols, METRICS.values()):
        c.markdown(card(label, desc, direction), unsafe_allow_html=True)

    st.caption("Limits: the flooded share comes from the model, but population, road and hospital values are "
               "simulated or typed in by hand because the imagery has no geographic data. Results show the "
               "method works, not real flood outcomes.")

# ---------- situation ----------
with t_sit:
    k = st.columns(4)
    k[0].metric("Flood zones", len(zones))
    k[1].metric("People exposed", f"{zones.exposed.sum():,}", help="Population times flooded share, assuming people are spread evenly.")
    k[2].metric("Units available", int(res.qty.astype(float).sum()))
    k[3].metric("Units needed", int(env.need.sum()), help="Estimated total need across all zones and resource types.")

    fig = px.scatter(zones, x="x", y="y", size="exposed", color="flood_frac", text="zone_id",
                     color_continuous_scale=[[0, "#e9e1d3"], [1, "#1f4e79"]],
                     labels={"x": "East (km)", "y": "North (km)", "flood_frac": "Flooded share"})
    fig.update_traces(textposition="top center")
    fig.add_scatter(x=res.x, y=res.y, mode="markers+text", text=res.type, textposition="bottom center",
                    marker=dict(symbol="square", size=12, color="#9a4a2a"), name="Depot")
    fig.update_layout(height=460, plot_bgcolor="#fffdf9", paper_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.08))
    fig.update_xaxes(gridcolor="#eee6d8")
    fig.update_yaxes(gridcolor="#eee6d8")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Circle size is people exposed; darker means more flooded. Squares are resource depots.")

    show = zones.assign(flood_pct=zones.flood_frac * 100, road_pct=zones.road_access * 100)
    st.dataframe(show[["zone_id", "flood_pct", "exposed", "population", "road_pct", "hospital_km"]],
                 use_container_width=True, hide_index=True,
                 column_config={
                     "zone_id": "Zone",
                     "flood_pct": st.column_config.ProgressColumn("Flooded", min_value=0, max_value=100, format="%.0f%%"),
                     "exposed": st.column_config.NumberColumn("People exposed", format="%d"),
                     "population": st.column_config.NumberColumn("Population", format="%d"),
                     "road_pct": st.column_config.ProgressColumn("Road access", min_value=0, max_value=100, format="%.0f%%"),
                     "hospital_km": st.column_config.NumberColumn("Hospital (km)", format="%.1f"),
                 })

# ---------- plan ----------
with t_plan:
    st.caption(f"Strategy: {STRATEGIES[name]}. Scores below describe the recommended plan before any edits.")
    cols = st.columns(len(METRICS))
    for c, (key, (label, desc, direction)) in zip(cols, METRICS.items()):
        v = metrics[key]
        txt = f"{v:.0%}" if key in ("coverage", "severity_weighted_coverage", "worst_zone_coverage") else (
            f"{v:.1f}" if key == "mean_travel_km" else f"{int(v)}")
        c.metric(label, txt, help=f"{desc} {direction}.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### Recommended allocation")
        edited = st.data_editor(
            plan, disabled=["zone_id", "type", "avg_km"], use_container_width=True, hide_index=True,
            key=f"plan_{name}_{len(zones)}_{int(res.qty.astype(float).sum())}",
            column_config={"zone_id": "Zone", "type": "Resource",
                           "qty": st.column_config.NumberColumn("Units", min_value=0, step=1),
                           "avg_km": st.column_config.NumberColumn("Avg km", format="%.1f")})
    with right:
        st.markdown("#### Need met by zone")
        fig2 = px.bar(zones.sort_values("need_met"), x="need_met", y="zone_id", orientation="h",
                      color_discrete_sequence=["#1f4e79"], labels={"need_met": "Need met (%)", "zone_id": ""})
        fig2.update_layout(height=320, plot_bgcolor="#fffdf9", paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=10, r=10, t=10, b=10), xaxis_range=[0, 100])
        st.plotly_chart(fig2, use_container_width=True)

    problems = validate(edited, res.assign(qty=res.qty.astype(float)).groupby("type").qty.sum().to_dict())
    for p in problems:
        st.error(f"Over capacity: {p}")
    note = st.text_input("Coordinator note (optional)")
    b1, b2, _ = st.columns([1, 1, 4])
    if b1.button("Approve plan", type="primary", disabled=bool(problems)):
        log_decision("approved" if edited.equals(plan) else "overridden", edited, note)
        st.success("Decision logged to results/decisions.jsonl")
    if b2.button("Reject plan"):
        log_decision("rejected", edited, note)
        st.warning("Rejection logged to results/decisions.jsonl")

# ---------- briefing ----------
with t_brief:
    st.markdown("A plain-language summary of the current situation and the plan above, "
                "including any edits you made.")
    if st.button("Generate briefing", type="primary"):
        with st.spinner("Writing briefing"):
            text = llm_briefing(build_facts(zones, edited, metrics))
        st.markdown(card("Situation briefing", text.replace("\n", "<br>")), unsafe_allow_html=True)

# ---------- results ----------
with t_res:
    p = Path("results/rl_vs_heuristics.csv")
    if not p.exists():
        st.info("Run python -m src.evaluate to compare strategies.")
    else:
        df = pd.read_csv(p, index_col=0)
        df.index = [STRATEGIES.get(i, i) for i in df.index]
        st.markdown("Each strategy was tested on 200 randomized flood scenarios. Values are averages.")
        key = st.selectbox("Metric", list(METRICS), format_func=lambda k: METRICS[k][0])
        label, desc, direction = METRICS[key]
        st.caption(f"{desc} {direction}.")
        fig3 = px.bar(df[key].sort_values(ascending=(direction == "Lower is better")),
                      color_discrete_sequence=["#9a4a2a"], labels={"value": label, "index": ""})
        fig3.update_layout(height=340, plot_bgcolor="#fffdf9", paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(df.rename(columns={k: v[0] for k, v in METRICS.items()}), use_container_width=True)

# ---------- add from image ----------
with t_img:
    if st.session_state.get("saved"):
        st.success(f"Zone {st.session_state.pop('saved')} saved. The plan has been recalculated.")
    st.markdown("Upload satellite or aerial imagery. The U-Net marks flooded pixels, and the flooded share "
                "becomes a new zone, or updates an existing one.")
    up = st.file_uploader("Flood imagery", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up).convert("RGB")
        try:
            mask = _predict(up.getvalue())
        except Exception as ex:
            mask = None
            st.error(f"Could not run the model: {ex}")
        if mask is not None:
            ov = np.asarray(img).copy()
            ov[mask] = (0.5 * ov[mask] + 0.5 * np.array([31, 78, 121])).astype(np.uint8)
            c1, c2 = st.columns(2)
            c1.image(img, caption="Input image", use_container_width=True)
            c2.image(ov, caption="Predicted flood (blue)", use_container_width=True)
            ff = float(mask.mean())
            st.metric("Flooded share of image", f"{ff:.0%}")

            target = st.selectbox("Add as", ["New zone"] + list(zones.zone_id))
            base = zones[zones.zone_id == target].iloc[0] if target != "New zone" else None
            d = lambda k, v: float(base[k]) if base is not None else v
            if base is None:
                n = len(zones) + 1
                while f"Z{n}" in set(zones.zone_id):
                    n += 1
                zid = f"Z{n}"
            else:
                zid = target
            st.caption("Population, roads and hospital distance cannot come from the image. "
                       "Enter your best estimates or leave the defaults.")
            a, b, c = st.columns(3)
            pop = a.number_input("Population", 0, 10_000_000, int(d("population", 10000)), step=500, key=f"pop_{zid}")
            road = b.slider("Road access", 0.0, 1.0, d("road_access", 0.7), key=f"road_{zid}")
            hosp = c.number_input("Hospital distance (km)", 0.0, 200.0, d("hospital_km", 10.0), key=f"hosp_{zid}")
            e, f_ = st.columns(2)
            x = e.number_input("East position (km)", 0.0, 500.0, d("x", 25.0), key=f"x_{zid}")
            y = f_.number_input("North position (km)", 0.0, 500.0, d("y", 25.0), key=f"y_{zid}")
            if st.button(f"Save as {zid}", type="primary"):
                new = upsert_zone(pd.read_csv("data/zones.csv"), zid, ff, source=up.name,
                                  population=int(pop), road_access=float(road),
                                  hospital_km=float(hosp), x=float(x), y=float(y))
                new.to_csv("data/zones.csv", index=False)
                st.session_state["saved"] = zid
                st.rerun()
