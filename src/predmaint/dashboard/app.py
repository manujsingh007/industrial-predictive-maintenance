"""Streamlit fleet dashboard.

Run: ``uv run --group dashboard streamlit run src/predmaint/dashboard/app.py``

It talks **only** to the RUL API (``API_BASE_URL``, default http://127.0.0.1:8000), never
to the database or data files. In demo mode (``PREDMAINT_DEMO_SNAPSHOT`` as an environment
variable or Streamlit secret) it shows a static snapshot of the API's answers instead, with no
network access; see ``predmaint.dashboard.source``.

**No true RUL anywhere.** The dashboard shows only what a real operator would have: stored
readings and the model's predictions. In reality the true RUL is unknown until the engine fails.
The replayed engines come from the test set, and showing their labels would place test labels
next to the predictions, inviting thresholds to be tuned by eye on the test set. The API never
receives labels, so the dashboard cannot show them.
"""

import altair as alt
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from predmaint import config
from predmaint.dashboard.logic import (
    DEFAULT_THRESHOLDS,
    DRIFT_COLOURS,
    DRIFT_LABELS,
    STATUS_COLOURS,
    STATUS_LABELS,
    RiskThresholds,
    drift_table,
    fleet_kpis,
    fleet_table,
    histogram_frame,
    prediction_history,
)
from predmaint.dashboard.source import KEY_SENSORS, DataSourceError, make_source

REPO_URL = "https://github.com/manujsingh007/industrial-predictive-maintenance"
BLUE = "#2a78d6"

st.set_page_config(page_title="RUL fleet dashboard", page_icon="🛠", layout="wide")


def _snapshot_secret() -> str | None:
    try:
        return st.secrets.get("PREDMAINT_DEMO_SNAPSHOT")
    except StreamlitSecretNotFoundError:  # no secrets file: the normal case
        return None


try:
    client = make_source(config.demo_snapshot_dir(_snapshot_secret()))
except DataSourceError as exc:
    st.error(str(exc))
    st.stop()

# --- sidebar: connection, thresholds, refresh ------------------------------------------------
with st.sidebar:
    st.header("Settings")
    st.caption(client.label)
    critical = st.number_input(
        "Critical if lower bound <",
        min_value=0.0,
        value=DEFAULT_THRESHOLDS.critical,
        step=5.0,
        key="critical",
    )
    warning = st.number_input(
        "Warning if lower bound <",
        min_value=0.0,
        value=DEFAULT_THRESHOLDS.warning,
        step=5.0,
        key="warning",
    )
    if client.snapshot is None:
        st.button("Refresh", type="primary", width="stretch")  # any click re-runs the page
    st.caption(
        "Alerts use the **lower bound** of the 90% interval: the true RUL is very likely above "
        "it, so a low lower bound means failure is plausibly near."
    )

try:
    thresholds = RiskThresholds(critical=critical, warning=warning)
except ValueError:
    st.error("The critical threshold must be lower than the warning threshold.")
    st.stop()

try:
    health = client.health()
    units = client.units()
except DataSourceError as exc:
    st.error(str(exc))
    st.stop()

st.title("Turbofan fleet: remaining useful life")
if client.snapshot is not None:
    snap = client.snapshot
    st.info(
        "**Demo: static snapshot.** This is the full system's output on the NASA C-MAPSS FD001 "
        f"test engines (simulated), generated on {snap['generated_at'][:10]} with model "
        f"`{snap['model_name']}` v{snap['model_version']}. Nothing is computed live. "
        f"[Source code]({REPO_URL}) · "
        f"[Run the full system with Docker Compose]({REPO_URL}#quickstart-with-docker)",
        icon="📸",
    )
model_info = (
    f"model `{health.get('model_name')}` v{health.get('model_version')}"
    if health.get("status") == "ok"
    else "**no model loaded**: new predictions are unavailable"
)
st.caption(f"{model_info} · database: {health.get('database', 'unknown')}")

if not units:
    st.info(
        "No units yet. Stream some engines into the service, for example:\n\n"
        "`uv run python -m predmaint.simulation.replay --subset FD001 --units 1-10`"
    )
    st.stop()

# --- fleet view -----------------------------------------------------------------------------
table = fleet_table(units, thresholds)
kpis = fleet_kpis(table)
c1, c2, c3 = st.columns(3)
c1.metric("Units monitored", kpis["units"])
c2.metric(STATUS_LABELS["critical"], kpis["critical"], help=f"lower bound < {critical:g} cycles")
c3.metric(STATUS_LABELS["warning"], kpis["warning"], help=f"lower bound < {warning:g} cycles")
if kpis["no prediction"]:
    st.caption(f"{kpis['no prediction']} unit(s) have readings but no prediction yet.")


def _status_style(column: pd.Series) -> list[str]:
    text = {"warning": "#1a1a19"}  # dark text on amber for contrast; white elsewhere
    return [
        f"background-color: {STATUS_COLOURS[s]}; color: {text.get(s, '#ffffff')}; font-weight: 600"
        for s in column
    ]


st.subheader("Fleet ranked by risk")
# One .format call with every column's formatter: separate chained calls did not all reach
# the rendered table (RUL/lower/upper were shown unrounded).
TABLE_FORMAT = {
    "status": lambda s: STATUS_LABELS[s],
    "RUL": "{:.1f}",
    "lower": "{:.1f}",
    "upper": "{:.1f}",
    "predicted at cycle": "{:.0f}",
    "stale": lambda s: "yes" if s else "",
}
styled = table.style.apply(_status_style, subset=["status"]).format(TABLE_FORMAT, na_rep="–")
st.dataframe(styled, hide_index=True, width="stretch")
st.caption(
    "Sorted by the interval's lower bound (most at risk first). RUL, lower and upper are in "
    "cycles. **Stale** means newer readings arrived after the latest prediction."
)

# --- unit drill-down -------------------------------------------------------------------------
st.subheader("Unit drill-down")
unit = st.selectbox(
    "Unit",
    options=table["unit"].tolist(),
    format_func=lambda u: f"Unit {u} · {STATUS_LABELS[table.set_index('unit').at[u, 'status']]}",
)

try:
    history = prediction_history(client.predictions(unit))
    readings = pd.DataFrame(client.readings(unit))
except DataSourceError as exc:
    st.error(str(exc))
    st.stop()

if history.empty:
    st.info(f"Unit {unit} has no predictions yet: request one with GET /units/{unit}/prediction.")
else:
    base = alt.Chart(history).encode(x=alt.X("cycle:Q", title="cycle", scale=alt.Scale(zero=False)))
    tooltip = [
        alt.Tooltip("cycle:Q"),
        alt.Tooltip("rul:Q", title="RUL", format=".1f"),
        alt.Tooltip("lower:Q", format=".1f"),
        alt.Tooltip("upper:Q", format=".1f"),
    ]
    band = base.mark_area(opacity=0.22, color=BLUE).encode(
        y=alt.Y("lower:Q", title="predicted RUL (cycles)"), y2="upper:Q", tooltip=tooltip
    )
    line = base.mark_line(color=BLUE, strokeWidth=2).encode(y="rul:Q")
    points = base.mark_point(color=BLUE, filled=True, size=40).encode(y="rul:Q", tooltip=tooltip)
    limits = pd.DataFrame(
        {
            "value": [thresholds.critical, thresholds.warning],
            "label": [f"critical ({critical:g})", f"warning ({warning:g})"],
            "colour": [STATUS_COLOURS["critical"], STATUS_COLOURS["warning"]],
        }
    )
    rules = (
        alt.Chart(limits)
        .mark_rule(strokeDash=[5, 4], strokeWidth=1.5)
        .encode(y="value:Q", color=alt.Color("colour:N", scale=None))
    )
    rule_labels = (
        alt.Chart(limits)
        .mark_text(align="left", dx=4, dy=-6, fontSize=11)
        .encode(
            x=alt.value(0), y="value:Q", text="label:N", color=alt.Color("colour:N", scale=None)
        )
    )
    st.altair_chart(
        (band + line + points + rules + rule_labels).properties(
            height=320, title=f"Unit {unit}: predicted RUL with 90% interval"
        ),
        width="stretch",
    )

st.markdown("**Key sensors**")
if readings.empty:
    st.info("No stored readings for this unit.")
else:
    columns = st.columns(2)
    for i, sensor in enumerate(KEY_SENSORS):
        chart = (
            alt.Chart(readings)
            .mark_line(color=BLUE, strokeWidth=1.5)
            .encode(
                x=alt.X("cycle:Q", title="cycle"),
                y=alt.Y(f"{sensor}:Q", title=sensor, scale=alt.Scale(zero=False)),
                tooltip=["cycle:Q", alt.Tooltip(f"{sensor}:Q", format=".2f")],
            )
            .properties(height=180, title=sensor)
        )
        columns[i % 2].altair_chart(chart, width="stretch")

st.caption(
    "No true RUL is shown: an operator would not know it, and the service never receives it."
)

# --- data drift ----------------------------------------------------------------------------
st.subheader("Data drift")
last_n = st.number_input(
    "Most recent readings per unit",
    min_value=1,
    max_value=1000,
    value=client.drift_window or 30,
    step=10,
    key="drift_n",
    disabled=client.drift_window is not None,  # a snapshot holds one window only
)
try:
    report = client.drift(int(last_n))
except DataSourceError as exc:
    st.info(f"Drift monitoring is unavailable: {exc}")
    st.stop()

drift = drift_table(report)
st.caption(
    f"PSI of the {report['n_readings']} most recent readings ({int(last_n)} per unit, "
    f"{report['n_units']} units) against the {report['reference_rows']} training readings of "
    f"model v{report['model_version']}. Stable < 0.1 ≤ moderate < 0.25 ≤ significant."
)


def _drift_style(column: pd.Series) -> list[str]:
    text = {"moderate": "#1a1a19"}
    return [
        f"background-color: {DRIFT_COLOURS[s]}; color: {text.get(s, '#ffffff')}; font-weight: 600"
        for s in column
    ]


st.dataframe(
    drift.style.apply(_drift_style, subset=["status"]).format(
        {"PSI": "{:.3f}", "status": lambda s: DRIFT_LABELS[s]}
    ),
    hide_index=True,
    width="stretch",
)
st.warning(
    "**Read sensor drift with care.** " + report["caveat"],
    icon="⚠️",
)

if not drift.empty:
    chosen = st.selectbox("Column", options=drift["column"].tolist())
    column_drift = next(c for c in report["columns"] if c["column"] == chosen)
    hist = histogram_frame(column_drift)
    bars = (
        alt.Chart(hist)
        .mark_bar()
        .encode(
            x=alt.X(
                "bin:N", sort=alt.SortField("order"), title=f"{chosen} (reference decile bins)"
            ),
            xOffset=alt.XOffset("source:N", sort=["reference", "current"]),
            y=alt.Y("share:Q", title="share of readings", axis=alt.Axis(format="%")),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=["reference", "current"], range=["#8a8984", BLUE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=["bin:N", "source:N", alt.Tooltip("share:Q", format=".1%")],
        )
        .properties(
            height=260,
            title=f"{chosen}: PSI {column_drift['psi']:.3f} "
            f"({DRIFT_LABELS[column_drift['status']]})",
        )
    )
    st.altair_chart(bars, width="stretch")
