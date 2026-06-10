from pathlib import Path

p = Path(__file__).resolve().parent.parent / "src" / "visualizer.py"
t = p.read_text(encoding="utf-8")
start = t.find("        width_tag = self._crack_width_status_tag(result)\n\n        primary = (")
end = t.find("\n    def show_offline_training_metrics_panel", start)
if start == -1 or end == -1:
    raise SystemExit(f"anchors not found start={start} end={end}")

new = """
        width_tag = self._crack_width_status_tag(result)
        if width_tag == "处于安全区":
            width_tag = "安全范围"

        try:
            p_raw = float(sd.get("risk_probability", preds.get("risk_confidence", 0.0)))
        except (TypeError, ValueError):
            p_raw = 0.0
        if not math.isfinite(p_raw):
            p_raw = 0.0
        prob_band = self._prob_warning_band(min(max(p_raw, 0.0), 1.0))

        st.markdown(
            home_kpi_dashboard_html(
                risk_level=lvl,
                risk_tone=tone,
                risk_subtitle=self._risk_level_subtitle(lvl),
                prob_value=p_txt,
                prob_band=prob_band,
                width_value=w_txt,
                width_tag=width_tag,
                density_value=dens_txt,
                density_unit="条/m²",
                compressive_value=comp_txt,
                flexural_value=flex_txt,
            ),
            unsafe_allow_html=True,
        )
"""

p.write_text(t[:start] + new + t[end:], encoding="utf-8")
print("ok")
