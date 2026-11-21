import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Bars, Legend, Num, Reveal, ease } from "./ui.jsx";

const GREY = "#C7C7CC", BLUE = "#2C5AA0";
const PP = [["Phase 1", GREY], ["Phase 2", BLUE]];

/** Coverage vs accuracy curves with a hover crosshair. */
function Curve({ curve, auto }) {
  const W = 520, H = 260, L = 46, R = 16, T = 14, B = 40;
  const ys = [...curve.p1, ...curve.p2].map((p) => p[1]); const y0 = Math.floor(Math.min(...ys) / 10) * 10;
  const X = (v) => L + (v / 100) * (W - L - R), Y = (v) => T + (1 - (v - y0) / (100 - y0)) * (H - T - B);
  const path = (pts) => [...pts].sort((a, b) => a[0] - b[0]).map((p, i) => `${i ? "L" : "M"}${X(p[0])},${Y(p[1])}`).join(" ");
  const [hx, setHx] = useState(null);
  const near = (pts, x) => pts.reduce((a, p) => (Math.abs(p[0] - x) < Math.abs(a[0] - x) ? p : a));
  const move = (e) => { const r = e.currentTarget.getBoundingClientRect(); const x = ((e.clientX - r.left) / r.width) * W; setHx(Math.max(0, Math.min(100, ((x - L) / (W - L - R)) * 100))); };
  const a = hx != null && near(curve.p1, hx), b = hx != null && near(curve.p2, hx);
  return (
    <div className="chart rel">
      <svg viewBox={`0 0 ${W} ${H}`} onMouseMove={move} onMouseLeave={() => setHx(null)}>
        {[y0, (y0 + 100) / 2, 100].map((v) => <g key={v}><line x1={L} x2={W - R} y1={Y(v)} y2={Y(v)} stroke="var(--tint)" /><text x={L - 8} y={Y(v) + 4} textAnchor="end" fontSize="12" fill="var(--muted)">{v}%</text></g>)}
        {[0, 25, 50, 75, 100].map((v) => <text key={v} x={X(v)} y={H - 20} textAnchor="middle" fontSize="12" fill="var(--muted)">{v}%</text>)}
        <text x={(L + W - R) / 2} y={H - 2} textAnchor="middle" fontSize="12.5" fill="var(--body)">reviews answered automatically</text>
        {[["p1", GREY], ["p2", BLUE]].map(([k, c], i) => (
          <motion.path key={k} d={path(curve[k])} fill="none" stroke={c} strokeWidth="4" strokeLinecap="round" initial={{ pathLength: 0 }} whileInView={{ pathLength: 1 }} viewport={{ once: true }} transition={{ duration: 1.4, ease, delay: i * .2 }} />
        ))}
        <motion.circle cx={X(auto.coverage)} cy={Y(auto.acc)} r="7" fill={BLUE} stroke="white" strokeWidth="3" initial={{ scale: 0 }} whileInView={{ scale: 1 }} viewport={{ once: true }} transition={{ delay: 1.5 }} />
        <motion.text x={X(auto.coverage) + 12} y={Y(auto.acc) - 10} fontSize="13" fontWeight="700" fill={BLUE} initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ delay: 1.6 }}>
          {auto.acc.toFixed(1)}% right on {auto.coverage.toFixed(0)}%</motion.text>
        {hx != null && <g><line x1={X(hx)} x2={X(hx)} y1={T} y2={H - B} stroke="var(--muted)" strokeDasharray="3 4" />
          <circle cx={X(a[0])} cy={Y(a[1])} r="5" fill={GREY} /><circle cx={X(b[0])} cy={Y(b[1])} r="5" fill={BLUE} /></g>}
      </svg>
      {hx != null && <div className="tt" style={{ left: `${(X(hx) / W) * 100}%`, top: `${(T / H) * 100}%`, transform: "translate(-50%, 0)" }}>
        answering {b[0].toFixed(0)}%: Phase 2 {b[1].toFixed(1)}% right · Phase 1 {near(curve.p1, b[0])[1].toFixed(1)}%</div>}
    </div>
  );
}

const Chart = ({ title, sub, children, delay }) => (
  <Reveal className="card ch" delay={delay}><h4>{title}</h4><div className="sub">{sub}</div>{children}</Reveal>
);

export default function Results() {
  const [d, setD] = useState(null);
  useEffect(() => { fetch("/api/results").then((r) => r.json()).then(setD); }, []);
  if (!d) return <div className="empty"><span className="spin" /> Loading the frozen results…</div>;
  const tiles = [
    ["Exact star · balanced", d.exact.bal[1], 1, "%", `vs ${d.exact.bal[0].toFixed(1)}% for Phase 1 (+${(d.exact.bal[1] - d.exact.bal[0]).toFixed(1)} points)`],
    ["Macro-F1 · natural", d.f1.nat[1], 3, "", `vs ${d.f1.nat[0].toFixed(3)} for Phase 1; fair to every star`],
    ["Right when it answers alone", d.auto.acc, 1, "%", `on the ${d.auto.coverage.toFixed(0)}% of natural reviews it is sure about`],
    ["Explanations that pass", d.critic.nat[1], 1, "%", `Critic pass rate, from ${d.critic.nat[0].toFixed(1)}% at first draft; human check ${d.human.toFixed(0)}%`],
  ];
  return (
    <div style={{ marginTop: 18 }}>
      <p className="sub" style={{ fontSize: 16 }}>Frozen results on two test sets the system never trained on: <b>balanced</b> (1,000 reviews, 200 per star) and <b>natural</b> (2,000 reviews in their real mix, 65% five-star).</p>
      <div className="grid g4">
        {tiles.map(([l, v, dec, suf, sub], i) => (
          <Reveal key={l} className="card" delay={i * .08}><div className="lab">{l}</div><div className="big blue"><Num value={v} decimals={dec} suffix={suf} /></div><div className="sub">{sub}</div></Reveal>
        ))}
      </div>
      <div className="grid g2" style={{ marginTop: 16 }}>
        <Chart title="More stars exactly right" sub="Exact-star accuracy (%)."><Legend items={PP} />
          <Bars groups={["Balanced test", "Natural test"]} series={["Phase 1", "Phase 2"]} values={[d.exact.bal, d.exact.nat]} colors={[GREY, BLUE]} max={100} /></Chart>
        <Chart title="Fairer to every star" sub="Macro-F1 weighs all five stars equally, so the common 5★ cannot hide mistakes." delay={.08}><Legend items={PP} />
          <Bars groups={["Balanced test", "Natural test"]} series={["Phase 1", "Phase 2"]} values={[d.f1.bal, d.f1.nat]} colors={[GREY, BLUE]} fmt={(v) => v.toFixed(2)} max={1} /></Chart>
        <Chart title="Knows when it is unsure" sub="Answer only the reviews it is most sure of; accuracy on those (natural test). Hover the chart."><Legend items={PP} />
          <Curve curve={d.curve} auto={d.auto} /></Chart>
        <Chart title="Steadier on messy text" sub="Stars that changed when the review was altered (%, lower is better). Phase 1 never reads the title." delay={.08}><Legend items={PP} />
          <Bars groups={Object.keys(d.robust)} series={["Phase 1", "Phase 2"]} values={Object.values(d.robust)} colors={[GREY, BLUE]} max={36} /></Chart>
        <Chart title="The Critic loop repairs explanations" sub="Share of explanations that pass all five checks (%)."><Legend items={[["First draft", GREY], ["After the loop", BLUE]]} />
          <Bars groups={["Balanced test", "Natural test"]} series={["First draft", "After the loop"]} values={[d.critic.bal, d.critic.nat]} colors={[GREY, BLUE]} max={100} /></Chart>
        <Chart title="Not fooled by negative words" sub={`Happy reviews that use negative words: exact star (%, n = ${d.hard.n[0]} / ${d.hard.n[1]}).`} delay={.08}><Legend items={PP} />
          <Bars groups={["Balanced test", "Natural test"]} series={["Phase 1", "Phase 2"]} values={[d.hard.bal, d.hard.nat]} colors={[GREY, BLUE]} max={100} /></Chart>
      </div>
      <p className="note" style={{ marginTop: 16 }}>Read with care: on the natural test 65% of reviews are 5★, so raw exact accuracy flatters Phase 2. Against Phase 1 adjusted to that mix the gain is +2.5 points,
        which is why macro-F1 is the headline there. The two-agent system fixed before any test read scores 67.5% / 75.7%.</p>
      <Reveal className="card" style={{ marginTop: 16 }}>
        <h4 style={{ margin: "0 0 8px" }}>Every model on the test sets</h4>
        <table className="t">
          <thead><tr><th>Model</th><th>Balanced exact %</th><th>Balanced macro-F1</th><th>Natural exact %</th><th>Natural macro-F1</th></tr></thead>
          <tbody>{d.models.map((m) => <tr key={m.name} className={m.name === d.main ? "main" : ""}><td>{m.name}</td><td>{m.be.toFixed(1)}</td><td>{m.bf.toFixed(3)}</td><td>{m.ne.toFixed(1)}</td><td>{m.nf.toFixed(3)}</td></tr>)}</tbody>
        </table>
      </Reveal>
      <footer>Sources: results_frozen.md, s1_selective.json, b2_report.json, q4_robustness.json, hard_cases.json, human_check.out (v2_reflective/e23/).</footer>
    </div>
  );
}
