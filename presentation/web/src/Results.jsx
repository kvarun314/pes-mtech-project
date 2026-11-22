import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Bars, Legend, Num, ease } from "./ui.jsx";

const GREY = "#C7C7CC", BLUE = "#2C5AA0", RED = "#A93636";
const PP = [["Phase 1", GREY], ["Phase 2", BLUE]];
const ST = ["1★", "2★", "3★", "4★", "5★"];

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

/** 5×5 confusion matrix: rows = true star, columns = predicted star, shaded by share of the row. */
function Confusion({ cm, title, color }) {
  const [hov, setHov] = useState(null);
  return (
    <div className="cm">
      <div className="cmt">{title}</div>
      <div className="cmg">
        <span className="cmax">true ↓ · predicted →</span><span />{ST.map((s) => <span key={s} className="cmh">{s}</span>)}
        {cm.map((row, i) => { const n = row.reduce((a, b) => a + b, 0); return (
          <React.Fragment key={i}><span className="cmh">{ST[i]}</span>
            {row.map((v, j) => { const f = v / n; return (
              <motion.span key={j} className={`cmc${i === j ? " diag" : ""}`} onMouseEnter={() => setHov(`${ST[i]} reviews rated ${ST[j]}: ${v} of ${n} (${(f * 100).toFixed(0)}%)`)} onMouseLeave={() => setHov(null)}
                style={{ background: i === j ? color : RED, color: f > .35 ? "white" : "var(--ink)" }}
                initial={{ opacity: 0 }} whileInView={{ opacity: Math.max(.06, Math.min(1, f * 1.4)) }} viewport={{ once: true }} transition={{ duration: .6, delay: (i + j) * .04 }}>
                <b>{(f * 100).toFixed(0)}</b></motion.span>); })}
          </React.Fragment>); })}
      </div>
      <div className="note cmn">{hov || "Diagonal = right star. Hover a cell."}</div>
    </div>
  );
}

/** Where each answer lands: exact, one star off, further off. */
function Misses({ rows }) {
  const seg = [["Exact", BLUE], ["One star off", "#9DB7DE"], ["Two or more off", RED]];
  return (
    <div>
      <Legend items={seg} />
      {rows.map(([l, v], r) => (
        <div key={l} className="stk">
          <span className="cl">{l}</span>
          <div className="stkb">{v.map((x, i) => (
            <motion.div key={i} style={{ background: seg[i][1] }} initial={{ width: 0 }} whileInView={{ width: `${x}%` }} viewport={{ once: true }} transition={{ duration: 1, ease, delay: r * .2 + i * .15 }}>
              {x >= 6 && <span>{x.toFixed(0)}%</span>}</motion.div>))}</div>
        </div>))}
    </div>
  );
}

/** Horizontal ranking of every model on the balanced test. */
function Ladder({ models, main }) {
  const ms = [...models].filter((m) => !/seed 2|dev2? fit\)|DistilBERT \+|Base LLaMA \+|TF-IDF, 100K/.test(m.name)).sort((a, b) => a.be - b.be);
  return (
    <div className="ladder">{ms.map((m, i) => { const p1 = m.name.startsWith("Phase 1"), mn = m.name === main; return (
      <div key={m.name} className="lrow">
        <span className={mn ? "blue" : ""} style={{ fontWeight: mn || p1 ? 700 : 400 }}>{m.name.replace(" (main)", " (Phase 2)")}</span>
        <div className="ctrack"><motion.div style={{ background: mn ? BLUE : p1 ? "#8E8E93" : "#9DB7DE" }} initial={{ width: 0 }} whileInView={{ width: `${m.be}%` }} viewport={{ once: true }} transition={{ duration: .9, ease, delay: i * .06 }} /></div>
        <b>{m.be.toFixed(1)}</b>
      </div>); })}</div>
  );
}

function Section({ i, title, why, big, bigLabel, worse, children }) {
  return (
    <motion.section className="rsec" initial={{ opacity: 0, y: 24 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-60px" }} transition={{ duration: .8, ease }}>
      <div className="rtext">
        <div className="cnum">{String(i).padStart(2, "0")}</div>
        <h3>{title}</h3><p>{why}</p>
        {big && <div className={`rbig${worse ? " bad" : ""}`}>{big}<span>{bigLabel}</span></div>}
      </div>
      <div className="card rvis">{children}</div>
    </motion.section>
  );
}

export default function Results() {
  const [d, setD] = useState(null); const [more, setMore] = useState(false);
  useEffect(() => { fetch("/api/results").then((r) => r.json()).then(setD); }, []);
  if (!d) return <div className="empty"><span className="spin" /> Loading the frozen results…</div>;
  const B = d.breakdown, s = d.sure.eval, gain = d.exact.bal[1] - d.exact.bal[0];
  const err = B && ((p) => [p.exact, p.off_by_one, p.off_by_more]);
  let k = 0;
  return (
    <div style={{ marginTop: 18 }}>
      <div className="rhead">
        <div><div className="lab">The result in one line</div>
          <h2>+<Num value={gain} decimals={1} /> points on the exact star, and every explanation is checked.</h2></div>
        <p className="sub">Phase 1 is the reference method (one fine-tuned LLaMA-3). Phase 2 is our agent system. Both are scored once on test reviews neither ever saw: <b>balanced</b> (1,000, 200 per star) and <b>natural</b> (2,000, real mix, 65% 5★).</p>
      </div>

      {B && <Section i={++k} title="Right more often, and closer when wrong" big={`${d.exact.bal[0].toFixed(1)}% → ${d.exact.bal[1].toFixed(1)}%`} bigLabel="exact star, balanced test (p = 6.7e-14)"
        why="Phase 2 gets more stars exactly right, and when it misses, it usually misses by one star. Misses by two or more stars drop by more than half.">
        <h4>Where each answer lands · balanced test</h4>
        <Misses rows={[["Phase 1", err(B.eval.p1)], ["Phase 2", err(B.eval.p2)]]} />
        <h4 style={{ marginTop: 22 }}>Right star, per true star · balanced test (%)</h4><Legend items={PP} />
        <Bars groups={ST} series={["Phase 1", "Phase 2"]} values={ST.map((_, i) => [B.eval.p1.recall[i], B.eval.p2.recall[i]])} colors={[GREY, BLUE]} max={100} fmt={(v) => v.toFixed(0)} h={210} />
      </Section>}

      {B && <Section i={++k} title="Where the mistakes go" big={`${B.eval.p1.recall[1].toFixed(0)}% → ${B.eval.p2.recall[1].toFixed(0)}%`} bigLabel="of 2★ reviews rated right"
        why="Each row is the true star, each column the star given. Phase 1 spreads its answers across neighbouring stars. Phase 2 keeps a stronger diagonal on four of five stars; on 4★ the two tie.">
        <div className="cms"><Confusion cm={B.eval.p1.confusion} title="Phase 1" color="#6E6E73" /><Confusion cm={B.eval.p2.confusion} title="Phase 2" color={BLUE} /></div>
      </Section>}

      {B && <Section i={++k} title="Fair to every star, not just 5★" big={`${d.f1.nat[0].toFixed(3)} → ${d.f1.nat[1].toFixed(3)}`} bigLabel="macro-F1, natural test"
        why="On real data 65% of reviews are 5★, so raw accuracy flatters any system. F1 per star shows Phase 2 is better on every star, including the rare 2★ and 3★.">
        <h4>F1 per star · natural test (×100)</h4><Legend items={PP} />
        <Bars groups={ST} series={["Phase 1", "Phase 2"]} values={ST.map((_, i) => [B.natural_test.p1.f1[i] * 100, B.natural_test.p2.f1[i] * 100])} colors={[GREY, BLUE]} max={100} fmt={(v) => v.toFixed(0)} h={230} />
      </Section>}

      <Section i={++k} title="Every agent helps; together they are best" big={`${d.exact.bal[0].toFixed(1)}% → ${d.exact.bal[1].toFixed(1)}%`} bigLabel="Phase 1 → three agents + Reflector"
        why="Each agent alone already beats Phase 1. Combined by the Reflector, they beat each of them. Phase 1 trains on 2K reviews, the agents on 40K–497K: a disclosed caveat.">
        <h4>Exact star · balanced test (%)</h4><Ladder models={d.models} main={d.main} />
      </Section>

      <Section i={++k} title="Knows when it is sure" big={`${(s.p1.acc_auto * 100).toFixed(1)}% → ${(s.p2.acc_auto * 100).toFixed(1)}%`} bigLabel={`right on the ${(s.p2.coverage * 100).toFixed(0)}% of balanced reviews each answers alone`}
        why="Sort reviews by confidence and answer only the surest ones. Phase 2's line stays higher at every level: its confidence means something, so hard cases can go to a person.">
        <h4>Accuracy vs share answered automatically · natural test</h4><Legend items={PP} /><Curve curve={d.curve} auto={d.auto} />
      </Section>

      <Section i={++k} title="Not fooled by negative words" big={`${d.hard.bal[0].toFixed(1)}% → ${d.hard.bal[1].toFixed(1)}%`} bigLabel={`happy reviews with negative words, balanced (n ${d.hard.n[0]})`}
        why="“Not bad at all” sounds negative to a single reader. The Evidence model weighs whole phrases and the probe reads meaning, so happy reviews stay happy.">
        <h4>Exact star on 4★–5★ reviews that use negative words (%)</h4><Legend items={PP} />
        <Bars groups={[`Balanced (n ${d.hard.n[0]})`, `Natural (n ${d.hard.n[1]})`]} series={["Phase 1", "Phase 2"]} values={[d.hard.bal, d.hard.nat]} colors={[GREY, BLUE]} max={100} />
      </Section>

      <Section i={++k} title="Steady on messy text, but it needs the title" big={`${d.robust["No punctuation"][0].toFixed(1)}% → ${d.robust["No punctuation"][1].toFixed(1)}%`} bigLabel="stars changed by flat text (lower is better)"
        why="The same review is altered and rated again. Phase 2 changes its mind far less on typos and flat text. One honest weakness: two agents read the title, so deleting it changes more stars.">
        <h4>Stars that changed (%, lower is better) · {d.robust_n.toLocaleString()} balanced reviews</h4><Legend items={PP} />
        <Bars groups={Object.keys(d.robust)} series={["Phase 1", "Phase 2"]} values={Object.values(d.robust)} colors={[GREY, BLUE]} max={36} />
      </Section>

      <Section i={++k} title="Explanations get checked and fixed" big={`${d.critic.nat[0].toFixed(1)}% → ${d.critic.nat[1].toFixed(1)}%`} bigLabel="pass all five checks, natural test · 0 made worse"
        why="The Critic sends failing drafts back with a precise instruction. A blind human check agrees most of the time, less when the agents disagreed: that is the next thing to fix.">
        <h4>Pass rate before and after the Critic loop (%)</h4><Legend items={[["First draft", GREY], ["After the loop", BLUE]]} />
        <Bars groups={["Balanced test", "Natural test"]} series={["First draft", "After the loop"]} values={[d.critic.bal, d.critic.nat]} colors={[GREY, BLUE]} max={100} h={200} />
        {d.human_split?.agreed && <>
          <h4 style={{ marginTop: 18 }}>Blind human check of 50 explanations: faithful (%)</h4><Legend items={[["Person", BLUE], ["Critic", GREY]]} />
          <Bars groups={[`Agents agreed (n ${d.human_split.agreed.n})`, `Agents disagreed (n ${d.human_split.disagreed.n})`]} series={["Person", "Critic"]}
            values={[[d.human_split.agreed.human, d.human_split.agreed.critic], [d.human_split.disagreed.human, d.human_split.disagreed.critic]]} colors={[BLUE, GREY]} max={100} h={200} fmt={(v) => v.toFixed(0)} />
        </>}
      </Section>

      <button className="lg" style={{ marginTop: 18 }} onClick={() => setMore(!more)}>{more ? "Hide" : "Show"} every model and the fine print</button>
      {more && (
        <motion.div className="card" style={{ marginTop: 12 }} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .4, ease }}>
          <p className="note" style={{ marginTop: 0 }}>On the natural test 65% of reviews are 5★, so raw exact accuracy flatters Phase 2 ({d.exact.nat[0].toFixed(1)}% to {d.exact.nat[1].toFixed(1)}%).
            Against Phase 1 adjusted to that mix the gain is +2.5 points, which is why macro-F1 is used there. The two-agent system fixed before any test read scores 67.5% / 75.7%.</p>
          <table className="t">
            <thead><tr><th>Model</th><th>Balanced exact %</th><th>Balanced macro-F1</th><th>Natural exact %</th><th>Natural macro-F1</th></tr></thead>
            <tbody>{d.models.map((m) => <tr key={m.name} className={m.name === d.main ? "main" : ""}><td>{m.name}</td><td>{m.be.toFixed(1)}</td><td>{m.bf.toFixed(3)}</td><td>{m.ne.toFixed(1)}</td><td>{m.nf.toFixed(3)}</td></tr>)}</tbody>
          </table>
        </motion.div>
      )}
      <footer>Sources: results_frozen.md, demo_breakdown.json, s1_selective.json, b2_report.json, q4_robustness(_1k).json, hard_cases.json, human_check.out (v2_reflective/e23/).</footer>
    </div>
  );
}
