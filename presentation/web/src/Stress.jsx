import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { ease, pct } from "./ui.jsx";

/** Rate every demo review in all four versions and count how often each phase changes its star. */
function AllDemos({ demos }) {
  const [rows, setRows] = useState([]); const [busy, setBusy] = useState(false); const [err, setErr] = useState(null);
  const run = async () => {
    setBusy(true); setErr(null); setRows([]); const acc = [];
    for (const d of demos) {
      try { const r = await fetch("/api/stress", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: d.title, text: d.text, id: d.id }) }); const v = await r.json();
        if (v.error) { setErr(v.error); break; } acc.push({ d, v }); setRows([...acc]); } catch (e) { setErr(String(e)); break; }
    }
    setBusy(false);
  };
  const n = rows.length * 3, ch = (k) => rows.reduce((a, { v }) => a + v.slice(1).filter((x) => x[k] !== v[0][k]).length, 0);
  return (
    <div className="card" style={{ marginTop: 18 }}>
      <h4 style={{ margin: 0 }}>All {demos.length} demo reviews at once</h4>
      <p className="sub" style={{ margin: "4px 0 12px" }}>Every demo review in all four versions: {demos.length * 4} live ratings, about a minute.</p>
      <button className="btn inline" onClick={run} disabled={busy || !demos.length}>{busy ? <>Rating {rows.length + 1} of {demos.length}… <span className="spin" /></> : "Test all demo reviews"}</button>
      {err && <div className="warnbar">Live stress test failed: {err}</div>}
      {rows.length > 0 && <>
        <table className="t" style={{ marginTop: 14 }}>
          <thead><tr><th>Review</th>{rows[0].v.map((x) => <th key={x.name}>{x.name}</th>)}</tr></thead>
          <tbody>{rows.map(({ d, v }, i) => (
            <motion.tr key={d.id} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .4, ease }}>
              <td><b>Demo {demos.indexOf(d) + 1}</b> <span className="note">true {d.truth}★</span></td>
              {v.map((x, j) => <td key={x.name}><span className="note">P1</span> <span className={j && x.p1 !== v[0].p1 ? "bad" : ""}>{x.p1}★</span> · <span className="note">P2</span> <b className={j && x.p2 !== v[0].p2 ? "bad" : "blue"}>{x.p2}★</b></td>)}
            </motion.tr>))}</tbody>
        </table>
        <p className="sub" style={{ marginTop: 12 }}>Star changed in <b>{ch("p1")} of {n}</b> altered versions for Phase 1 and <b className="blue">{ch("p2")} of {n}</b> for Phase 2 (red = changed).</p>
      </>}
    </div>
  );
}

export default function Stress({ inp, demos = [], data, setData }) {
  const [busy, setBusy] = useState(false); const [err, setErr] = useState(null); const [rob, setRob] = useState(null); const [show, setShow] = useState(false);
  useEffect(() => { fetch("/api/results").then((r) => r.json()).then((d) => setRob(d)).catch(() => {}); }, []);
  const run = async () => {
    setBusy(true); setErr(null);
    try { const r = await fetch("/api/stress", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(inp) }); const d = await r.json();
      if (d.error) setErr(d.error); else setData(d); } catch (e) { setErr(String(e)); }
    setBusy(false);
  };
  const o = data?.[0];
  const badge = (v, ref, i) => i === 0 ? null : <span className={`chip ${v === ref ? "okc" : "badc"}`}>{v === ref ? "same" : "changed"}</span>;
  return (
    <div style={{ marginTop: 18 }}>
      <p className="sub" style={{ fontSize: 16 }}>Real reviews are messy. Here the same review is changed in three ways and rated again. A good system keeps the same star.</p>
      <button className="btn inline" onClick={run} disabled={busy}>{busy ? <>Rating four versions… <span className="spin" /></> : "Rate all four versions"}</button>
      {err && <div className="warnbar">Live stress test failed: {err}</div>}
      {data && (
        <div className="card" style={{ marginTop: 18 }}>
          <table className="t">
            <thead><tr><th>Version</th><th>Phase 1</th><th>Phase 2</th></tr></thead>
            <tbody>
              {data.map((r, i) => (
                <motion.tr key={r.name} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: i * .12, duration: .4, ease }}>
                  <td><b>{r.name}</b><br /><span className="note">{r.what}</span></td>
                  <td>{r.p1}★ {badge(r.p1, o.p1, i)}</td>
                  <td><b className="blue" style={{ fontSize: 18 }}>{r.p2}★</b> {badge(r.p2, o.p2, i)} <span className="note">{pct(r.conf)} sure</span></td>
                </motion.tr>
              ))}
            </tbody>
          </table>
          <button className="lg" style={{ marginTop: 14 }} onClick={() => setShow(!show)}>{show ? "Hide" : "Show"} the altered text</button>
          {show && data.map((r) => <p key={r.name} className="note"><b>{r.name}</b><br />{r.title || "(no title)"} · {r.text}</p>)}
        </div>
      )}
      {rob && (
        <p className="note" style={{ marginTop: 16 }}>On {rob.robust_n.toLocaleString()} test reviews: typos changed {rob.robust.Typos[1].toFixed(1)}% of Phase 2 stars vs {rob.robust.Typos[0].toFixed(1)}% for Phase 1; flat text {rob.robust["No punctuation"][1].toFixed(1)}% vs {rob.robust["No punctuation"][0].toFixed(1)}%.
          Removing the title changed {rob.robust["No title"][1].toFixed(1)}% of Phase 2 stars, because Phase 2 reads the title and Phase 1 does not.</p>
      )}
      <AllDemos demos={demos} />
    </div>
  );
}
