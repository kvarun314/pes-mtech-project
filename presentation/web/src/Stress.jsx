import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { ease, pct } from "./ui.jsx";

export default function Stress({ inp, data, setData }) {
  const [busy, setBusy] = useState(false); const [err, setErr] = useState(null); const [rob, setRob] = useState(null); const [show, setShow] = useState(false);
  useEffect(() => { fetch("/api/results").then((r) => r.json()).then((d) => setRob(d.robust)).catch(() => {}); }, []);
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
        <p className="note" style={{ marginTop: 16 }}>On 300 test reviews: typos changed {rob.Typos[1].toFixed(1)}% of Phase 2 stars vs {rob.Typos[0].toFixed(1)}% for Phase 1; flat text {rob["No punctuation"][1].toFixed(1)}% vs {rob["No punctuation"][0].toFixed(1)}%.
          Removing the title changed {rob["No title"][1].toFixed(1)}% of Phase 2 stars, because Phase 2 reads the title and Phase 1 does not.</p>
      )}
    </div>
  );
}
