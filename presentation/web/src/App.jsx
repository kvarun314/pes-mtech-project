import React, { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Rate from "./Rate.jsx";
import Explain from "./Explain.jsx";
import Stress from "./Stress.jsx";
import Results from "./Results.jsx";
import { ease } from "./ui.jsx";

const TABS = ["How it rates", "How it explains", "Stress test", "Results"];

export default function App() {
  const [demos, setDemos] = useState([]); const [pick, setPick] = useState(null);
  const [title, setTitle] = useState(""); const [text, setText] = useState("");
  const [res, setRes] = useState(null); const [inp, setInp] = useState(null); const [busy, setBusy] = useState(false); const [err, setErr] = useState(null);
  const [tab, setTab] = useState(0); const [expl, setExpl] = useState(null); const [stress, setStress] = useState(null);

  useEffect(() => { fetch("/api/demos").then((r) => r.json()).then((d) => { setDemos(d); if (d.length) choose(d[0]); }).catch(() => {}); }, []);
  const choose = (d) => { setPick(d?.id ?? null); setTitle(d?.title ?? ""); setText(d?.text ?? ""); };
  const demo = demos.find((d) => d.id === pick);

  const rate = async () => {
    setBusy(true); setErr(null);
    const payload = { title, text, id: pick };
    try {
      const r = await fetch("/api/rate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) }); const d = await r.json();
      if (d.error) setErr(d.error);
      else { setRes(d); setInp({ ...payload, truth: demo && demo.text === text ? demo.truth : null }); setExpl(null); setStress(null); if (tab === 3) setTab(0); }
    } catch (e) { setErr(String(e)); }
    setBusy(false);
  };

  return (
    <>
      <div className="hero"><div className="wrap">
        <motion.div className="tag" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>M.Tech project · live demo</motion.div>
        <motion.h1 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .6, ease }}>Agentic review rating</motion.h1>
        <motion.p initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .6, ease, delay: .1 }}>Three agents rate the star. LLaMA-3 explains it. A Critic checks it. Everything runs live on this laptop.</motion.p>
      </div></div>

      <div className="wrap">
        <motion.div className="input" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .6, ease, delay: .2 }}>
          <div>
            <label>Review</label>
            <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Title, e.g. Stopped working after a week" />
            <textarea value={text} onChange={(e) => setText(e.target.value)} placeholder="Paste or type a product review…" />
          </div>
          <div>
            <label>Start from</label>
            <div className="demos">
              {demos.map((d, i) => <button key={d.id} className={`demo${pick === d.id ? " on" : ""}`} onClick={() => choose(d)}>Demo {i + 1} · {d.truth}★</button>)}
              <button className={`demo${pick === null ? " on" : ""}`} onClick={() => choose(null)}>Write my own</button>
            </div>
            <button className="btn" onClick={rate} disabled={busy || !text.trim()}>{busy ? <>Reading the review… <span className="spin" style={{ borderTopColor: "white" }} /></> : "Rate this review"}</button>
            <div className="hint">First run loads the models (about 40 s).</div>
          </div>
        </motion.div>
        {err && <div className="warnbar">Live run failed: {err}</div>}

        <div className="tabs">
          {TABS.map((t, i) => (
            <button key={t} className={`tab${tab === i ? " on" : ""}`} onClick={() => setTab(i)}>
              {tab === i && <motion.span layoutId="pill" className="pill" transition={{ type: "spring", stiffness: 420, damping: 34 }} />}{t}
            </button>
          ))}
        </div>

        <AnimatePresence mode="wait">
          <motion.div key={tab + (res ? "r" : "")} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }} transition={{ duration: .3, ease }}>
            {tab === 3 ? <Results /> : !res ? <div className="card empty" style={{ marginTop: 18 }}>Pick a demo review or write your own, then press <b>Rate this review</b>.</div>
              : tab === 0 ? <Rate res={res} inp={inp} /> : tab === 1 ? <Explain res={res} inp={inp} state={expl} setState={setExpl} /> : <Stress inp={inp} data={stress} setData={setStress} />}
          </motion.div>
        </AnimatePresence>
        <footer>Phase 1: LLaMA-3-8B + LoRA (reference method, Wang 2025). Phase 2: LLaMA probe, Evidence model and DeBERTa-v3, combined by a Reflector, with a Hand-off agent and an Explainer–Critic loop. Data: Amazon Reviews 2023, Electronics.</footer>
      </div>
    </>
  );
}
