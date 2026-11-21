import React, { useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Highlight, Reveal, Step, ease, pct } from "./ui.jsx";

const CHECKS = {
  1: ["Right star", "States the same star the system chose."],
  2: ["Real quotes", "Every quoted phrase appears in the review."],
  3: ["Key evidence", "Uses at least two of the strongest phrases."],
  4: ["Honest about doubt", "Says so if the agents disagreed."],
  5: ["Nothing made up", "A second LLaMA-3 finds no unsupported claim."],
};

/** Reads a POST text/event-stream response and calls onEvent for every `data:` line. */
async function stream(url, payload, onEvent) {
  const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
  const rd = r.body.getReader(), dec = new TextDecoder(); let buf = "";
  for (;;) {
    const { value, done } = await rd.read(); if (done) break;
    buf += dec.decode(value, { stream: true }); const parts = buf.split("\n\n"); buf = parts.pop();
    for (const p of parts) if (p.startsWith("data: ")) onEvent(JSON.parse(p.slice(6)));
  }
}

export default function Explain({ res, inp, state, setState }) {
  const [phase, setPhase] = useState(null);           // which box of the process is live
  const s = state || { drafts: [], critic: [], pack: null, done: false };
  const run = async () => {
    let cur = { drafts: [], critic: [], pack: null, done: false, running: true, t0: Date.now() }; setState(cur); setPhase("evidence");
    const up = (patch) => { cur = { ...cur, ...patch }; setState(cur); };
    try {
      await stream("/api/explain", { title: inp.title, text: inp.text, id: inp.id, res }, (e) => {
        if (e.type === "pack") { up({ pack: e.pack }); setPhase("explainer"); }
        if (e.type === "draft") { up({ drafts: [...cur.drafts, e.text] }); setPhase("critic"); }
        if (e.type === "critic") { up({ critic: [...cur.critic, { feedback: e.feedback, failed: e.failed }] }); if (e.feedback?.length) setPhase("explainer"); }
        if (e.type === "warning") up({ warning: e.text });
        if (e.type === "error") up({ error: e.text, running: false });
        if (e.type === "done") { up({ done: true, running: false, seconds: e.seconds }); setPhase(null); }
      });
    } catch (ex) { up({ error: String(ex), running: false }); setPhase(null); }
  };
  const pk = s.pack || { for: res.pos, against: res.neg };
  const last = s.critic.length ? s.critic[s.critic.length - 1] : null; const passed = last && !last.feedback?.length;
  const boxes = [["evidence", "1 · Evidence", "The star, its confidence and the strongest phrases for and against it."],
                 ["explainer", "2 · Explainer", "LLaMA-3 writes one or two plain sentences from that evidence only."],
                 ["critic", "3 · Critic", "Runs five checks. If one fails, it says what to fix and sends the draft back (up to four rewrites)."]];
  return (
    <div style={{ marginTop: 18 }}>
      <div className="procs">
        {boxes.map(([id, t, d], i) => (
          <React.Fragment key={id}>
            <motion.div className={`proc${phase === id ? " live" : ""}`} animate={{ scale: phase === id ? 1.03 : 1 }} transition={{ duration: .3 }}>
              <b>{t} {phase === id && <span className="spin" />}</b>{d}
            </motion.div>
            {i < 2 && <div className="arr">→</div>}
          </React.Fragment>
        ))}
      </div>
      <div style={{ marginTop: 18, display: "flex", gap: 14, alignItems: "center" }}>
        <button className="btn inline" onClick={run} disabled={s.running}>{s.running ? "Working…" : s.drafts.length ? "Run again" : "Write and check the explanation"}</button>
        <span className="note">{s.running ? `Live on this laptop · ${s.drafts.length} draft${s.drafts.length === 1 ? "" : "s"} so far` : s.done && s.seconds != null ? `Done in ${s.seconds} s` : "Takes about 15–60 s, depending on rewrites."}</span>
      </div>
      {s.warning && <div className="warnbar">{s.warning}</div>}
      {s.error && <div className="warnbar">Live explanation failed: {s.error}</div>}

      {s.drafts.length > 0 && (
        <>
          <AnimatePresence>
            {s.done && (
              <motion.div className="card final" style={{ marginTop: 24 }} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: .5, ease }}>
                <div className="lab" style={{ marginBottom: 6 }}>Final explanation · {passed ? "passed all five checks" : "best draft after four rewrites"} · {s.drafts.length} draft{s.drafts.length > 1 ? "s" : ""}</div>
                <Highlight text={s.drafts[s.drafts.length - 1]} pos={pk.for} neg={pk.against} />
              </motion.div>
            )}
          </AnimatePresence>

          <div className="section">
            <Step n={1} title="What the Explainer was given">It may only use this. The Critic checks the draft against it.</Step>
            <Reveal className="card">
              <p style={{ marginTop: 0 }}><b style={{ fontSize: 20 }}>{res.star}★</b> <span className="sub">at {pct(Math.max(...res.P))} confidence · {res.agreed ? "agents agreed" : "agents disagreed, so the explanation must say so"}</span></p>
              {pk.for.map((p, i) => <motion.span key={"f" + i} className="chip for" initial={{ opacity: 0, scale: .8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * .07 }}>{p.display}</motion.span>)}
              {pk.against.map((p, i) => <motion.span key={"a" + i} className="chip against" initial={{ opacity: 0, scale: .8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: .35 + i * .07 }}>{p.display}</motion.span>)}
            </Reveal>
          </div>

          <div className="section">
            <Step n={2} title="Draft by draft">Each draft and how it scored on the five checks. A red check comes with the Critic's instruction for the next draft.</Step>
            <div className="card">
              {s.drafts.map((d, i) => {
                const c = s.critic[i]; const failed = c?.failed ? new Set(c.failed) : null; const judged = failed && [...failed].every((k) => k === 5);
                return (
                  <motion.div key={i} className={`draft${c && !c.feedback?.length ? " pass" : ""}`} initial={{ opacity: 0, x: -10 }} animate={{ opacity: 1, x: 0 }} transition={{ duration: .4, ease }}>
                    <div className="lab">Draft {i + 1}{c ? (c.feedback?.length ? " · sent back" : " · passed") : ""} {!c && <span className="spin" />}</div>
                    <div className="txt"><Highlight text={d} pos={pk.for} neg={pk.against} /></div>
                    {c && failed && (
                      <div className="checks">
                        {Object.entries(CHECKS).map(([k, [t, why]], j) => {
                          k = +k; const st = failed.has(k) ? "fail" : k === 5 && !judged ? "skip" : "pass";
                          return (
                            <motion.div key={k} className={`ck ${st}`} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: j * .08 }}>
                              <b>{st === "fail" ? "✗" : st === "skip" ? "–" : "✓"} {t}</b>{why}
                            </motion.div>
                          );
                        })}
                      </div>
                    )}
                    {c?.feedback?.map((f, j) => <div key={j} className="ask">Critic: {f}</div>)}
                  </motion.div>
                );
              })}
              <div className="note">Check 5 (grey dash) runs only after checks 1–4 pass. On all 3,000 test reviews this loop raises the Critic pass rate from 62–64% to 90.5–91.6%; a blind human check of 50 explanations rated 80% faithful.</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
