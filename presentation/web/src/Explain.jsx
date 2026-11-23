import React from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Highlight, Reveal, Step, ease, pct } from "./ui.jsx";

export const CHECKS = {
  1: ["Right star", "States the same star the system chose."],
  2: ["Real quotes", "Every quoted phrase appears in the review."],
  3: ["Key evidence", "Uses at least two of the strongest phrases."],
  4: ["Honest about doubt", "Says so if the agents disagreed."],
  5: ["Nothing made up", "A second LLaMA-3 finds no unsupported claim."],
};

/** For a review passed to a person because the agents split, a hand-off note replaces the star explanation. */
export function HandoffNote({ res, onRun }) {
  const q = (ps) => ps.slice(0, 3).map((p) => `“${p.display}”`).join(", ");
  return (
    <div className="card final" style={{ marginTop: 18 }}>
      <div className="lab" style={{ marginBottom: 6 }}>Hand-off note · no star is explained, because none is given</div>
      <p style={{ margin: 0, fontSize: 18, lineHeight: 1.5 }}>The agents strongly disagree: the LLaMA probe says {res.votes.probe}★, the Evidence model {res.votes.evidence}★ and DeBERTa {res.votes.deberta}★.
        {res.pos?.length ? <> Words such as {q(res.pos)} point to {res.star}★</> : null}{res.neg?.length ? <>, while {q(res.neg)} point the other way</> : null}.
        {" "}A split like this often means sarcasm or mixed feelings, so the review goes to a person instead of being explained as {res.star}★.</p>
      {onRun && <div style={{ marginTop: 12 }}><button className="btn inline" onClick={onRun}>Run the Explainer anyway</button>
        <span className="note"> It would explain the {res.star}★ the agents leaned to, which the system does not give.</span></div>}
    </div>
  );
}

export default function Explain({ res, state, run }) {
  const s = state || { drafts: [], critic: [], pack: null, done: false }; const phase = s.phase;
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
      {res.split && <HandoffNote res={res} />}
      <div style={{ marginTop: 18, display: "flex", gap: 14, alignItems: "center" }}>
        <button className="btn inline" onClick={run} disabled={s.running}>{s.running ? "Working…" : s.drafts.length ? "Run again" : res.split ? "Run the Explainer anyway" : "Write and check the explanation"}</button>
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
