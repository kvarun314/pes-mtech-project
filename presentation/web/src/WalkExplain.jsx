import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Highlight, ease, pct } from "./ui.jsx";
import { CHECKS, HandoffNote, RangeLine } from "./Explain.jsx";

const PART = "Part 2 · How it explains";
const sg2 = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;

/** Text that writes itself in, then shows the evidence highlights. */
function Typed({ text, pos, neg, speed = 14 }) {
  const [n, setN] = useState(0);
  useEffect(() => { setN(0); const t = setInterval(() => setN((k) => (k >= text.length ? (clearInterval(t), k) : k + 2)), speed); return () => clearInterval(t); }, [text]);
  return n >= text.length ? <motion.span initial={{ opacity: .85 }} animate={{ opacity: 1 }}><Highlight text={text} pos={pos} neg={neg} /></motion.span>
    : <span>{text.slice(0, n)}<span className="caret" /></span>;
}

/** Live run status shown inside a slide while the Explainer and Critic work. */
function RunBox({ expl, runExplain, label, res }) {
  if (res?.split && !expl?.running && !expl?.drafts?.length && !expl?.error) return <HandoffNote res={res} onRun={runExplain} />;
  if (expl?.running) {
    const msg = { evidence: "Collecting the evidence…", explainer: `Explainer writing draft ${(expl.drafts?.length || 0) + 1}…`, critic: `Critic checking draft ${expl.drafts.length}…` }[expl.phase] || "Working…";
    return <div className="runbox"><span className="spin" /> {msg} <span className="note">Live on this laptop, about 15–60 s.</span></div>;
  }
  if (expl?.error) return <div className="warnbar">Live explanation failed: {expl.error}</div>;
  return <div className="runbox"><button className="btn inline" onClick={runExplain}>{label}</button><span className="note">Runs the Explainer and Critic live (about 15–60 s). The walkthrough fills in as they work.</span></div>;
}

function CheckRows({ c, D }) {
  const failed = new Set(c.failed || []); const judged = c.failed && [...failed].every((k) => k === 5);
  return (
    <div className="ckrows">
      {Object.entries(CHECKS).map(([k, [t, why]], j) => {
        k = +k; const st = !c.failed ? (c.feedback?.length ? "skip" : "pass") : failed.has(k) ? "fail" : k === 5 && !judged ? "skip" : "pass";
        const how = k === 5 ? "Checked by a second LLaMA-3 acting as a judge" : "Checked by a fixed rule, no AI involved";
        return (
          <motion.div key={k} className={`ckrow ${st}`} {...D(j * .7, .2)}>
            <span className="ckic">{st === "fail" ? "✗" : st === "skip" ? "–" : "✓"}</span>
            <div><b>{k} · {t}</b><span>{why}</span></div>
            <small>{st === "skip" ? "not run: rules 1–4 must pass first" : how}</small>
          </motion.div>
        );
      })}
    </div>
  );
}

export default function Explainability({ res, D, expl, runExplain, R }) {
  const e = expl || { drafts: [], critic: [] }; const pk = e.pack || { for: res.pos, against: res.neg };
  const n = e.drafts.length; const last = e.critic[e.critic.length - 1]; const passed = last && !last.feedback?.length;
  return [
    { part: PART, t: "Why explain, and what the explanation may use", s: "A star alone does not say why. So the system writes a short explanation, but only from an evidence pack: the chosen star, its confidence, whether the agents agreed, and the review's own strongest phrases. Every phrase comes straight from the review, so the explanation cannot drift away from it.",
      body: (
        <div className="grid g2" style={{ alignItems: "start" }}>
          <motion.div className="wbox" {...D(0)}>
            <b>The decision</b>
            <div style={{ fontSize: 44, fontWeight: 800, color: "var(--blue)", letterSpacing: "-0.03em", marginTop: 6 }}>{res.star}★</div>
            <span>{pct(Math.max(...res.P))} confident · {res.agreed ? "agents agreed" : "agents disagreed, so the explanation must say so"}</span>
          </motion.div>
          <motion.div className="wbox" {...D(1)}>
            <b>Evidence phrases from the Evidence model</b>
            <span>Weight = how strongly the phrase pushes toward {res.star}★ (blue) or away from it (red)</span>
            <div style={{ marginTop: 8 }}>
              {pk.for.map((x, i) => <motion.span key={"f" + i} className="chip for" {...D(i * .4, .8)}>{x.display} <small>{sg2(x.weight)}</small></motion.span>)}
              {pk.against.map((x, i) => <motion.span key={"a" + i} className="chip against" {...D(i * .4, 1.8)}>{x.display} <small>{sg2(x.weight)}</small></motion.span>)}
            </div>
          </motion.div>
        </div>) },
    { part: PART, t: "The Explainer writes a draft", s: "Llama-3-8B-Instruct gets the evidence pack and the review, with one instruction: explain the star to a non-expert in one or two sentences, using only this evidence.",
      body: n ? (
        <motion.div className="card final" {...D(0)}><div className="lab" style={{ marginBottom: 8 }}>Draft 1</div><Typed text={e.drafts[0]} pos={pk.for} neg={pk.against} /></motion.div>
      ) : <RunBox res={res} expl={expl} runExplain={runExplain} label="Write the explanation live" /> },
    { part: PART, t: "The Critic checks the draft", s: "Five checks. The first four are plain rules that compare the draft with the evidence and the review. The fifth asks a second LLaMA-3 whether anything is claimed that the review does not support; it runs only once the rules pass.",
      body: e.critic[0] ? (
        <div>
          <CheckRows c={e.critic[0]} D={D} />
          {e.critic[0].feedback?.map((f, j) => <motion.div key={j} className="ask" {...D(4 + j)}>Critic → Explainer: “{f}”</motion.div>)}
        </div>
      ) : <RunBox res={res} expl={expl} runExplain={runExplain} label="Write and check the explanation live" /> },
    { part: PART, t: n > 1 ? `Rewrite until it passes: ${n} drafts` : "Rewrite until it passes", s: "A failed check is not just a red mark. The Critic turns it into a precise instruction, for example “quote at least 2 of these phrases” or “you said 5 stars; the final rating is 4 stars”, and the Explainer rewrites. At most four rewrites.",
      body: !n ? <RunBox res={res} expl={expl} runExplain={runExplain} label="Write and check the explanation live" />
        : n === 1 ? (
          <motion.div className="wbox" {...D(0)}>
            <b>{passed ? "Draft 1 passed every check, so no rewrite was needed." : "Waiting for the Critic…"}</b>
            <span>On other reviews the loop does real work: across the 3,000 test reviews only about 6 in 10 first drafts pass, and after rewriting about 9 in 10 do.</span>
          </motion.div>)
        : (
          <div className="rewrites">
            {e.drafts.map((d, i) => (
              <motion.div key={i} className={`rw${e.critic[i] && !e.critic[i].feedback?.length ? " pass" : ""}`} {...D(i)}>
                <div className="lab">Draft {i + 1} {e.critic[i] ? (e.critic[i].feedback?.length ? "· sent back" : "· passed") : ""}</div>
                <div className="txt"><Highlight text={d} pos={pk.for} neg={pk.against} /></div>
                {e.critic[i]?.feedback?.map((f, j) => <div key={j} className="ask">Critic → Explainer: “{f}”</div>)}
              </motion.div>))}
          </div>) },
    { part: PART, t: "The final explanation", s: "This is what a shopper or seller would read. Each highlighted phrase is evidence quoted from the review; hover one to see its weight.",
      body: n && e.done ? (
        <div>
          <motion.div className="card final" style={{ fontSize: 22 }} {...D(0)}><RangeLine res={res} /><Highlight text={e.drafts[n - 1]} pos={pk.for} neg={pk.against} /></motion.div>
          <motion.p className="sub" {...D(1)} style={{ marginTop: 14 }}>{passed ? `Passed all five checks after ${n} draft${n > 1 ? "s" : ""}.` : "Best draft after four rewrites; it still fails a check, so it would be flagged."}{e.seconds != null ? ` Written and checked in ${e.seconds} s.` : ""}</motion.p>
        </div>
      ) : <RunBox res={res} expl={expl} runExplain={runExplain} label="Write and check the explanation live" /> },
    { part: PART, t: "How the Evidence model and the Critic help", s: "Evidence keeps the explanation tied to the review. The Critic catches the usual failures of a language model: wrong star, invented quotes, hiding doubt, unsupported claims.",
      body: R ? (
        <div className="grid g2" style={{ alignItems: "start" }}>
          <motion.div className="wbox" {...D(0)}>
            <b>Critic pass rate, 3,000 test reviews</b><span>share of explanations passing all five checks</span>
            {[["Balanced test", R.critic.bal], ["Natural test", R.critic.nat]].map(([l, [a, b]], i) => (
              <div key={l} style={{ marginTop: 14 }}>
                <div className="note">{l}</div>
                {[["First draft", a, "#C7C7CC"], ["After the Critic loop", b, "#2C5AA0"]].map(([t, v, c], j) => (
                  <div key={t} className="pb" style={{ marginTop: 6 }}>
                    <span style={{ width: 150, fontSize: 13 }}>{t}</span>
                    <div className="pbt" style={{ height: 18 }}><motion.div className="pbf" style={{ background: c }} initial={{ width: 0 }} animate={{ width: `${v}%` }} transition={{ duration: 1.2, ease, delay: .4 + i * .4 + j * .2 }} /></div>
                    <span className="sbv" style={{ width: 52, fontWeight: 700, color: j ? "var(--blue)" : undefined }}>{v.toFixed(1)}%</span>
                  </div>))}
              </div>))}
          </motion.div>
          <motion.div className="wbox" {...D(1)}>
            <b>Does a person agree?</b><span>blind check of 50 explanations by the author, without seeing the Critic's verdict</span>
            <div style={{ fontSize: 44, fontWeight: 800, color: "var(--blue)", marginTop: 8 }}>{R.human.toFixed(0)}%</div>
            <span>judged faithful to the review overall</span>
            {R.human_split?.agreed && (
              <div style={{ marginTop: 12 }}>
                {[["When the agents agreed", R.human_split.agreed], ["When they disagreed", R.human_split.disagreed]].map(([t, v], i) => (
                  <div key={t} className="pb" style={{ marginTop: 6 }}>
                    <span style={{ width: 170, fontSize: 13 }}>{t} (n {v.n})</span>
                    <div className="pbt" style={{ height: 18 }}><motion.div className="pbf" style={{ background: i ? "#A93636" : "#2C5AA0" }} initial={{ width: 0 }} animate={{ width: `${v.human}%` }} transition={{ duration: 1.2, ease, delay: 1 + i * .2 }} /></div>
                    <span className="sbv" style={{ width: 52, fontWeight: 700 }}>{v.human.toFixed(0)}%</span>
                  </div>))}
                <span style={{ display: "block", marginTop: 10 }}>Honest limit: when the agents disagree, the Critic still passes most drafts, but a person finds many of them unconvincing. That is where the explanation needs the most work.</span>
              </div>)}
          </motion.div>
        </div>
      ) : <span className="spin" /> },
  ];
}
