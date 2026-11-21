import React, { useEffect, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { AGENTS, STARS, Highlight, argmax, ease, pct } from "./ui.jsx";
import Explainability from "./WalkExplain.jsx";

/* A guided walkthrough of one rating, using this review's real numbers.
   "Scores" are a model's raw outputs before softmax. For every softmax model, centred log-probabilities equal its
   centred raw scores exactly, so they are recovered here from the probabilities the API returns. */
const EPS = 1e-6;
const scores = (p) => { const l = p.map((v) => Math.log(Math.max(v, EPS))); const m = l.reduce((a, b) => a + b, 0) / l.length; return l.map((v) => v - m); };
const sg = (v, d = 1) => `${v >= 0.5 * 10 ** -d ? "+" : v <= -0.5 * 10 ** -d ? "−" : ""}${Math.abs(v).toFixed(d)}`;
export const D = (i, base = 0) => ({ initial: { opacity: 0, y: 16, filter: "blur(6px)" }, animate: { opacity: 1, y: 0, filter: "blur(0px)" }, transition: { duration: .8, ease, delay: base + i * .3 } });

/* ------------------------------------------------------------------ building blocks */
export const Box = ({ title, sub, i, hi, color, children, w }) => (
  <motion.div className="wbox" {...D(i)} style={{ borderColor: color || undefined, background: hi ? "#2C5AA0" : undefined, color: hi ? "white" : undefined, flex: w || 1 }}>
    <b style={{ color: hi ? "white" : undefined }}>{title}</b>{sub && <span>{sub}</span>}{children}
  </motion.div>
);
export const Arr = ({ i, label }) => <motion.div className="warr" {...D(i)}>{label && <small>{label}</small>}→</motion.div>;

function ScoreBars({ s, color, delay = 0 }) {
  const M = Math.max(...s.map(Math.abs)) || 1;
  return (
    <div className="sbars">
      {s.map((v, i) => (
        <div key={i} className="sb">
          <span className="sbl">{i + 1}★</span>
          <div className="sbt"><div className="sbz" />
            <motion.div className="sbf" style={{ background: color, [v >= 0 ? "left" : "right"]: "50%" }} initial={{ width: 0 }} animate={{ width: `${(Math.abs(v) / M) * 50}%` }} transition={{ duration: .6, ease, delay: delay + i * .05 }} />
          </div>
          <span className="sbv">{sg(v)}</span>
        </div>
      ))}
    </div>
  );
}

function ProbBars({ p, color = "#2C5AA0", delay = 0 }) {
  const top = argmax(p);
  return (
    <div className="pbars">
      {p.map((v, i) => (
        <div key={i} className="pb">
          <span className="sbl">{i + 1}★</span>
          <div className="pbt"><motion.div className="pbf" style={{ background: i === top ? color : "#C7C7CC" }} initial={{ width: 0 }} animate={{ width: `${v * 100}%` }} transition={{ duration: .7, ease, delay: delay + i * .05 }} /></div>
          <span className="sbv" style={{ fontWeight: i === top ? 800 : 500, color: i === top ? color : undefined }}>{pct(v)}</span>
        </div>
      ))}
    </div>
  );
}

/** Model inside -> scores -> softmax -> probabilities, revealed left to right. */
function Pipe({ inside, p, color, start = 0 }) {
  return (
    <div className="wpipe">
      {inside}
      <Arr i={start} />
      <motion.div className="wbox" {...D(start)} style={{ flex: 1.3 }}><b>Raw scores</b><span>one per star, before softmax</span><ScoreBars s={scores(p)} color={color} delay={.35 * start + .2} /></motion.div>
      <Arr i={start + 1} label="softmax" />
      <motion.div className="wbox" {...D(start + 1)} style={{ flex: 1.3 }}><b>Probabilities</b><span>add up to 100%</span><ProbBars p={p} color={color} delay={.35 * (start + 1) + .2} /></motion.div>
    </div>
  );
}

/* ------------------------------------------------------------------ slides */
function slides(res, inp, X) {
  const p1s = argmax(res.p1) + 1, conf = Math.max(...res.P), ag = Object.keys(AGENTS);
  const tot = res.push ? STARS.map((_, s) => ag.reduce((a, k) => a + res.push[k][s], 0) + res.base[s]) : scores(res.P);
  const mx = Math.max(...tot), ex = tot.map((v) => Math.exp(v - mx)), sum = ex.reduce((a, b) => a + b, 0);
  const H = res.hidden; const hm = H ? Math.max(...H.sample.map(Math.abs)) : 1;
  return [
    { t: "A review comes in", s: "Phase 2 sends the same review to three different agents at the same time. None of them sees the others' answers.",
      body: (
        <div>
          <motion.div className="review" {...D(0)}><Highlight title={inp.title} text={inp.text} /></motion.div>
          <div className="wfan">{ag.map((k, i) => <motion.div key={k} className="wchip" {...D(i + 1)} style={{ borderColor: AGENTS[k].hex }}><span className="dot" style={{ background: AGENTS[k].hex }} />{AGENTS[k].name}</motion.div>)}</div>
        </div>) },
    { t: "Agent 1 · LLaMA probe", s: "LLaMA-3 reads the review. We stop it two-thirds of the way through (layer 20 of 32) and take its internal summary of the review: a list of numbers. A small trained classifier reads that list and scores each star.",
      body: (
        <Pipe color={AGENTS.probe.hex} start={3} p={res.agents.probe} inside={<>
          <Box i={0} title="LLaMA-3-8B" sub="reads prompt + review" color={AGENTS.probe.hex} w={.8} />
          <Arr i={1} />
          <Box i={1} title={`Hidden state`} sub={H ? `${H.n.toLocaleString()} numbers; first 64 shown` : "4,096 numbers"} w={1.2}>
            {H && <div className="hid">{H.sample.map((v, i) => <motion.i key={i} style={{ background: v >= 0 ? "#2C5AA0" : "#A93636", opacity: .25 + .75 * Math.abs(v) / hm }} initial={{ scaleY: 0 }} animate={{ scaleY: 1 }} transition={{ delay: .5 + i * .012 }} />)}</div>}
          </Box>
          <Arr i={2} />
          <Box i={2} title="Probe" sub="logistic regression, trained on 40K reviews" w={.8} />
        </>} />) },
    { t: "Agent 2 · Evidence model", s: "This agent splits the title and text into words and two-word phrases. Each phrase has a learned weight for each star; it adds them up into a score per star.",
      body: (
        <Pipe color={AGENTS.evidence.hex} start={2} p={res.agents.evidence} inside={<>
          <Box i={0} title={`Phrases, weighed for ${res.star}★`} sub="blue pushes up, red pushes down" w={1.6}>
            <div style={{ marginTop: 8 }}>
              {res.pos.map((x, i) => <motion.span key={"p" + i} className="chip for" {...D(i * .3, .3)}>{x.display} <small>{sg(x.weight, 2)}</small></motion.span>)}
              {res.neg.map((x, i) => <motion.span key={"n" + i} className="chip against" {...D(i * .3, 1.2)}>{x.display} <small>{sg(x.weight, 2)}</small></motion.span>)}
            </div>
          </Box>
          <Arr i={1} />
          <Box i={1} title="Add up" sub="TF-IDF + logistic regression" w={.6} />
        </>} />) },
    { t: "Agent 3 · DeBERTa-v3", s: "A language model trained (fine-tuned) to rate reviews. It reads the title and text together and scores each star directly.",
      body: (
        <Pipe color={AGENTS.deberta.hex} start={2} p={res.agents.deberta} inside={<>
          <Box i={0} title="Title + text" sub="cut into word pieces" color={AGENTS.deberta.hex} w={.8} />
          <Arr i={1} />
          <Box i={1} title="DeBERTa-v3" sub="fine-tuned on labelled reviews" w={.8} />
        </>} />) },
    { t: "The Reflector, part 1: line up the three opinions", s: "It takes the log of every probability. A star an agent is sure about becomes close to 0; a star it rules out becomes a large negative number. That gives 15 numbers.",
      body: (
        <table className="wt">
          <thead><tr><th />{STARS.map((s) => <th key={s}>{s}★</th>)}</tr></thead>
          <tbody>{ag.map((k, r) => (
            <motion.tr key={k} {...D(r)}>
              <td><span className="dot" style={{ background: AGENTS[k].hex }} /> {AGENTS[k].name}</td>
              {res.agents[k].map((v, i) => { const l = Math.log(Math.max(v, EPS)); return (
                <td key={i}><div className="cell" style={{ background: `rgba(44,90,160,${Math.max(.04, 1 + l / 6)})`, color: l > -2 ? "white" : "var(--ink)" }}>
                  <b>{l.toFixed(1)}</b><small>{pct(v)}</small></div></td>); })}
            </motion.tr>))}</tbody>
        </table>) },
    { t: "The Reflector, part 2: weigh them", s: "Each of the 15 numbers is multiplied by a weight the Reflector learned on reviews no agent trained on. Per agent, that gives a push toward or away from every star. Adding the pushes gives each star's total score.",
      body: res.push ? (
        <table className="wt">
          <thead><tr><th />{STARS.map((s) => <th key={s}>{s}★</th>)}</tr></thead>
          <tbody>
            {ag.map((k, r) => (
              <motion.tr key={k} {...D(r)}>
                <td><span className="dot" style={{ background: AGENTS[k].hex }} /> {AGENTS[k].name}</td>
                {res.push[k].map((v, i) => <td key={i}><div className="cell" style={{ background: v >= 0 ? `rgba(44,90,160,${Math.min(.85, .08 + Math.abs(v) / 4)})` : `rgba(169,54,54,${Math.min(.85, .08 + Math.abs(v) / 4)})`, color: Math.abs(v) > 1.6 ? "white" : "var(--ink)" }}><b>{sg(v)}</b></div></td>)}
              </motion.tr>))}
            <motion.tr {...D(3)}><td className="note">Starting bias</td>{res.base.map((v, i) => <td key={i} className="note" style={{ textAlign: "center" }}>{sg(v)}</td>)}</motion.tr>
            <motion.tr {...D(4)} className="sum"><td><b>Total score</b></td>{tot.map((v, i) => <td key={i}><div className={`cell tot${i === res.star - 1 ? " win" : ""}`}><b>{sg(v)}</b></div></td>)}</motion.tr>
          </tbody>
        </table>) : <p className="sub">Weights are not available for a recorded result.</p> },
    { t: "The Reflector, part 3: scores become probabilities", s: "Softmax: raise e to each score, then divide by the sum so the five numbers add up to 100%. Bigger gaps between scores give a more confident answer.",
      body: (
        <table className="wt">
          <thead><tr><th>Star</th><th>Total score</th><th>exp(score), scaled</th><th>÷ sum</th><th>Probability</th></tr></thead>
          <tbody>{STARS.map((s, i) => (
            <motion.tr key={s} {...D(i * .5)} className={i === res.star - 1 ? "winrow" : ""}>
              <td><b>{s}★</b></td><td>{sg(tot[i], 2)}</td><td>{ex[i].toFixed(3)}</td><td>{ex[i].toFixed(3)} ÷ {sum.toFixed(3)}</td>
              <td style={{ width: "34%" }}><div className="pbt"><motion.div className="pbf" style={{ background: i === res.star - 1 ? "#2C5AA0" : "#C7C7CC" }} initial={{ width: 0 }} animate={{ width: `${res.P[i] * 100}%` }} transition={{ duration: .7, ease, delay: .3 + i * .2 }} /></div>
                <span className="sbv" style={{ marginLeft: 8 }}>{pct(res.P[i], 1)}</span></td>
            </motion.tr>))}</tbody>
        </table>) },
    { t: `The star: ${res.star}★`, s: "The system picks the star with the highest probability. That probability is its confidence.",
      body: (
        <div className="wbig">
          <motion.div initial={{ scale: .6, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ type: "spring", stiffness: 200, damping: 16 }} className="wstar">{res.star}★</motion.div>
          <motion.div {...D(1)} className="sub" style={{ fontSize: 20 }}>{pct(conf)} confident · {res.agreed ? "all three agents agreed" : "the agents disagreed; the Reflector settled it"}</motion.div>
        </div>) },
    { t: "Hand-off: answer, give a range, or ask a person", s: "The confidence is compared with a bar fixed on development data before testing. Above the bar it answers alone. Below it, it gives a narrow range of stars if one is likely enough, or passes the review to a person.",
      body: (
        <div style={{ padding: "10px 0" }}>
          {res.htau != null && <div className="gauge" style={{ height: 22 }}>
            <motion.div className="fill" style={{ background: res.decision === "auto" ? "var(--blue)" : "var(--grey)" }} initial={{ width: 0 }} animate={{ width: `${conf * 100}%` }} transition={{ duration: 1.2, ease }} />
            <div className="tick" style={{ left: `${res.htau * 100}%` }} /><div className="tlab" style={{ left: `${res.htau * 100}%` }}>bar {pct(res.htau)}</div></div>}
          <motion.p {...D(2)} style={{ fontSize: 20 }}>
            {res.decision === "auto" ? <>Confidence {pct(conf)} is above the bar: it <b className="blue">answers {res.star}★</b> on its own.</>
              : res.decision === "range" ? <>Confidence {pct(conf)} is below the bar, so it answers with a range: <b className="blue">{res.range[0]}–{res.range[1]}★</b>. Ranges like this hold the true star about 90% of the time.</>
              : <>Confidence {pct(conf)} is below the bar and no narrow range is likely enough: it <b className="bad">passes the review to a person</b>.</>}
          </motion.p>
        </div>) },
    { t: "Phase 1, for comparison", s: "The reference method asks one fine-tuned LLaMA-3 for the star and reads its scores for the answers “1” to “5”. One model, text only, no checks.",
      body: (
        <div>
          <Pipe color="#8E8E93" start={2} p={res.p1} inside={<>
            <Box i={0} title="Prompt + review text" sub="no title" w={.8} />
            <Arr i={1} />
            <Box i={1} title="LLaMA-3-8B + LoRA" sub="trained on 2,000 reviews" w={.8} />
          </>} />
          <motion.p {...D(4)} className="sub" style={{ fontSize: 17, marginTop: 18 }}>Phase 1 says <b>{p1s}★</b> ({pct(Math.max(...res.p1))}); Phase 2 says <b className="blue">{res.star}★</b> ({pct(conf)}){inp.truth ? <>; the reviewer gave <b>{inp.truth}★</b>.</> : "."}</motion.p>
        </div>) },
    ...Explainability({ res, D, ...X }),
  ];
}

export default function Walk({ res, inp, expl, runExplain }) {
  const [R, setR] = useState(null);
  useEffect(() => { fetch("/api/results").then((r) => r.json()).then(setR).catch(() => {}); }, []);
  return <Slideshow S={slides(res, inp, { expl, runExplain, R })} reset={res} part="Part 1 · How it rates" />;
}

/** Slide deck with dots, keys, play and a progress bar. Slides: {t, s, body, part?}. */
export function Slideshow({ S, reset, part }) {
  const [i, setI] = useState(0); const [play, setPlay] = useState(false);
  const go = (n) => setI(Math.max(0, Math.min(S.length - 1, n)));
  useEffect(() => { setI(0); }, [reset]);
  useEffect(() => {
    const k = (e) => { if (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT") return; if (e.key === "ArrowRight") go(i + 1); if (e.key === "ArrowLeft") go(i - 1); };
    window.addEventListener("keydown", k); return () => window.removeEventListener("keydown", k);
  });
  useEffect(() => { if (!play) return; if (i === S.length - 1) { setPlay(false); return; } const t = setTimeout(() => go(i + 1), 7000); return () => clearTimeout(t); }, [play, i]);
  const s = S[i];
  return (
    <div style={{ marginTop: 18 }}>
      <div className="wnav">
        <div className="wdots">{S.map((x, j) => <React.Fragment key={j}>{j > 0 && x.part !== S[j - 1].part && <span className="wsep" />}<button className={`wdot${j === i ? " on" : j < i ? " done" : ""}`} onClick={() => go(j)} title={x.t} /></React.Fragment>)}</div>
        <span className="note">Step {i + 1} of {S.length} · use ← → keys</span>
        <div style={{ flex: 1 }} />
        <button className="lg" onClick={() => setPlay(!play)}>{play ? "❚❚ Pause" : "▶ Play"}</button>
        <button className="lg" onClick={() => go(i - 1)} disabled={i === 0}>← Back</button>
        <button className="btn inline" onClick={() => go(i + 1)} disabled={i === S.length - 1}>Next →</button>
      </div>
      <div className="wprog"><motion.div animate={{ width: `${((i + 1) / S.length) * 100}%` }} transition={{ duration: .5, ease }} /></div>
      <AnimatePresence mode="wait">
        <motion.div key={i} className="card wslide" initial={{ opacity: 0, x: 40, filter: "blur(8px)" }} animate={{ opacity: 1, x: 0, filter: "blur(0px)" }}
          exit={{ opacity: 0, x: -40, filter: "blur(8px)" }} transition={{ duration: .55, ease }}>
          <div className="lab">{s.part || part} · step {i + 1}</div>
          <h2>{s.t}</h2>
          <p className="sub wsub">{s.s}</p>
          {s.body}
        </motion.div>
      </AnimatePresence>
    </div>
  );
}
