import React, { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { Num, ease } from "./ui.jsx";

/** One result = one claim, two numbers, one reason. */
function Claim({ i, title, why, a, b, al = "Phase 1", bl = "Phase 2", dec = 1, suf = "%", max = 100, lower, worse, src }) {
  const d = b - a, good = worse ? false : lower ? d < 0 : d > 0;
  const fmtD = `${d > 0 ? "+" : "−"}${Math.abs(d).toFixed(dec)}${suf === "%" ? " pts" : ""}`;
  return (
    <motion.div className={`claim${worse ? " worse" : ""}`} initial={{ opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-40px" }} transition={{ duration: .7, ease }}>
      <div className="cnum">{String(i + 1).padStart(2, "0")}</div>
      <div className="ctext"><h3>{title}</h3><p>{why}</p>{src && <small>{src}</small>}</div>
      <div className="cvals">
        {[[al, a, "#C7C7CC"], [bl, b, worse ? "#A93636" : "#2C5AA0"]].map(([l, v, c], j) => (
          <div key={l} className="crow">
            <span className="cl">{l}</span>
            <div className="ctrack"><motion.div style={{ background: c }} initial={{ width: 0 }} whileInView={{ width: `${(v / max) * 100}%` }} viewport={{ once: true }} transition={{ duration: 1.1, ease, delay: .2 + j * .15 }} /></div>
            <b className={j ? (worse ? "bad" : "blue") : ""}><Num value={v} decimals={dec} suffix={suf === "%" ? "%" : ""} /></b>
          </div>
        ))}
        <span className={`delta ${good ? "up" : "down"}`}>{fmtD}{lower && !worse ? " (lower is better)" : ""}</span>
      </div>
    </motion.div>
  );
}

export default function Results() {
  const [d, setD] = useState(null); const [more, setMore] = useState(false);
  useEffect(() => { fetch("/api/results").then((r) => r.json()).then(setD); }, []);
  if (!d) return <div className="empty"><span className="spin" /> Loading the frozen results…</div>;
  const s = d.sure.eval;
  const C = [
    { title: "Gets the exact star more often", why: "Three agents read meaning, key words and wording; the Reflector learns which one to trust for each review. One model alone cannot do that.",
      a: d.exact.bal[0], b: d.exact.bal[1], src: "Balanced test, 1,000 reviews, 200 per star · p = 6.7e-14" },
    { title: "Fair to every star, not just 5★", why: "Macro-F1 scores each star equally, so the common 5★ cannot hide mistakes on the rare 2★ and 3★.",
      a: d.f1.nat[0], b: d.f1.nat[1], dec: 3, suf: "", max: 1, src: "Natural test, 2,000 reviews in their real mix" },
    { title: "Not fooled by negative words", why: "“Not bad at all” sounds negative to a single reader. The Evidence model weighs phrases, and the probe reads meaning, so happy reviews stay happy.",
      a: d.hard.bal[0], b: d.hard.bal[1], src: `Happy reviews that use negative words, balanced test (n ${d.hard.n[0]})` },
    { title: "Knows when it is sure", why: "Its confidence means something. When it answers alone it is right far more often, and the hard cases go to a person.",
      a: s.p1.acc_auto * 100, b: s.p2.acc_auto * 100, src: `Accuracy on the ${(s.p2.coverage * 100).toFixed(0)}% of balanced reviews each system is most sure about` },
    { title: "Steadier on messy text", why: "Remove all punctuation and capitals and Phase 1 changes its mind three times as often. Phase 2 has three readers to fall back on.",
      a: d.robust["No punctuation"][0], b: d.robust["No punctuation"][1], lower: true, max: 40, src: `Stars that changed, ${d.robust_n.toLocaleString()} balanced reviews` },
    { title: "Explanations get checked and fixed", why: "The Critic catches wrong stars, invented quotes and hidden doubt, and sends the draft back. 0 drafts were made worse.",
      a: d.critic.nat[0], b: d.critic.nat[1], al: "First draft", bl: "After Critic", src: `Natural test, all five checks · a person agreed on ${d.human.toFixed(0)}%` },
    { title: "One thing is worse: it needs the title", why: "Two agents read the title, so deleting it changes the star more often. Phase 1 never reads the title.",
      a: d.robust["No title"][0], b: d.robust["No title"][1], lower: true, worse: true, max: 40, src: `Stars that changed, ${d.robust_n.toLocaleString()} balanced reviews` },
  ];
  return (
    <div style={{ marginTop: 18 }}>
      <div className="rhead">
        <div><div className="lab">The result in one line</div>
          <h2>+{(d.exact.bal[1] - d.exact.bal[0]).toFixed(1)} points on the exact star, and every explanation is checked.</h2></div>
        <p className="sub">Phase 1 is the reference method (one fine-tuned LLaMA-3). Phase 2 is our agent system. Both are scored once on test reviews neither ever saw.</p>
      </div>
      <div className="claims">{C.map((c, i) => <Claim key={i} i={i} {...c} />)}</div>
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
      <footer>Sources: results_frozen.md, s1_selective.json, b2_report.json, q4_robustness(_1k).json, hard_cases.json, human_check.out (v2_reflective/e23/).</footer>
    </div>
  );
}
