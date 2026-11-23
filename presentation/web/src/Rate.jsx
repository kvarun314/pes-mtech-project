import React, { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { AGENTS, STARS, Bars, Hist, Highlight, Num, Reveal, Stars, Step, argmax, ease, pct } from "./ui.jsx";

const softmax = (z) => { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((v) => v / s); };
const go = (id) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

/* ------------------------------------------------------------------ data flow diagram */
function Flow({ res, p1s }) {
  const [tip, setTip] = useState(null);
  const W = 1000, NW = 158, NH = 60;
  const ag = Object.keys(AGENTS), ay = { probe: 52, evidence: 140, deberta: 228 };
  const N = {
    review: { x: 92, y: 165, t: "Review", s: "title + text", tip: "The review goes to all three agents in parallel, and to Phase 1 (text only).", to: "s1" },
    ...Object.fromEntries(ag.map((k) => [k, { x: 360, y: ay[k], t: AGENTS[k].name, s: `${res.votes[k]}★ · ${pct(Math.max(...res.agents[k]))} sure`, c: AGENTS[k].hex, tip: AGENTS[k].what, to: "s2" }])),
    reflector: { x: 620, y: 140, t: "Reflector", s: `${res.star}★ · ${pct(Math.max(...res.P))}`, hi: true, tip: "Combines the three votes. Line thickness = how much each agent counted for this star.", to: "s3" },
    handoff: { x: 800, y: 140, t: "Hand-off", s: { auto: "answers alone", range: `range ${res.range[0]}–${res.range[1]}★`, person: "to a person" }[res.decision], tip: "Answers alone only when confident enough; otherwise a range or a person.", to: "s4" },
    answer: { x: 935, y: 140, t: "Answer", s: `${res.star}★`, hi: true, w: 110, tip: "What the system returns.", to: "s4" },
    p1: { x: 620, y: 318, t: "Phase 1", s: "LLaMA-3 + LoRA", grey: true, tip: "The reference method: one fine-tuned LLaMA-3 reads the text and answers in one step.", to: "s5" },
    p1a: { x: 935, y: 318, t: "Answer", s: `${p1s}★`, grey: true, w: 110, tip: "Phase 1's star, for comparison.", to: "s5" },
  };
  const cur = (a, b) => { const A = N[a], B = N[b], x1 = A.x + (A.w || NW) / 2, x2 = B.x - (B.w || NW) / 2, mx = (x1 + x2) / 2; return `M${x1},${A.y} C${mx},${A.y} ${mx},${B.y} ${x2},${B.y}`; };
  const sh = res.shares;
  const E = [
    ...ag.map((k, i) => ({ d: cur("review", k), c: AGENTS[k].hex, w: 2.5, delay: .2 + i * .08 })),
    ...ag.map((k, i) => ({ d: cur(k, "reflector"), c: AGENTS[k].hex, w: 2 + (sh[k] || 0) * 16, delay: .7 + i * .08, label: pct(sh[k] || 0), k })),
    { d: cur("reflector", "handoff"), c: "#2C5AA0", w: 5, delay: 1.2 }, { d: cur("handoff", "answer"), c: "#2C5AA0", w: 5, delay: 1.4 },
    { d: cur("review", "p1"), c: "#C7C7CC", w: 2.5, delay: .3 }, { d: cur("p1", "p1a"), c: "#C7C7CC", w: 2.5, delay: .9 },
  ];
  const order = ["review", ...ag, "reflector", "handoff", "answer", "p1", "p1a"];
  return (
    <div className="card flow">
      <div className="lab">How the data flows · hover a box, click to jump to its step</div>
      <svg viewBox={`0 0 ${W} 360`}>
        <text x={360} y={14} textAnchor="middle" fontSize="12" fill="var(--muted)" fontWeight="600" letterSpacing="1">PHASE 2 · OUR SYSTEM</text>
        <text x={620} y={286} textAnchor="middle" fontSize="12" fill="var(--muted)" fontWeight="600" letterSpacing="1">PHASE 1 · REFERENCE METHOD</text>
        {E.map((e, i) => (
          <g key={i}>
            <motion.path className="edge" d={e.d} stroke={e.c} strokeOpacity={.35} strokeWidth={e.w} initial={{ pathLength: 0 }} animate={{ pathLength: 1 }} transition={{ duration: .7, delay: e.delay, ease }} />
            <motion.path className="edge anim" d={e.d} stroke={e.c} strokeWidth={Math.max(2, e.w * .45)} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: e.delay + .7 }} />
          </g>
        ))}
        {ag.map((k) => (
          <motion.text key={k} x={505} y={(N[k].y + 140) / 2 - 6} textAnchor="middle" fontSize="12.5" fontWeight="700" fill={AGENTS[k].hex}
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.3 }}>{pct(sh[k] || 0)}</motion.text>
        ))}
        {order.map((id, i) => {
          const n = N[id], w = n.w || NW;
          return (
            <motion.g key={id} className="node" initial={{ opacity: 0, scale: .85 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: .1 + i * .07, duration: .45, ease }}
              style={{ transformOrigin: `${n.x}px ${n.y}px` }} whileHover={{ scale: 1.05 }} onMouseEnter={() => setTip(n.tip)} onMouseLeave={() => setTip(null)} onClick={() => go(n.to)}>
              <rect x={n.x - w / 2} y={n.y - NH / 2} width={w} height={NH} rx={14} fill={n.hi ? "#2C5AA0" : n.grey ? "#F5F5F7" : "white"}
                stroke={n.hi ? "none" : n.c || "#E5E5EA"} strokeWidth={n.c ? 2 : 1} />
              <text x={n.x} y={n.y - 4} textAnchor="middle" fontSize="15" fontWeight="700" fill={n.hi ? "white" : n.grey ? "#6E6E73" : "#1D1D1F"}>{n.t}</text>
              <text x={n.x} y={n.y + 16} textAnchor="middle" fontSize="13" fill={n.hi ? "#DCE8F7" : "#6E6E73"}>{n.s}</text>
            </motion.g>
          );
        })}
      </svg>
      <div className="tip note">{tip || "Edges into the Reflector are drawn thicker when that agent counted more for this review."}</div>
    </div>
  );
}

/* ------------------------------------------------------------------ Reflector pushes (interactive) */
function Pushes({ res }) {
  const ag = Object.keys(AGENTS); const [on, setOn] = useState(Object.fromEntries(ag.map((k) => [k, true])));
  const [hov, setHov] = useState(null);
  const act = ag.filter((k) => on[k]);
  const tot = STARS.map((_, s) => act.reduce((a, k) => a + res.push[k][s], 0) + res.base[s]);
  const P = softmax(tot); const star = argmax(P) + 1; const allOn = act.length === ag.length;
  const W = 560, H = 300, z = 142, pad = 30, gw = (W - pad * 2) / 5, bw = 46;
  const M = Math.max(...STARS.map((_, s) => Math.max(ag.reduce((a, k) => a + Math.max(0, res.push[k][s]), 0), ag.reduce((a, k) => a - Math.min(0, res.push[k][s]), 0)))) || 1;
  const sc = (v) => (v / M) * 100;
  const sg = (v) => `${v >= 0.05 ? "+" : v <= -0.05 ? "−" : ""}${Math.abs(v).toFixed(1)}`;
  const ranked = [...ag].sort((a, b) => res.push[b][res.star - 1] - res.push[a][res.star - 1]);
  return (
    <div className="grid" style={{ gridTemplateColumns: "1.5fr 1fr" }}>
      <div className="card">
        <div className="lab">How each agent pushes each star · click an agent to switch its push off</div>
        <div className="legend" style={{ marginTop: 10 }}>
          {ag.map((k) => <button key={k} className={`lg${on[k] ? "" : " off"}`} onClick={() => setOn({ ...on, [k]: !on[k] })}><span className="dot" style={{ background: AGENTS[k].hex }} />{AGENTS[k].name}</button>)}
        </div>
        <div className="chart rel">
          <svg viewBox={`0 0 ${W} ${H}`}>
            <line x1={pad} x2={W - pad} y1={z} y2={z} stroke="var(--muted)" strokeWidth="1" />
            <text x={pad} y={22} fontSize="12" fill="var(--muted)">toward the star ↑</text>
            <text x={pad} y={H - 6} fontSize="12" fill="var(--muted)">away from the star ↓</text>
            {STARS.map((s, si) => {
              let up = 0, dn = 0; const cx = pad + si * gw + gw / 2;
              return (
                <g key={s}>
                  {ag.map((k) => {
                    const v = on[k] ? res.push[k][si] : 0; const h = Math.abs(sc(v)); const y = v >= 0 ? z - sc(up) - h : z + sc(dn);
                    if (v >= 0) up += v; else dn -= v;
                    return (
                      <motion.rect key={k} x={cx - bw / 2} width={bw} fill={AGENTS[k].hex} rx={3} animate={{ y, height: h }} initial={{ y: z, height: 0 }}
                        transition={{ duration: .6, ease }} onMouseEnter={() => setHov({ x: cx, y, t: `${AGENTS[k].name} → ${s}★: ${v >= 0 ? "+" : ""}${v.toFixed(2)}` })}
                        onMouseLeave={() => setHov(null)} style={{ cursor: "pointer" }} />
                    );
                  })}
                  <motion.line x1={cx - bw / 2 - 6} x2={cx + bw / 2 + 6} stroke="var(--ink)" strokeWidth="3" animate={{ y1: z - sc(tot[si]), y2: z - sc(tot[si]) }} transition={{ duration: .6, ease }} />
                  <motion.text x={cx + bw / 2 + 10} fontSize="12.5" fontWeight="700" fill="var(--ink)" animate={{ y: z - sc(tot[si]) + 4 }} transition={{ duration: .6, ease }}>{sg(tot[si])}</motion.text>
                  <text x={cx} y={H - 22} textAnchor="middle" fontSize="14" fontWeight={s === star ? 800 : 500} fill={s === star ? "var(--blue)" : "var(--body)"}>{s}★</text>
                </g>
              );
            })}
          </svg>
          {hov && <div className="tt" style={{ left: `${(hov.x / W) * 100}%`, top: `${(hov.y / H) * 100}%` }}>{hov.t}</div>}
        </div>
        <div className="note">Each bar is an agent's push on that star's score (log-odds, from the Reflector's own weights). The black line is the total; the highest total wins.</div>
      </div>
      <div className="card">
        <div className="lab">{allOn ? "Final probability per star" : "Probability with those pushes switched off"}</div>
        <div className="hist" style={{ height: 150 }}>
          {P.map((v, i) => (
            <div className="hc" key={i}>
              <div className="hv">{pct(v)}</div>
              <motion.div className="hb" animate={{ height: Math.max(3, v * 110), background: i === star - 1 ? "#2C5AA0" : "#C7C7CC" }} initial={{ height: 0 }} transition={{ duration: .6, ease }} />
              <div className="hl">{i + 1}★</div>
            </div>
          ))}
        </div>
        <motion.div key={star + act.join()} initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} className="big blue" style={{ fontSize: 36, marginTop: 14 }}>
          {star}★ <span className="sub" style={{ fontSize: 15 }}>{pct(Math.max(...P))} confident</span>
        </motion.div>
        {allOn ? (
          <p className="sub"><b>Why {res.star}★:</b> {ranked.map((k) => `${AGENTS[k].name} ${sg(res.push[k][res.star - 1])}`).join(" · ")} toward {res.star}★.
            {res.agreed ? " All three agreed, so combining mostly raises confidence." : " They disagreed, so these pushes settle the star."}</p>
        ) : (
          <p className="sub">{star === res.star ? `Still ${star}★ without ${ag.filter((k) => !on[k]).map((k) => AGENTS[k].name).join(" and ")}.` : `Without ${ag.filter((k) => !on[k]).map((k) => AGENTS[k].name).join(" and ")}, the answer would move from ${res.star}★ to ${star}★.`}
            <br /><span className="note">Same Reflector with those agents' pushes removed; a what-if view, not a retrained model.</span></p>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ page */
export default function Rate({ res, inp }) {
  const p1s = argmax(res.p1) + 1, truth = inp.truth, conf = Math.max(...res.P);
  const verdict = (s) => !truth ? "" : s === truth ? " · ✓ matches the true star" : ` · off by ${Math.abs(s - truth)}`;
  const dec = {
    auto: ["okc", "Answers automatically", `Confidence ${pct(conf)} is above the bar of ${pct(res.htau || 0)}.`],
    range: ["warn", `Gives a range: ${res.range[0]}–${res.range[1]}★`, `Confidence ${pct(conf)} is below the bar, but the likely stars sit close together. Ranges like this hold the true star about 90% of the time.`],
    person: ["badc", "Hands it to a person", res.split ? `Confidence ${pct(conf)} is below the bar and the agents strongly disagree (probe ${res.votes.probe}★, Reflector ${res.star}★): possible sarcasm.` : `Confidence ${pct(conf)} is below the bar and the likely stars are far apart.`],
  }[res.decision];
  const fig = useMemo(() => STARS.map((_, i) => [res.p1[i], res.P[i]]), [res]);
  return (
    <div>
      <div className="grid g3" style={{ marginTop: 18 }}>
        <Reveal className="card tile main"><div className="lab">Phase 2 · our system</div><div className="big blue">{res.star}★</div>
          <div className="sub"><Stars n={res.star} /><br /><Num value={conf * 100} suffix="%" /> confident{verdict(res.star)}</div></Reveal>
        <Reveal className="card tile" delay={.08}><div className="lab">Phase 1 · reference method</div><div className="big">{p1s}★</div>
          <div className="sub"><Stars n={p1s} grey /><br /><Num value={Math.max(...res.p1) * 100} suffix="%" /> confident{verdict(p1s)}</div></Reveal>
        <Reveal className="card tile" delay={.16}><div className="lab">True star</div><div className="big">{truth ? `${truth}★` : "—"}</div>
          <div className="sub">{truth ? "what the reviewer gave" : "not known for your own review"}</div></Reveal>
      </div>
      {res.warning && <div className="warnbar">{res.warning}</div>}

      <div className="section"><Flow res={res} p1s={p1s} /></div>

      <div className="section" id="s1">
        <Step n={1} title="Find the evidence">The Evidence model marks the words that push toward the final star and the words that push against it. Hover a phrase for its weight.</Step>
        <Reveal className="review"><Highlight title={inp.title} text={inp.text} pos={res.pos} neg={res.neg} /></Reveal>
      </div>

      <div className="section" id="s2">
        <Step n={2} title="Three agents vote, independently">Each agent reads the review its own way and gives a probability for every star. None sees the others.</Step>
        <div className="grid g3">
          {Object.entries(res.agents).map(([k, p], i) => (
            <Reveal key={k} className="card agent" delay={i * .1}>
              <div className="name"><span className="dot" style={{ background: AGENTS[k].hex }} />{AGENTS[k].name}</div>
              <div className="what">{AGENTS[k].what}</div>
              <div className="vote">{res.votes[k]}★ <small><Num value={Math.max(...p) * 100} suffix="%" /> sure</small></div>
              <Hist p={p} color={AGENTS[k].hex} />
              {res.times?.[k] != null && <div className="note" style={{ marginTop: 8 }}>{res.times[k]} ms</div>}
            </Reveal>
          ))}
        </div>
        <p style={{ marginTop: 14 }}><span className={`chip ${res.agreed ? "okc" : "warn"}`}>{res.agreed ? "All three agree" : "The agents disagree"}</span>
          <span className="note">{Object.entries(res.votes).map(([k, v]) => `${AGENTS[k].name} ${v}★`).join(" · ")}</span></p>
      </div>

      <div className="section" id="s3">
        <Step n={3} title="The Reflector combines the votes">A logistic-regression model, trained on agent votes for reviews none of the agents trained on, learned how far to trust each agent for each star.
          Each agent pushes every star's score up or down; the scores become the final probabilities.</Step>
        {res.push ? <Pushes res={res} /> : <div className="card"><Hist p={res.P} /></div>}
      </div>

      <div className="section" id="s4">
        <Step n={4} title="The Hand-off agent decides who answers">If the system is confident enough it answers on its own. If not, it gives a narrow star range or passes the review to a person.</Step>
        <Reveal className="card">
          {res.htau != null && (
            <div className="gauge">
              <motion.div className="fill" style={{ background: res.decision === "auto" ? "var(--blue)" : "var(--grey)" }} initial={{ width: 0 }} whileInView={{ width: `${conf * 100}%` }} viewport={{ once: true }} transition={{ duration: 1, ease }} />
              <div className="tick" style={{ left: `${res.htau * 100}%` }} /><div className="tlab" style={{ left: `${res.htau * 100}%` }}>bar {pct(res.htau)}</div>
            </div>
          )}
          <p><span className={`chip ${dec[0]}`}>{dec[1]}</span> <span className="sub">{dec[2]}</span></p>
        </Reveal>
      </div>

      <div className="section" id="s5">
        <Step n={5} title="Compare with the reference method">Phase 1 is one LLaMA-3 fine-tuned with LoRA on 2,000 reviews. It reads only the text and answers in one step.</Step>
        <Reveal className="card">
          <div className="legend"><span className="lg" style={{ cursor: "default" }}><span className="dot" style={{ background: "#C7C7CC" }} />Phase 1</span>
            <span className="lg" style={{ cursor: "default" }}><span className="dot" style={{ background: "#2C5AA0" }} />Phase 2</span></div>
          <Bars groups={STARS.map((x) => `${x}★`)} series={["Phase 1", "Phase 2"]} values={fig.map(([x, y]) => [x * 100, y * 100])} colors={["#C7C7CC", "#2C5AA0"]} fmt={(v) => `${v.toFixed(0)}%`} max={100} h={250} />
        </Reveal>
      </div>
    </div>
  );
}
