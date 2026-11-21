import React, { useEffect, useRef, useState } from "react";
import { animate, motion, useInView } from "framer-motion";

export const AGENTS = {
  probe: { name: "LLaMA probe", color: "var(--c-probe)", hex: "#7FA8E8", what: "Reads the review with LLaMA-3 and takes the star from what the model understood (its hidden state)." },
  evidence: { name: "Evidence model", color: "var(--c-evidence)", hex: "#2C5AA0", what: "Weighs the words and short phrases in the title and text, as learned from labelled reviews." },
  deberta: { name: "DeBERTa-v3", color: "var(--c-deberta)", hex: "#1B3C6E", what: "A language model fine-tuned to rate reviews. Reads title and text together." },
};
export const STARS = [1, 2, 3, 4, 5];
export const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
export const argmax = (a) => a.indexOf(Math.max(...a));
export const ease = [0.22, 1, 0.36, 1];

/** Number that counts up when it scrolls into view. */
export function Num({ value, decimals = 0, suffix = "", prefix = "" }) {
  const ref = useRef(null); const inView = useInView(ref, { once: true }); const [v, setV] = useState(0);
  useEffect(() => {
    if (!inView) return;
    const c = animate(0, value, { duration: 1.1, ease, onUpdate: setV }); return () => c.stop();
  }, [inView, value]);
  return <span ref={ref} style={{ fontVariantNumeric: "tabular-nums" }}>{prefix}{v.toFixed(decimals)}{suffix}</span>;
}

export const Stars = ({ n, grey }) => <span className={`stars${grey ? " grey" : ""}`}>{"★".repeat(n)}{"☆".repeat(5 - n)}</span>;

export function Step({ n, title, children }) {
  return (
    <motion.div className="step" initial={{ opacity: 0, y: 14 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true, margin: "-60px" }} transition={{ duration: .5, ease }}>
      <div className="num">{n}</div><div><h3>{title}</h3><p>{children}</p></div>
    </motion.div>
  );
}

export const Reveal = ({ children, delay = 0, className = "", style }) => (
  <motion.div className={className} style={style} initial={{ opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true, margin: "-40px" }} transition={{ duration: .55, ease, delay }}>{children}</motion.div>
);

/** Five animated bars, one per star; the top star is coloured. */
export function Hist({ p, color = "var(--blue)", height = 120 }) {
  const top = argmax(p);
  return (
    <div className="hist" style={{ height }}>
      {p.map((v, i) => (
        <div className="hc" key={i} title={`${i + 1}★: ${pct(v, 1)}`}>
          <div className="hv">{pct(v)}</div>
          <motion.div className="hb" style={{ background: i === top ? color : "var(--grey)" }}
            initial={{ height: 0 }} whileInView={{ height: Math.max(3, v * (height - 40)) }} viewport={{ once: true }}
            transition={{ duration: .8, ease, delay: i * .06 }} />
          <div className="hl">{i + 1}★</div>
        </div>
      ))}
    </div>
  );
}

/** Review text with evidence phrases marked (first match of each, longest first, no overlaps). */
export function Highlight({ title, text, pos = [], neg = [] }) {
  const mark = (s) => {
    const items = [...pos.map((x) => ({ ...x, cls: "for" })), ...neg.map((x) => ({ ...x, cls: "against" }))].sort((a, b) => b.display.length - a.display.length);
    const spans = []; const low = s.toLowerCase();
    for (const it of items) {
      const i = low.indexOf(it.display.toLowerCase());
      if (i >= 0 && !spans.some((r) => i < r.e && i + it.display.length > r.s)) spans.push({ s: i, e: i + it.display.length, ...it });
    }
    spans.sort((a, b) => a.s - b.s); const out = []; let at = 0;
    spans.forEach((r, k) => {
      out.push(s.slice(at, r.s));
      out.push(<mark key={k} className={r.cls} title={`${r.cls === "for" ? "Supports" : "Argues against"} the star · weight ${r.weight > 0 ? "+" : ""}${r.weight.toFixed(2)}`}>{s.slice(r.s, r.e)}</mark>);
      at = r.e;
    });
    out.push(s.slice(at)); return out;
  };
  return <>{title ? <><b>{mark(title)}</b><br /></> : null}{mark(text)}</>;
}

/** Grouped vertical bars in SVG: groups x series, hover shows the value. */
export function Bars({ groups, series, values, colors, fmt = (v) => v.toFixed(1), max, h = 230 }) {
  const W = 520, H = h, pad = 28, gw = (W - pad * 2) / groups.length, bw = Math.min(64, (gw - 30) / series.length);
  const [hov, setHov] = useState(null); const M = max ?? Math.max(...values.flat()) * 1.15;
  return (
    <div className="chart rel">
      <svg viewBox={`0 0 ${W} ${H}`}>
        <line x1={pad} x2={W - pad} y1={H - 30} y2={H - 30} stroke="var(--rule)" />
        {groups.map((g, gi) => (
          <g key={g}>
            {series.map((s, si) => {
              const v = values[gi][si], bh = (v / M) * (H - 70), x = pad + gi * gw + (gw - bw * series.length - 8 * (series.length - 1)) / 2 + si * (bw + 8);
              return (
                <g key={s} onMouseEnter={() => setHov({ x: x + bw / 2, y: H - 30 - bh, t: `${s}: ${fmt(v)}` })} onMouseLeave={() => setHov(null)}>
                  <motion.rect x={x} width={bw} rx={7} fill={colors[si]} initial={{ height: 0, y: H - 30 }} whileInView={{ height: bh, y: H - 30 - bh }}
                    viewport={{ once: true }} transition={{ duration: .9, ease, delay: gi * .12 + si * .08 }} style={{ cursor: "pointer" }} />
                  <motion.text x={x + bw / 2} y={H - 38 - bh} textAnchor="middle" fontSize="14" fontWeight="700" fill={si === series.length - 1 ? "var(--blue)" : "var(--muted)"}
                    initial={{ opacity: 0 }} whileInView={{ opacity: 1 }} viewport={{ once: true }} transition={{ delay: .7 + gi * .12 }}>{fmt(v)}</motion.text>
                </g>
              );
            })}
            <text x={pad + gi * gw + gw / 2} y={H - 8} textAnchor="middle" fontSize="13.5" fill="var(--body)">{g}</text>
          </g>
        ))}
      </svg>
      {hov && <div className="tt" style={{ left: `${(hov.x / W) * 100}%`, top: `${(hov.y / H) * 100}%` }}>{hov.t}</div>}
    </div>
  );
}

export const Legend = ({ items }) => (
  <div className="legend">{items.map(([name, color]) => <span key={name} className="lg" style={{ cursor: "default" }}><span className="dot" style={{ background: color }} />{name}</span>)}</div>
);
