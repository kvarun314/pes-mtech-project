import React from "react";
import { motion } from "framer-motion";
import { Arr, Box, D, Slideshow } from "./Walk.jsx";
import { AGENTS } from "./ui.jsx";

/* How both phases were built, from data to result. Every number is from results_frozen.md, LOG.md or an e23 .out file. */
const P0 = "The data", P1 = "Phase 1 · The reference method", P2 = "Phase 2 · Our agent system", P3 = "Wrap-up";

/** Left-to-right pipeline of boxes: [title, sub, colour?]. */
const Flow = ({ items, hi }) => (
  <div className="wpipe">{items.map(([t, s, c], i) => (
    <React.Fragment key={t}>{i > 0 && <Arr i={i} />}<Box i={i} title={t} sub={s} color={c} hi={hi === i} /></React.Fragment>))}</div>
);
/** Big-number tiles: [value, label]. */
const Facts = ({ items, start = 0 }) => (
  <div className="facts">{items.map(([v, l], i) => <motion.div key={l} className="fact" {...D(start + i)}><b>{v}</b><span>{l}</span></motion.div>)}</div>
);
const Note = ({ i = 3, children }) => <motion.p className="sub bnote" {...D(i)}>{children}</motion.p>;

const S = [
  { part: P0, t: "One dataset, split once", s: "Amazon Reviews 2023, Electronics. Each review has a title, text and the star the buyer gave. We split it once, before any experiment, and no review appears in two parts.",
    body: <>
      <Flow items={[["Training", "≈497K reviews; the models learn here"], ["Development", "dev 500 · dev2 2,000 · natural_dev 500; every choice is made here"], ["Balanced test", "1,000 reviews, 200 per star"], ["Natural test", "2,000 reviews in the real mix, 65% are 5★"]]} />
      <Note>The two test sets were read once, at the end. A balanced test means the star mix gives no free points; the natural test shows real-world use.</Note></> },

  { part: P1, t: "Phase 1: rebuild the reference paper", s: "Wang et al. (2025) fine-tune one LLaMA-3-8B to read a review and give the star. They report 92.6% accuracy on their own protocol. We rebuilt it with their settings to see what it really does.",
    body: <><Flow items={[["Review text", "no title"], ["LLaMA-3-8B + LoRA", "one model, one pass"], ["Star 1–5", "plus a justification"]]} hi={1} />
      <Note>Phase 1 is the baseline that Phase 2 must beat, measured on the same test sets.</Note></> },

  { part: P1, t: "Step 1 · Prepare the training reviews", s: "The paper cleans and balances its data before training. We built that pipeline and a plain balanced version, and let the dev set choose.",
    body: <>
      <Flow items={[["TextBlob filter", "drops reviews whose tone seems to contradict the star"], ["Stratified sample", "same number per star"], ["VGST", "prefers reviews with new words"], ["3★ doubled", "2,000 rows"]]} />
      <Facts start={4} items={[["52% / 67%", "of 1★ / 2★ reviews removed by the filter"], ["55.2% vs 52.2%", "dev exact: plain balanced vs paper pipeline"], ["Plain balanced", "chosen on dev: 2,000 reviews, 400 per star"]]} /></> },

  { part: P1, t: "Step 2 · Train with LoRA", s: "LoRA freezes the 8 billion weights and trains a few small extra matrices. The paper's Table 1 settings, run with MLX on this laptop with the base model in 4 bits.",
    body: <Facts items={[["rank 4 · α 64", "on query and value, last 3 layers (29–31)"], ["≈160K numbers", "trained, 0.002% of the model; adapter 640 KB"], ["lr 5e-5", "cosine schedule, 3 epochs"], ["batch 12", "3 × gradient accumulation 4 = 400 updates"], ["1,024 tokens", "maximum length"], ["≈2 h 15 min", "training time on the laptop"]]} /> },

  { part: P1, t: "Step 3 · Ask for the star", s: "The prompt gives a 1–5 rubric and the review. We read the model's scores for the answers “1” to “5” and apply softmax to get probabilities.",
    body: <><Flow items={[["Prompt + review", "rubric, no worked example"], ["LLaMA + LoRA", "next-token scores"], ["Scores for “1”…“5”", "softmax gives probabilities"], ["Star", "the highest"]]} />
      <Note>Our first rebuild put a worked example in the prompt. It hinted at the answer: a random example cost 7.7 points. So the final protocol uses no example (zero-shot) and no leak.</Note></> },

  { part: P1, t: "What Phase 1 showed", s: "The reference method reads the direction well but often misses the exact star, and nothing checks it.",
    body: <Facts items={[["54.0%", "exact star, balanced test"], ["93.7%", "within one star"], ["5 sentences", "distinct justifications for 150 reviews: a template"], ["0 checks", "one model, one pass; nobody catches a mistake"]]} /> },

  { part: P2, t: "The idea: trained models decide, the LLM explains", s: "We tried three times to let an LLM decide the star, and it broke more than it fixed. So three trained agents decide the star, LLaMA-3 explains it, and a Critic checks the explanation.",
    body: <Flow items={[["3 agents", "probe · Evidence · DeBERTa", AGENTS.probe.hex], ["Reflector", "combines them"], ["Hand-off", "answer, range or a person"], ["Explainer ⇄ Critic", "write, check, rewrite"]]} /> },

  { part: P2, t: "Agent 1 · LLaMA probe", s: "LLaMA-3 is not trained at all. It reads each review, and we keep its internal state at layer 20 of 32. A logistic regression learns to read the star from those numbers.",
    body: <><Flow items={[["40K reviews", "from the training split"], ["Frozen LLaMA-3-8B", "no LoRA, 4-bit, MLX", AGENTS.probe.hex], ["Layer 20 state", "4,096 numbers per review"], ["Scale + logistic regression", "learns star from state"]]} />
      <Facts start={4} items={[["L16 .622 · L20 .628 · L24 .584", "dev exact by layer; 20 chosen"], ["+0.3", "from 120K reviews instead of 40K: saturated"]]} /></> },

  { part: P2, t: "Agent 2 · Evidence model", s: "It counts words and two-word phrases in the title and text (TF-IDF) and learns a weight for each phrase and star. Fast, transparent, and the source of the quotes used later.",
    body: <><Flow items={[["≈497K reviews", "the whole training split"], ["TF-IDF", "words + 2-word phrases", AGENTS.evidence.hex], ["Logistic regression", "a weight per phrase per star"]]} />
      <Note>It is the only agent whose reasons can be read directly, so its strongest phrases become the evidence for the explanation.</Note></> },

  { part: P2, t: "Agent 3 · DeBERTa-v3", s: "A smaller language model built for understanding text. We fine-tuned all of it to rate reviews, on a Colab GPU.",
    body: <><Flow items={[["100K reviews", "20K per star"], ["deberta-v3-base", "title + text, up to 256 tokens", AGENTS.deberta.hex], ["5-star classifier", "fine-tuned end to end"]]} />
      <Facts start={4} items={[["lr 2e-5 · batch 32", "2 epochs, 6% warm-up"], ["dev picks", "the better epoch; nothing tuned on test"], ["+2.3 points", "natural test, stable over 3 seeds"]]} /></> },

  { part: P2, t: "The Reflector learns whom to trust", s: "Each agent rates the 2,000 dev2 reviews, which none of them trained on. A logistic regression learns how much to trust each agent for each star.",
    body: <><Flow items={[["dev2 · 2,000", "unseen by the agents"], ["3 × 5 probabilities", "log taken: 15 numbers"], ["Logistic regression", "C = 1, 15 × 5 weights"], ["Final star", "and its confidence"]]} />
      <Facts start={5} items={[["Short reviews", "the Evidence model is best"], ["Long reviews", "the LLaMA probe is best"], ["58–62% vs 36–50%", "right when probe and Evidence disagree: combined vs either alone"]]} /></> },

  { part: P2, t: "Hand-off and star ranges", s: "Both are set on dev2 before testing. A confidence bar decides when the system answers alone. Below the bar, split conformal prediction gives a range of stars that holds the true star 90% of the time.",
    body: <Facts items={[["71%", "confidence bar, fixed on dev2"], ["56% · 92.8%", "natural test: answered alone, and right"], ["42% · 86%", "given a range, and the range held the star"], ["1.6%", "passed to a person"]]} /> },

  { part: P2, t: "Explainer and Critic: prompts and rules, no training", s: "Llama-3-8B-Instruct (4-bit, MLX) writes one or two sentences from the evidence pack. The Critic runs four fixed rules and one LLM judge, and sends a failed draft back, up to four times.",
    body: <><Flow items={[["Evidence pack", "star, confidence, phrases"], ["Explainer", "Llama-3-8B-Instruct"], ["Critic", "4 rules + LLM judge"], ["Rewrite", "with the Critic's instruction"]]} />
      <Facts start={5} items={[["62–64% → 91%", "explanations passing all checks, before and after the loop"], ["0", "drafts made worse"], ["2.9", "LLM calls per review"]]} /></> },

  { part: P2, t: "What we tried and dropped", s: "Every idea had a pass rule written before it ran, checked on development data only. These failed it.",
    body: <div className="drops">{[
      ["LLM decides the star", "fixed 76, broke 89 reviews on dev2"], ["Product-photo agent", "constant output, no signal"], ["More views of the text", "MiniLM, character n-grams, NB-SVM: no gain"],
      ["Bigger probe", "120K reviews: +0.3, saturated"], ["Gradient-boosted Reflector", "overfits: .621 vs .642"], ["Second LLM judge", "rejects almost everything"],
    ].map(([t, w], i) => <motion.div key={t} className="drop" {...D(i * .5)}><b>{t}</b><span>{w}</span></motion.div>)}</div> },

  { part: P3, t: "How we kept it honest", s: "The rules that make the numbers believable.",
    body: <Facts items={[["Written first", "every pass rule is in LOG.md before the run"], ["Dev only", "choices never look at the test sets"], ["Read once", "test sets scored at the end"], ["Traceable", "every number comes from a saved output file"]]} /> },

  { part: P3, t: "End to end: what it bought", s: "Same test reviews, Phase 1 against Phase 2.",
    body: <Facts items={[["54.0% → 67.5%", "exact star, balanced test"], [".488 → .634", "macro-F1, natural test"], ["66.9% → 86.9%", "right when most sure, balanced"], ["≈91%", "explanations that pass every check"]]} /> },
];

export default function Build() { return <Slideshow S={S} part={P0} />; }
