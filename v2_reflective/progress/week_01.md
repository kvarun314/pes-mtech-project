# Week 1 (Oct 19-25) — progress (running note)
Day 0/1 (Oct 18): natural-prior dev set built and scored; five Reflector variants read out (all reported); fair prior-adjusted baselines added.
Finding: natural_test exact accuracy is dominated by the 67% five-star prior. Prior-adjusted Evidence alone reaches 76.3% exact (stack C2 79.0%), so exact gains of natural-prior stackers are mostly prior correction and cost minority recall (2* recall .12-.16) and balanced-eval accuracy. Honest metrics: macro-F1 and per-class recall. C0 (balanced-dev stacker) keeps macro-F1 .578 vs Evidence .541 on natural_test.
Routing (P2-1) ties C0 and lowers regressions; extra features (P2-3) add nothing.
Next: Oct 19 base-LLaMA scoring; Oct 21 CI of C2 vs prior-adjusted Evidence; Oct 22-24 adjudicator Gate 0.
