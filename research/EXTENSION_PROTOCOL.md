# Training-budget and compositing follow-up

This stage continues the publication objective after the first-stage negative result. Its specification is written before evaluating the extended models. It preserves all original model and test artifacts.

Each of the six locally trained refiners continues from its saved optimizer, scaler and RNG state, from 1,500 to 6,000 total steps. Architecture, sampled data sequence, optimizer, learning rate and tuning examples are unchanged. This matches total budgets across generic and weighted training and the three seeds. It tests sensitivity to a fourfold training budget; 6,000 updates alone do not prove convergence.

Checkpoint selection retains the same tuning-loss criterion, including the initial best checkpoint. Each new checkpoint gets its own five-candidate threshold selection using FNR + 4 FPR on the same 48 tuning identity labels. Assessment reuses the 48 development identity labels with matched area/location masks. No previously observed test cases are rerun or relabeled as fresh.

Two compositors share the same frozen-backbone candidate and effective mask E: hard replacement and `alpha = p` inside E (zero outside), where p is the learned mask score. Soft output is alpha × candidate + (1 − alpha) × observation. This is a learned probability-compositing control with no additional fitting or parameter selection. It is not a separately RGB-trained alpha network, an OSOR reproduction, or a calibrated probability guarantee.

All three seeds are compared on LaMa and predeclared seed 17 on ResShift. Per-case raw metrics, paired identity-bootstrap differences, accurate-mask results and learning curves must be retained. The learning-budget contrast uses hard compositing in both stages. The compositing contrast uses the same extended checkpoint, threshold and backbone output. Any contribution claim must survive these controls and a closer robust-mask comparison.

The publication plan now states the completed training and evaluation accurately. The initial broad conclusion about weighted refinement is limited to the initial budget until this extension is measured.
