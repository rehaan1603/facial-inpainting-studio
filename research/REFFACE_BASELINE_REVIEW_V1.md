# RefFaceInpainting reproduction and failure review

Completed 27 September 2026. This is an external author baseline on the project's development inputs, not reproduction of the paper's published benchmark scores and not a new model contribution.

## Implemented and verified

The official [generator and ArcFace source](https://github.com/WuyangLuo/RefFaceInpainting/tree/0f1ad75677cc8fae4ae14d878e4c6cfce9365f28) and [reference parser source](https://github.com/zllrunning/face-parsing.PyTorch/tree/d2e684cf1588b46145635e8fe7bcc29544e5537e) are pinned, unmodified local checkouts. All three author checkpoints were downloaded from their published links, hashed, loaded with restricted weight deserialization and matched strictly to their architectures. The wrapper loads only inference networks, not the author's training losses, discriminators or optimizers.

The author's `Trainer.test` function is isolated and executed on synthetic tensors as a check against the adapter. Both native outputs are bit-exact, with maximum difference zero. Reference parsing also ran on a synthetic image. This is an implementation check, not an accuracy experiment. The parser's redundant ImageNet initialization download is skipped before every parameter is replaced by the complete strict-loaded parser checkpoint. No author inference architecture was changed.

For the experiment, references alone are parsed. The generator receives observed RGB, a missing mask, a supplied reference, its predicted semantic labels and its author ArcFace embedding. It never receives a clean target, target parsing, target identity embedding or withheld gallery. Missing original RGB is replaced by a constant before downsampling, so arbitrary erased-region payload cannot leak through interpolation. The native 256 missing mask conservatively covers all painted pixels; final512 composition preserves every original outside-mask byte. Native256 generation followed by resizing is not native512 synthesis. Two preprocessing tests verify mask coverage, erased-payload invariance and exact visible-pixel preservation.

## Observed result

Sixteen predictions (four known development cases × four references separately) and 28 evaluation rows including twelve reused controls completed. All sixteen new image hashes differ. Every reconstructed output has both identity detections and all three gallery comparisons; three detector-failure events occur only in unchanged damaged controls. All 28 rows preserve known pixels exactly.

The frozen primary choice, reference0, has FaceNet **0.519579**, hole MAE **0.092307** and LPIPS **0.043791**. Matched seed17 ResShift has **0.726530**, **0.063200**, **0.025153**; the prior corrected ReF-LDM cascade has **0.776886**, **0.065539**, **0.027261**. These are four-identity descriptive means, not recognition accuracy percentages. The four prespecified Holm p-values are 0.5, with insufficient identity units for a superiority claim. The secondary reference choices are all reported; none is retrospectively selected as the primary result.

All four eight-panel comparison sheets were inspected, covering every new prediction:

| Case | Visible failure across reference choices |
|---|---|
| 1306 | Blurred/ghosted eyes, enlarged or smeared lips, changed gaze and loss of the target's teeth/open-mouth expression; reference choice changes lip texture without repairing the structure. |
| 2790 | Misaligned eye structure, nose/lip distortions and blurry feature boundaries. |
| 1043 | Ghosted eyes and smeared teeth/mouth; reconstructed geometry differs despite visible surrounding context. |
| 787 | Blurred eyes, nose and lips; all four reference choices retain visible reconstruction artifacts. |

A primary native 256 output was also inspected: artifacts already exist before final upsampling/composition. Exact pixel preservation prevents changes outside the mask; it does not make the replaced features correct. These are internal assessments, not blinded human ratings. The method remains a research comparator and is not promoted to the website.

## Component-gate diagnostic

Read-only forward hooks measured the author's style-control behavior on all sixteen saved conditions. The sixteen repeated native predictions match the saved native pixels exactly. They are repeat integrity/diagnostic forwards, not sixteen new examples or evidence of independent replication.

Among **163 component/layer events** with predicted missing-component pixels and nonzero reference texture available, **46** lose their entire texture contribution. Every suppressed event also has predicted component pixels outside the missing mask, matching the author's rule to suppress a component if any of it survives. The count combines layers, components and references; it is not 163 identities or independently sampled trials. The other 117 events are not fully suppressed. See `refface_component_diagnostic_v1.json`.

This identifies an inference behavior worth testing, **not a proven cause of the blur or a novel fix**. Masks are compared with the generator's predicted labels, not anatomical ground truth. A controlled change must test both the benefit of supplied texture and harm to partially surviving features, keep the same source model/weights/inputs, report every reference choice, and remain separate from this frozen baseline. It cannot simply claim that opening the gate improves accuracy.

## Research boundary and next decision

The [paper](https://arxiv.org/html/2303.07014v1) already introduces component-specific reference texture and identity control. It also discusses segmentation alignment and pose-related failures. Our local output differences do not invalidate its published results, because the masks, data and preprocessing differ; pretraining identity overlap is unknown. Code uses `1` for missing pixels, and the adapter follows the released code convention.

The next bounded hypothesis is whether the all-or-nothing component rule discards useful reference evidence for **partially missing** components. A prospective causal ablation against the unchanged rule, texture-disabled control and reference-shuffled control can reject that explanation before any training effort. Preserve surviving pixels, distinguish changes to identity from changes to expression, and avoid choosing the best output with clean-target scores. Even a positive four-person screen would require independent confirmation and a narrower prior-art distinction. Merely changing this rule is not established novelty.

The current evidence does not support training the gated fusion adapter, opening the eight reserved-final identities, claiming reliable unknown-client recovery, or declaring the project publication-ready. Additional access was not required for these author checkpoints. Broader independent data and blinded visual evaluation remain unresolved. Photos, per-person feature arrays and checkpoints stay local; only code, numerical evidence, hashes and reports enter GitHub.
