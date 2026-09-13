# Learned context-transfer protocol v1

This protocol was fixed before the first numerical run. It evaluates the actual
predicted SDR, not context familiarity or the number of stable clusters. It is a
supervised composition diagnostic of the factor memory. It is not a replacement
architecture, a language benchmark, or evidence of general intelligence.

The fixed default experiment uses seeds 7, 17 and 42, alphabet `abcd`, five-symbol
strings, six encoder positions, 256 input/output bits and eight bits per
symbol-position code. Source context 0 is mapped to target context 1. The same
symbol-position primitives are rearranged into unseen whole strings.

For each seed, 128 unique training strings, 32 development strings and 64 test
strings are mutually disjoint. The four repeated-symbol strings are training
anchors, guaranteeing that every primitive appears during training. All other
strings are allocated by seeded permutation of the finite 1,024-string universe.
Training uses three epochs of the saved order: 384 exposures, not 384 independent
examples. Development results are diagnostic. No parameter or threshold is
chosen from either held-out split in protocol v1.

The two memories use identical random encodings, 1,024 receptive points, 32
receptor bits, creation threshold 6, activation threshold 4, probation/stability
counts 3/6, output vote threshold 2 and at most 32 clusters per point. They differ
only in frequency versus coactivation consolidation; the latter retains at most
32 history entries and runs three normalization passes. Both use a pruning ratio
of 0.75; their boundary inequalities retain the respective algorithm semantics.

At inference, a predictor receives only the source SDR. The evaluator constructs
that example's target after prediction. The nearest-source control stores only
training source/target pairs, retrieves by Jaccard overlap, and breaks ties by
the saved training order. Zero and identity controls make output sparsity and
unchanged-input overlap visible. None of these controls becomes the factor core.

Every report saves raw training pairs and source, target and predicted active-bit
indices for every held-out case. Pooled bit precision, recall and F1, whole-vector
exact match, active counts and density can therefore be recomputed. A zero
denominator gives precision/recall/F1 zero; sparse zero predictions cannot earn
credit from the many correctly inactive bits. Per-seed values and all failures
remain visible; no successful-seed selection is permitted.

The first full default run completed all three seeds. Mean test bit F1 was
0.3616 for frequency consolidation, 0.3591 for coactivation, 0.7950 for nearest
source SDR retrieval, 0.3062 for identity and 0 for zero prediction. Both learned
memories had approximately 0.99 bit precision and 0.22 recall; neither produced
a fully correct held-out target SDR. The coactivation mechanism did not improve
this task at the fixed budget. Low recall describes incomplete output recovery;
this experiment alone does not isolate its cause. No settings were retuned in
response to these test results.

Timing covers model construction plus training for the learned memories and
source-SDR-to-prediction latency for each method. Shared encoding work is separate
from inference. NumPy payload measurements are lower bounds and exclude Python
objects, allocator overhead and peak working memory. The per-seed default budget
is 180 seconds, checked between training/prediction calls; use an outer process
timeout as well. A timed-out run is marked incomplete rather than presented as a
completed comparison.

Passing this test would establish transfer within one supplied encoder/operator
family. It would not show that the encoder was learned, that arbitrary operators
were discovered, that unique factors were recovered, or that natural-language
abilities exceed any GPT model. Any later parameter changes require a separately
labeled protocol and fresh confirmatory data. Reports explicitly distinguish the
frozen default from configuration overrides, including small smoke-test runs.
