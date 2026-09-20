# Block 1 Gate 1 result

Status: **FAILED — architectural stop condition remains active.**

This is a research result, not an infrastructure failure. The frozen Gate 1 run
completed source verification, corpus conversion, training, validation and held-out
test evaluation. The final predeclared quality gate rejected the model.

Run: GitHub Actions `Block1 Quality Gate`, run `35513028763`.

## Frozen test result

The held-out split contained 4 whole GUM academic/biography documents and 1152
gold mentions.

Mention extraction:

- learned exact-span precision: 0.4003
- learned exact-span recall: 0.5417
- learned exact-span F1: 0.4603
- train-surface baseline F1: 0.2677
- learned F1 delta: +0.1926
- structurally unsupported gold spans: 103 / 1152 = 8.94%

Coreference:

- validation could not find a link operating point with >= 10 evaluable decisions
  and >= 90% precision;
- therefore automatic learned linking remained disabled;
- held-out accepted learned links: 0;
- learned end-to-end pair-coreference F1: 0;
- learned oracle-mention pair-coreference F1: 0;
- nearest-identical-surface oracle-mention baseline F1: 0.2584.

The validation grid showed that the strongest learned setting with at least ten
accepted decisions still had only 1 correct decision out of 11 (9.09% precision),
and 10 of the 11 accepted links involved at least one wrong mention boundary.

## Gate decisions

- candidate representability <= 5% unsupported: **FAIL** (8.94%)
- mention F1 >= surface baseline: **PASS**
- validation link gate >= 90% precision with >= 10 decisions: **FAIL**
- held-out accepted-link precision >= 80% with >= 10 decisions: **FAIL**
- oracle-mention coreference F1 >= identical-surface baseline: **FAIL**

## Consequence

Gate 1 test labels are now open and this split is regression-only. Thresholds are
not weakened after observing the result.

The next implementation must be selected using train/validation evidence and then
tested on a separately frozen unseen split. Gate 2 is frozen in
`GUM_BLOCK1_GATE2_SELECTION.json` before the v2.1 changes are evaluated.
