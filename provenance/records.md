# Registration and freeze records

SHA-256 prefixes of the registration records (specifications, decision rules and scripts) and freeze records (pinned inputs) for each analysis of the paper, with the number of fits. Note numbers refer to the Supplementary Information.

| Analysis | Registration (sha256 prefix) | Freeze (sha256 prefix) | Fits |
|---|---|---|---|
| PPC | b4cc4f5f | 057ff2c1 (experiment) | 8 × 196 + surface |
| Simulation 1 | 6f965efa | b81b82eb | 900 |
| Simulation 2 | e5bc35bb | b7176d1c | 1,260 |
| Readout law g | — | b9775f9d | — |
| V1 | ca8e047c (validator; g b9775f9d) | 4d93537d | 1,568 |
| V2 | 09af266b (validator v2) | 04b2d6f0 | 960 |
| Simulation 3 | d13020dc | 73c04618 | 380 |
| Simulation 4 | 1b916615 (g_sim ca8176a2) | b4a061e8 | 240 |
| Hierarchy analysis | 9b2fa8e5 (hierarchy 5ff96356; fit manifest 73efc97c) | — (no new fits) | 0 |
| Participant-mask analysis | 9f0db5dc | 0054730f | 190 |
| Simulation 5 | 727690c0 | a0b8dea0 | 240 |
| Simulation 6 | rules d7aa4cf0 (run 6206da01; analysis d752a2e0) | f90caf99 | 1,100 |
| Simulation 6b | rules 6e727148 (run c6aa6a76; analysis b0a80a93) | 4b21ad6e | 810 |
| Group-level FC correlation | 92a7c70b | — (frozen PPC fits; evaluation simulations regenerated with the frozen keys) | 0 |
| Group-level held-out comparison and ICA-FIX transport | 3ab324b5 | — (no new fits) | 0 |
| Spin test | 232657ea | — (no new fits) | 0 |
| Paired spin test | b288b663 | — (no new fits) | 0 |
| Participant-specific SC and 232 regions | specification 2cde268c; analysis 9fc0898c | a6ba6d80 | 2,288 |
| Implementation check against the authors' code | specification dfac543a; rules 4e8b0ae1 (wrapper 46799b12, runner 103ecf75, analysis a7a717f2) | — (inputs: repository archive f92681c4) | 24 MATLAB + 24 reimplementation |
| Released-configuration check | specification cd54e543; rules 2ea70d93 (script baff60c7) | 23f0151a | 760 |
| Tissue-pool check | specification cd54e543; rules d8b44e93 (script b5977771) | 8fbe1ef9 | 480 |
| Original hierarchy estimand | specification cd54e543; rules 5bf8618c (script 37677f47) | — (no new fits) | 0 |
| Filter cross-terms, operator-only baseline, random-placement null | specification cd54e543; rules fa64fde2 (script 9b9244d5) | — (no new fits) | 0 |
| Distributional implementation comparison | specification b95699c0; rules ae0df4fd (runner 49876783, wrapper 841dbf91) | — (inputs: repository archive f92681c4) | 96 MATLAB + 96 reimplementation |
| Frequency control | specification b95699c0 with amendments A1 3544b6f6 and A2 c54b9d9f; rules 3ad768d9 (script bd0acd8c) | b493521f | 1,140 |
| Closest-configuration check (K4) | specification cd54e543 with addendum 742ccfdb; rules b4cfa93e (script b6fc903c) | b59bec48 | 760 |
| Bifurcation-parameter check, authors' code | specification f01f5718; rules 038cf125 | — (inputs: repository archive f92681c4) | 240 MATLAB |
| Bifurcation-parameter check, K1a and K4a | specification f01f5718 with amendment A1 fbbce04f; rules 282c7410 (run) and 5fb74439 (analysis) | 0d9e65b9 | 1,140 |
| Bifurcation-parameter check, K3a | specification f01f5718; rules 0a45bd13 | a34a5c00 | 570 |
