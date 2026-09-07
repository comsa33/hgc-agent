# LLM judge validation against human labels

Sampling: stratified 25 judge=yes / 25 judge=no per benchmark, seed 20260424.

Bootstrap: 10000 resamples for 95% CI on Cohen's kappa.

## bcp (n=50)

- Percent agreement: 100.0%
- Cohen's kappa: 1.000 (95% bootstrap CI [1.000, 1.000])
- Confusion matrix (rows=human, cols=judge):
    human=yes, judge=yes: 25 (TP)
    human=yes, judge=no : 0 (FN — judge missed a correct answer)
    human=no,  judge=yes: 0 (FP — judge accepted an incorrect answer)
    human=no,  judge=no : 25 (TN)
- Judge precision: 1.000, recall: 1.000, F1: 1.000

## qasper (n=50)

- Percent agreement: 100.0%
- Cohen's kappa: 1.000 (95% bootstrap CI [1.000, 1.000])
- Confusion matrix (rows=human, cols=judge):
    human=yes, judge=yes: 25 (TP)
    human=yes, judge=no : 0 (FN — judge missed a correct answer)
    human=no,  judge=yes: 0 (FP — judge accepted an incorrect answer)
    human=no,  judge=no : 25 (TN)
- Judge precision: 1.000, recall: 1.000, F1: 1.000

## financebench (n=50)

- Percent agreement: 100.0%
- Cohen's kappa: 1.000 (95% bootstrap CI [1.000, 1.000])
- Confusion matrix (rows=human, cols=judge):
    human=yes, judge=yes: 25 (TP)
    human=yes, judge=no : 0 (FN — judge missed a correct answer)
    human=no,  judge=yes: 0 (FP — judge accepted an incorrect answer)
    human=no,  judge=no : 25 (TN)
- Judge precision: 1.000, recall: 1.000, F1: 1.000

## qasper_rag (n=50)

- Percent agreement: 100.0%
- Cohen's kappa: 1.000 (95% bootstrap CI [1.000, 1.000])
- Confusion matrix (rows=human, cols=judge):
    human=yes, judge=yes: 25 (TP)
    human=yes, judge=no : 0 (FN — judge missed a correct answer)
    human=no,  judge=yes: 0 (FP — judge accepted an incorrect answer)
    human=no,  judge=no : 25 (TN)
- Judge precision: 1.000, recall: 1.000, F1: 1.000

## overall (n=200)

- Percent agreement: 100.0%
- Cohen's kappa: 1.000 (95% bootstrap CI [1.000, 1.000])
- Confusion matrix (rows=human, cols=judge):
    human=yes, judge=yes: 100 (TP)
    human=yes, judge=no : 0 (FN — judge missed a correct answer)
    human=no,  judge=yes: 0 (FP — judge accepted an incorrect answer)
    human=no,  judge=no : 100 (TN)
- Judge precision: 1.000, recall: 1.000, F1: 1.000

## Inter-annotator agreement (n=200)

Raters: annotator_1, annotator_2, annotator_3. Every annotator after the first labeled a blind copy of the sample, with the judge's verdict and the other annotators' labels withheld, so their judgements are independent.

- Fleiss' kappa across 3 human raters: 0.813
- Unanimous rows: 172/200 (86.0%)
- Judge matches the human majority: 193/200 (96.5%)
- Pairwise percent agreement and Cohen's kappa:
    judge vs annotator_1: 100.0% (200/200), kappa=1.000
    judge vs annotator_2: 91.0% (182/200), kappa=0.820
    judge vs annotator_3: 91.5% (183/200), kappa=0.830
    annotator_1 vs annotator_2: 91.0% (182/200), kappa=0.820
    annotator_1 vs annotator_3: 91.5% (183/200), kappa=0.830
    annotator_2 vs annotator_3: 89.5% (179/200), kappa=0.789
