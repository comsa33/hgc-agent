# Labeling Guide for Annotators

## What you are doing
Decide whether a model's answer (`predicted_answer`) is correct with respect to
the reference answer (`gold_answer`). Several people label the same rows
**independently**, so that agreement with an automatic LLM judge can be
measured.

## How to label
Fill in `your_label` with `yes` or `no`:

- `yes` -- the prediction conveys the reference answer.
- `no`  -- it does not, or it declines to answer.

Judge meaning, not wording. Different phrasing, extra detail, or a different
surface form is still `yes` when the substance matches. A prediction that
refuses or says it cannot determine the answer is `no`, even when refusing is
the sensible thing to do.

`justification` is optional; a short note helps when a row is a close call.

## Ground rules
- Label independently. Do not discuss rows with other annotators while
  labeling, and do not look at anyone else's file.
- The automatic judge's verdict is deliberately withheld, so there is nothing
  to agree or disagree with -- record your own reading.
- If a row is genuinely ambiguous, pick the reading you would defend and note
  why, rather than leaving it blank.

## What happens to your labels
Labels are released with the paper in anonymized form, identified only as
`annotator_2` and `annotator_3`. No information about you is collected or
published. The labels are used solely to compute inter-annotator agreement.
