# PathSnap Model Comparison Figures

Figures are generated from the saved aggregate CSV result files, not manually entered chart values.

- Figures 01–04 and 08 use `model_comparison.csv` (five-fold GroupKFold on training trips).
- Figures 05–06 use `student_test_model_comparison.csv` (10 student-test trips).
- Figure 07 compares observation-level Top-1 from both result files.

Interpretation: higher Top-1 accuracy and Macro F1 are better; lower Log Loss is better.
Training time is the total time reported by the five-fold model comparison experiment.
Student-test labels use the project's independent nearest-reference-arc matching and 50 m eligibility filter;
they should be described as development metrics, not fully sequence-aligned ground truth.
