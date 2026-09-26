# 📓 Notebook Walkthrough — `Malware_Ransomware_KNN_SVM_Lab.ipynb`

This document explains **every block of the notebook**, in order: what the code does, *why* it's written that way, and what any metric or abbreviation actually means. Read this alongside the notebook, or on its own if you just want to understand the methodology.

---

## 🔤 Glossary — terms and abbreviations used throughout

| Term | Meaning |
|---|---|
| **KNN** | K-Nearest Neighbors. To classify a new sample, look at its `k` closest points (by distance) in the training set and take a majority vote of their labels. |
| **SVM** | Support Vector Machine. Finds the boundary ("hyperplane") that separates classes with the widest possible margin. A **kernel** (`linear`, `rbf`, `poly`) controls how that boundary can bend to fit non-linear patterns. |
| **PE header** | Portable Executable header — metadata every Windows `.exe`/`.dll` file carries (section sizes, imports, entry point, etc.). CLaMP's features come from here. |
| **Cuckoo Sandbox** | A tool that runs a suspicious file in an isolated virtual machine and logs everything it does (API calls, files touched, registry keys changed). The ransomware dataset's features come from these logs. |
| **Train / Validation / Test split** | Train = data the model learns from. Validation = data used to *choose* settings (like `k` or the SVM kernel) without peeking at the test set. Test = data touched **only once**, at the very end, to report the final honest score. |
| **Stratified split** | A split that preserves each class's proportion in every subset (e.g., if 62% of samples are goodware, ~62% of train, val, and test are too). |
| **Leakage** | When information from validation/test data accidentally influences training (e.g., scaling using statistics computed over *all* the data instead of just train). Leakage makes results look better than they really are. |
| **Near-zero-variance column** | A feature that's (almost) the same value for every sample — it carries no useful signal and can be safely dropped. |
| **StandardScaler** | Rescales each feature to have mean 0 and standard deviation 1, so features with big raw numbers don't dominate distance-based models like KNN. |
| **LabelEncoder** | Converts text categories (e.g., `"UPX290..."`, `"NoPacker"`) into integer codes so a model can use them. |
| **Confusion matrix** | A table of predicted vs. actual labels — shows exactly how many samples were correctly/incorrectly classified, and *how* (which class got confused with which). |
| **True Positive (TP) / False Positive (FP) / True Negative (TN) / False Negative (FN)** | The four cells of a binary confusion matrix. TP = correctly flagged malware. FP = benign file wrongly flagged as malware ("false alarm"). FN = malware that slipped through undetected. TN = benign file correctly left alone. |
| **Accuracy** | `(TP + TN) / total`. The percent of predictions that were correct overall. Misleading when classes are imbalanced (e.g., 942 goodware vs. 4 `PGPCODER` samples). |
| **Precision** | `TP / (TP + FP)`. Of everything the model *flagged* as malicious, what fraction actually was? High precision = few false alarms. |
| **Recall** (a.k.a. sensitivity / TPR) | `TP / (TP + FN)`. Of everything that *was* actually malicious, what fraction did the model catch? High recall = few missed threats. |
| **F1-score** | The harmonic mean of precision and recall — a single number that balances both. Useful when you care about both false alarms and missed detections. |
| **Macro-F1 / macro-averaged metric** | For multiclass problems: compute the metric separately for *each class*, then average the per-class scores with equal weight. This stops a huge class (like `Goodware`, 942 samples) from drowning out a tiny one (like `PGPCODER`, 4 samples). |
| **ROC curve** | Receiver Operating Characteristic curve. Plots True Positive Rate (recall) against False Positive Rate as the model's decision threshold is swept from strict to lenient. A curve that hugs the top-left corner means the model separates the classes well. |
| **AUC / AUC-ROC** | Area Under the ROC Curve. A single number from 0 to 1 summarizing the whole curve. 0.5 = no better than random guessing; 1.0 = perfect separation. Unlike accuracy, it doesn't depend on picking one threshold, and it's much more robust to class imbalance. |
| **One-vs-rest (OvR)** | A trick for extending binary metrics (like ROC/AUC) to multiclass problems: for each class, treat "this class" vs. "everything else" as its own binary problem, then combine the results. |
| **decision_function** | An SVM method that returns a raw distance-from-the-boundary score (not a probability) for each sample — how confidently, and on which side, it was classified. Used to draw ROC curves and rank predictions. |
| **predict_proba** | Returns actual class probabilities (they sum to 1 across classes for each sample). KNN produces these naturally, from the neighbor vote. |
| **Softmax** | A formula that converts any set of raw scores into probability-like values that sum to 1. Used here to make SVM's `decision_function` scores usable in the same probability-based AUC calculation as KNN's. |
| **PCA** | Principal Component Analysis. Compresses many features down to 2 (or 3) new "components" that capture the most variation in the data, so it can be plotted and visually inspected. |
| **Random Forest** | An ensemble of decision trees, used here only as a quick, interpretable way to rank which original features matter most (its "feature importance"). It is not one of the two models being compared in this lab. |

---

## 🧩 Block 0 — Title & overview *(markdown)*

Explains the goal of the notebook (compare KNN vs. SVM on two cybersecurity datasets), summarizes both datasets' size/feature count, and cites the original data sources. This is the "abstract" — read it first to know what you're about to see.

---

## 🧩 Block 1 — Setup: imports & shared utility functions

**What it does:** Imports `pandas`/`numpy` (data handling), `matplotlib`/`seaborn` (plotting), and the specific `scikit-learn` pieces used everywhere below: `train_test_split`, `StandardScaler`, `LabelEncoder`, `KNeighborsClassifier`, `SVC` (SVM), `RandomForestClassifier`, `PCA`, and the metric functions (`accuracy_score`, `roc_auc_score`, etc.). Sets a fixed `RANDOM_STATE = 42` so every run is reproducible.

**Why:** Doing all imports once at the top, rather than scattering them through the notebook, keeps the rest of the code short and makes dependencies obvious at a glance.

### `safe_stratify()` and friends

**What it does:** Defines four reusable helper functions used identically by *both* datasets later:
- `safe_stratify(y)` — returns `y` if every class has ≥2 members (so `train_test_split(..., stratify=y)` will work), otherwise returns `None` so the split silently falls back to a plain random split. This matters because the ransomware dataset has a family (`PGPCODER`) with only 4 samples total.
- `drop_low_variance(X_train, others, threshold)` — computes each column's standard deviation **using `X_train` only**, drops any column below the threshold, and applies the *same* drop to whatever other sets (`others`, e.g. validation/test) are passed in.
- `evaluate_binary(y_true, y_pred, y_score)` — returns a dictionary of accuracy, precision, recall, F1, and AUC-ROC for a binary task, in one call.
- `evaluate_multiclass(y_true, y_pred, y_score, labels)` — same idea, but with macro-averaged precision/recall/F1 and macro-averaged one-vs-rest AUC, plus a `try/except` because AUC can be mathematically undefined if a rare class happens to have zero samples in a given split.

**Why:** Writing these once and reusing them for both CLaMP and the Ransomware dataset guarantees every model is scored **the exact same way** — no risk of accidentally using a different formula or threshold for one dataset vs. the other.

---

# 🧪 TASK 1 — CLaMP (Malware Detection)

## Block 1.1 — Load & inspect the data

**What it does:** Reads `ClaMP_Integrated-5184.csv` into a DataFrame, fills any missing values with 0, prints its shape (5,210 rows × 70 columns) and the first few rows, then plots the class balance (benign vs. malware) as a bar chart and a pie chart.

**Why:** Always look at your data before modeling it — confirms the file loaded correctly, shows whether the classes are balanced (roughly 47% benign / 53% malware here, which is healthy — no extreme imbalance to correct for).

## Block 1.2 — Encoding & class balance

**What it does:** Loops through every column; any column that's text (only `packer_type` — the name of the packer/compressor used on the file, e.g. `"NoPacker"`, `"UPX290..."`) gets converted to integer codes via `LabelEncoder`. Splits the DataFrame into `X_clamp` (features) and `y_clamp` (the `class` label: 0 = benign, 1 = malware).

**Why:** Every classifier here needs pure numbers — text columns must be encoded first.

## Block 1.3 — Splitting without leakage

**What it does:** Splits the data twice: first 80% (`X_trainfull`) / 20% (`X_test`), then splits that 80% again into 80% train / 20% validation. Both splits use `stratify=safe_stratify(y)` to preserve the class ratio, and the same `RANDOM_STATE` for reproducibility. End result: ~64% train / ~16% validation / ~20% test.

**Why the three-way split:** The **validation** set exists so we can try different values of `k` or different SVM kernels and pick the best one **without ever looking at the test set**. If we tuned directly against the test set, our final "test accuracy" would be overly optimistic — we'd effectively be cheating by picking whichever setting happened to do best on the exact data we're using to report our final grade.

## Block 1.4 — Avoiding feature leakage: drop near-zero-variance columns

**What it does:** Calls `drop_low_variance()` from Block 1 with a threshold of `1e-4`, computing standard deviations from `X_train` only, then applying the same column drops to `X_val` and `X_test`.

**Why train-only statistics matter:** If we'd computed variance across the *full* dataset (including test), a column that happens to vary a lot in the test set — but not in train — might get kept when it shouldn't be (or vice versa), subtly leaking information about the test set's structure into a "training-time" decision. Computing everything from `X_train` alone keeps the test set genuinely unseen until the final evaluation.

## Block 1.5 — Scaling features

**What it does:** Fits a `StandardScaler` on `X_train` only (`fit_transform`), then applies that *same* fitted scaler to transform `X_val` and `X_test` (`.transform`, not `.fit_transform`).

**Why:** KNN measures raw Euclidean distance between points — a feature ranging 0–1,000,000 would completely dominate a feature ranging 0–1 unless everything is put on the same scale first. SVM (especially with `rbf`/`poly` kernels) also benefits from scaled features. Again, fitting the scaler only on train avoids leakage: the scaler's mean/std should reflect only what the model was "allowed" to see during training.

## Block 1.6 — KNN: choosing *k* on the validation set

**What it does:** Trains a `KNeighborsClassifier` for each `k` in `[3, 5, 7, 9, 11]`, computes each one's **AUC-ROC on the validation set**, and picks the `k` with the highest validation AUC. Plots a bar chart of validation AUC vs. `k`.

**Why AUC-ROC as the selection metric:** It doesn't require picking a hard 0.5 classification threshold up front, and it's insensitive to class imbalance — a good general-purpose way to compare models before locking in a final threshold-based decision.

## Block 1.7 — KNN: final test results

**What it does:** Refits `KNeighborsClassifier` with the chosen `best_k`, predicts on the **test set** (touched for the first time here), and reports accuracy, precision, recall, F1, and AUC-ROC via `evaluate_binary()`, plus scikit-learn's full `classification_report`.

**Why "fit once, evaluate once":** This is the one and only time the test set is used for this model — giving an honest, unbiased estimate of how KNN would perform on genuinely new files.

## Block 1.8 — SVM: choosing a kernel on the validation set

**What it does:** Same idea as Block 1.6, but for SVM: trains an `SVC` with `kernel='linear'`, `'rbf'`, and `'poly'` (degree 2), scores each on validation AUC-ROC using `decision_function` (SVM's raw margin score) instead of `predict_proba`, and picks the best kernel.

**Why `decision_function` instead of `predict_proba` here:** Requesting probability estimates from `SVC` (`probability=True`) is much slower (it runs extra internal cross-validation), and `roc_auc_score` only needs a score that *ranks* samples correctly — the raw decision-function margin does that job just as well, faster.

## Block 1.9 — SVM: final test results

**What it does:** Refits `SVC` with the chosen kernel, predicts on the test set, and reports the same five metrics via `evaluate_binary()`.

## Block 1.10 — Visualizing model performance

**What it does:** Two plots. First, side-by-side **confusion matrices** for KNN and SVM on the test set (showing exact TP/FP/TN/FN counts). Second, both models' **ROC curves** overlaid on one chart, with a diagonal "chance" line for reference.

**Why:** A confusion matrix shows *where* a model's errors land (e.g., is it missing malware, or crying wolf on benign files?) — something a single accuracy number hides. The ROC-curve overlay makes it visually obvious which model separates the classes better across all thresholds, not just the default one.

## Block 1.11 — CLaMP results at a glance

**What it does:** Puts both models' metric dictionaries into one small table, bar-charts them side by side, and displays a color-graded (heatmap-style) version of the table.

**Why:** A quick, skimmable summary before moving to Task 2.

## Bonus — feature importance & PCA view

**What it does:** Trains a `RandomForestClassifier` (a different, tree-based model, used only as a diagnostic tool here — not part of the KNN vs. SVM comparison) purely to rank which of the 59 remaining features it relies on most, and plots the top 15. Then runs PCA to compress the test set's features down to 2 dimensions and scatter-plots them, colored by what KNN predicted for each point.

**Why:** Feature importance gives a sense of *which PE-header fields actually drive the malware/benign decision* (useful for a security analyst, not just a data scientist). The PCA scatter plot is a sanity check — if the two predicted classes form visually distinct clusters even after being squashed into just 2 dimensions, that's a good sign the problem is genuinely learnable.

---

# 🦠 TASK 2 — Ransomware (Detection & Family Classification)

## Block 2.1 — Preparing raw features

**What it does:** The raw `RansomwareData.csv` has **no header row** (just numbers), so column names are parsed separately from `VariableNames.txt` (a semicolon-separated index-to-name mapping) and attached to the DataFrame. A dictionary `family_names` maps each numeric family ID (0–11) to its real name (`Goodware`, `CryptLocker`, `PGPCODER`, etc.).

**Why:** The original dataset authors shipped the feature names in a separate file rather than as a CSV header — this block reconstructs a normal, readable DataFrame from their format.

**What follows immediately (still Block 2.1):** Prints the shape (1,524 × 30,970), the binary label counts (942 goodware / 582 ransomware), and a full breakdown by family name — then a horizontal bar chart of sample counts per family, highlighting that `PGPCODER` is the rarest class (only 4 samples).

## Block 2.2 — Dropping constant columns

**What it does:** Separates the three "administrative" columns (`ID`, the binary `Label`, and the `Ransomware Family` ID) from the actual 30,966 behavioral feature columns, into `X_ransom_full`, `y_bin_full`, and `y_multi_full`. Then counts how many columns have exactly 1 unique value (constant → useless) vs. 2 unique values (a real binary signal).

**Why:** Because these are *binary* behavioral flags ("did this API get called: yes/no"), a huge fraction never fire at all across this specific sample of 1,524 programs — dropping those constant columns later (in Blocks 2.3/2.5) shrinks the feature space and speeds up training without losing any information.

## Block 2.3 — Binary task: Ransomware vs. Goodware

**What it does:** The same leakage-safe 80/20 → 80/20 stratified split pattern as Task 1 (Block 1.3), applied to `X_ransom_full` / `y_bin_full`, followed immediately by `drop_low_variance()` with a tighter threshold (`1e-6`, appropriate for 0/1 binary features rather than continuous ones) computed on train only.

**Note:** unlike CLaMP, no `StandardScaler` is applied here — the features are already binary (0/1) flags, so standardizing them isn't necessary the way it is for continuous PE-header numbers.

## Block 2.4 — Binary: KNN vs. SVM

**What it does:** Repeats the exact same "tune on validation AUC-ROC, then evaluate once on test" procedure from Blocks 1.6–1.9, but for the ransomware binary task. Converts the DataFrames to NumPy `float32` arrays first (saves memory, given ~24,000 remaining columns). Plots a bar chart comparing accuracy/AUC and an overlaid ROC curve, same as Block 1.10.

**Why the story changes here:** With CLaMP's 59 features, KNN and SVM performed almost identically. With ~24,000 features here, SVM pulls clearly ahead. This is the notebook's central illustration of the **curse of dimensionality** — in very high-dimensional spaces, the notion of "nearest neighbor" that KNN relies on becomes less meaningful (most points end up roughly equidistant from each other), while SVM's margin-based boundary stays effective.

## Block 2.5 — Multiclass task: 12 ransomware families

**What it does:** A *fresh* three-way split (train/val/test) of the same features, but this time `y` is `y_multi_full` — the 12-category family label instead of the binary one. Trains KNN across the same `k` grid and SVM across the same three kernels, but this time selects the best setting using **validation macro-F1** instead of AUC-ROC.

**Why macro-F1 instead of AUC-ROC here:** With 12 unbalanced classes (942 `Goodware` down to 4 `PGPCODER`), plain accuracy or a single AUC number would be dominated by how well the model does on the giant `Goodware` class alone, and could look great even while completely failing on the rare ransomware families — which are exactly the ones a security team cares most about catching. Macro-F1 forces every family, no matter how small, to count equally toward the score.

**Also computed here:** each model's predicted-probability matrix (`knn_multi_proba` from `predict_proba`, `svm_multi_score` → `softmax_rows()` for SVM, since `decision_function` scores don't naturally sum to 1 the way probabilities need to for the multiclass AUC calculation) — these feed into `evaluate_multiclass()` and the ROC-curve block that follows.

## Block 2.6 — Multiclass ROC curves (one-vs-rest)

**What it does:** For each of the 12 families, treats "this family" vs. "everything else" as its own mini binary problem, and draws that family's individual ROC curve — 12 curves per model, in two side-by-side panels (KNN vs. SVM). Skips a family entirely if it happens to have zero samples in that particular test split.

**Why:** A single "multiclass AUC" number hides *which specific families* are hard to detect. Plotting all 12 curves at once shows, at a glance, that `Goodware` (the largest, most-represented class) is usually cleanly separated, while rare families produce noisier, less confident curves — an honest picture of where the model is weakest.

## Block 2.7 — Ransomware results at a glance

**What it does:** Summarizes accuracy, AUC-ROC (macro, for the multiclass rows), and F1 (macro, for the multiclass rows) for all four Task-2 models (KNN-binary, SVM-binary, KNN-multiclass, SVM-multiclass) into one color-graded table.

---

# 📊 Combined Results & Wrap-Up

## Block — Saving combined results

**What it does:** A small `flatten()` helper turns each of the six metric dictionaries computed across the whole notebook (CLaMP KNN, CLaMP SVM, Ransomware-binary KNN, Ransomware-binary SVM, Ransomware-multiclass KNN, Ransomware-multiclass SVM) into one row each of a single DataFrame, tagged with `model`, `dataset`, and `task` columns, and writes it to `results_lab_knn_svm.csv`.

**Why:** One tidy CSV that captures every result from the whole notebook in a consistent format — easy to load elsewhere (a spreadsheet, another notebook, a dashboard) without re-running anything.

## Block — Key takeaways *(markdown)*

**What it does:** A plain-English summary of the six lessons the notebook demonstrates:
1. Always tune on validation, evaluate on test, exactly once.
2. Fit any preprocessing statistic (scaler, variance threshold) on train data only.
3. High dimensionality specifically hurts distance-based models (KNN) more than margin-based ones (SVM).
4. Use `safe_stratify()`-style guards so rare classes don't crash your split.
5. Pick your evaluation metric to match the problem — AUC-ROC for balanced binary tasks, macro-F1/macro-AUC when classes are imbalanced or numerous.
6. Both malware and ransomware detection are, underneath the security framing, ordinary supervised classification problems — the same `load → prep → tune → evaluate` recipe applies to both.

Closes with the full reference list for both datasets and the workshop guide that inspired this notebook's structure.
