# 🛡️ Malware & Ransomware Classification with KNN and SVM

A hands-on machine learning lab that detects malware and ransomware from real static and behavioral features — and shows, side by side, when **K-Nearest Neighbors** and **Support Vector Machines** each shine.

> Two real datasets. Two algorithms. Three tasks (malware detection, ransomware detection, ransomware family classification). One fully-executed notebook you can read start to finish without running a single cell — or re-run yourself in a few minutes.

---

## 📖 What is this?

This repo trains and evaluates classifiers on two public cybersecurity datasets:

| | CLaMP | Ransomware (EldeRan) |
|---|---|---|
| **What it captures** | Static features pulled from a file's PE (Portable Executable) header | Dynamic behavior recorded while the sample runs in a Cuckoo Sandbox (API calls, registry/file operations, dropped files, embedded strings) |
| **Samples** | 5,210 | 1,524 |
| **Features** | 69 | ~31,000 |
| **Task** | Malware vs. benign | Ransomware vs. goodware, **and** which of 12 ransomware families |

The notebook walks through the same disciplined pipeline for both datasets — load, clean, split, tune, evaluate — and ends with a head-to-head comparison of KNN vs. SVM on each.

**Why it's interesting:** on the small, low-dimensional CLaMP dataset the two algorithms are nearly tied. On the ransomware dataset's ~31,000 features, SVM pulls noticeably ahead of KNN — a clean, real-world illustration of the *curse of dimensionality*.

---

## ✨ Results

| Model | Dataset | Task | Accuracy | F1 | AUC-ROC |
|---|---|---|---:|---:|---:|
| KNN | CLaMP | Malware detection | 96.7% | 0.969 | 0.993 |
| SVM | CLaMP | Malware detection | 96.7% | 0.969 | 0.996 |
| KNN | Ransomware | Ransomware detection | 89.5% | 0.868 | 0.970 |
| SVM | Ransomware | Ransomware detection | **93.4%** | **0.919** | **0.993** |
| KNN | Ransomware | Family classification (12 classes) | 78.4% | 0.426 (macro) | 0.789 (macro) |
| SVM | Ransomware | Family classification (12 classes) | **86.2%** | **0.565 (macro)** | **0.814 (macro)** |

Full per-metric breakdown: [`results_lab_knn_svm.csv`](results_lab_knn_svm.csv)

The notebook also includes, per task: confusion matrices, ROC curves (including one-vs-rest curves for all 12 ransomware families), validation curves used to pick `k` and the SVM kernel, a feature-importance chart, and a PCA view of the decision space.

---

## 🚀 Quick start

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Set up a Python environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, this covers everything the notebook needs:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

### 3. Open the notebook

```bash
jupyter notebook Malware_Ransomware_KNN_SVM_Lab.ipynb
```

That's it — no downloads or path edits needed. Both datasets already live in `data/`, and every path in the notebook is relative to the notebook's own folder.

- **Just want to read it?** The notebook is committed *with outputs*, so GitHub renders every table and chart directly in the file preview — click on `Malware_Ransomware_KNN_SVM_Lab.ipynb` above and scroll.
- **Want to rerun it?** `Kernel → Restart & Run All` in Jupyter. Full execution takes **1–2 minutes** (the ransomware task trains on ~31,000 features, so SVM fitting is the slowest step).

---

## 🗂️ Repository structure

```
.
├── Malware_Ransomware_KNN_SVM_Lab.ipynb   # ⭐ the main notebook — start here
├── results_lab_knn_svm.csv                # combined metrics, one row per model × dataset × task
├── README.md
└── data/
    ├── ClaMP_Integrated-5184.csv          # CLaMP: 5,210 samples × 69 features + label
    ├── ClaMP_Raw-5184.csv                 # CLaMP: raw (non-derived) feature variant
    └── ransomware/
        ├── RansomwareData.csv             # 1,524 samples × 30,970 columns (no header row)
        ├── VariableNames.txt              # maps each CSV column index → feature name
        ├── Family Names ID.txt            # maps each ransomware family → numeric ID
        └── README.txt                     # original dataset documentation, from the authors
```

---

## 🔬 How the notebook is organized

The same six-stage pipeline runs once for **CLaMP** and once for **Ransomware**:

1. **Load** — read the raw CSV(s) into pandas
2. **Prep** — encode any categorical columns, then split 80/20 into train+val / test, then 80/20 again into train / val. A `safe_stratify()` helper keeps `train_test_split` from crashing on ransomware families with as few as 4 samples, falling back to a plain split only when it must
3. **Leakage guard** — drop near-zero-variance columns and fit `StandardScaler`, both computed on the **training split only**, then applied unchanged to validation and test
4. **Train & tune** — sweep KNN's `k` (3, 5, 7, 9, 11) and SVM's kernel (`linear`, `rbf`, `poly`), each selected using **validation-set** performance only — AUC-ROC for the binary tasks, macro-F1 for the 12-way family task (so the 4-sample `PGPCODER` family counts as much as the 942-sample `Goodware` class)
5. **Evaluate** — the tuned model is scored exactly once, on the untouched **test** set
6. **Compare & visualize** — confusion matrices, ROC curves, bar-chart comparisons of KNN vs. SVM, and a merged results table saved to `results_lab_knn_svm.csv`

---

## 📚 Dataset sources

**CLaMP** — static PE-header features
Kumar, A., Kuppusamy, K.S., Aghila, G. (2019). *A learning model to detect maliciousness of portable executable using integrated feature set.* Journal of King Saud University — Computer and Information Sciences.
Source repo: [urwithajit9/ClaMP](https://github.com/urwithajit9/ClaMP)

**Ransomware** — dynamic Cuckoo Sandbox behavioral features
Sgandurra, D., Muñoz-González, L., Mohsen, R., Lupu, E.C. (2016). *Automated Dynamic Analysis of Ransomware: Benefits, Limitations and Use for Detection.* arXiv:1609.03020.
Source repo: [rissgrouphub/ransomwaredataset2016](https://github.com/rissgrouphub/ransomwaredataset2016)

If you reuse either dataset, please cite the original authors above.

---

## ❓ FAQ / troubleshooting

**"The notebook is huge / GitHub says it can't render it."**
It's a normal `.ipynb` with 11 embedded chart images, so the file is a few hundred KB — GitHub's built-in notebook viewer should handle it fine. If it ever times out, download it and open it locally, or view it via [nbviewer](https://nbviewer.org/).

**"`RansomwareData.csv` is ~94 MB — will `git clone` be slow?"**
It's under GitHub's 100 MB hard limit, but it is large. If you plan to keep growing this repo, consider moving it to [Git LFS](https://git-lfs.com/):
```bash
git lfs install
git lfs track "data/ransomware/RansomwareData.csv"
git add .gitattributes data/ransomware/RansomwareData.csv
git commit -m "Track large dataset with Git LFS"
```

**"Can I use my own dataset with this pipeline?"**
Yes — the `safe_stratify()`, `drop_low_variance()`, and `evaluate_binary()` / `evaluate_multiclass()` helper functions near the top of the notebook are written generically. Swap in your own `X`/`y` and the tuning/evaluation cells below them should work unchanged.

**"Why SVM over KNN on the ransomware data specifically?"**
With ~31,000 features and only ~1,500 samples, most points are roughly equidistant from each other in raw Euclidean space, which weakens KNN's nearest-neighbor voting. SVM's margin-maximizing decision boundary is more robust to this. See the "Binary: KNN vs. SVM" section of the notebook for the actual validation curves.

---

## 🙌 Contributing

Found a bug, want to add another model (e.g. Random Forest, Logistic Regression), or extend this to a new dataset? PRs and issues are welcome.

## 📄 License

The **code** in this repository is provided as-is for educational and research purposes. The **bundled datasets** remain the property of their original authors, under the terms specified in their respective source repositories (linked above).