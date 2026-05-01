import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
FILE_PATH = 'coindropperformance.xlsx'
# Column 1: predicted/sensor result
# Column 2: actual/true result
# Labels: '10h10d', '10h30d', '30h10d', '30h30d'

LABELS = ['10h10d', '10h30d', '30h10d', '30h30d']

# ── Load file ────────────────────────────────────────────────────────────────
def load_data(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext in ['.xlsx', '.xls']:
        df = pd.read_excel(filepath, header=0)
    elif ext == '.csv':
        df = pd.read_csv(filepath, header=0)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return df

# ── Confusion matrix ─────────────────────────────────────────────────────────
def build_confusion_matrix(df):
    pred_cat = df.iloc[:, 0].astype(str).str.strip()
    true_cat = df.iloc[:, 1].astype(str).str.strip()

    n = len(LABELS)
    matrix = pd.DataFrame(0, index=LABELS, columns=LABELS)

    skipped = 0
    for pred, true in zip(pred_cat, true_cat):
        if true in LABELS and pred in LABELS:
            matrix.loc[true, pred] += 1
        else:
            skipped += 1

    if skipped > 0:
        print(f"  Warning: {skipped} rows had unrecognised labels and were skipped.")

    return matrix

# ── Print results ─────────────────────────────────────────────────────────────
def print_results(matrix):
    total   = matrix.values.sum()
    correct = np.trace(matrix.values)
    accuracy = correct / total * 100 if total > 0 else 0

    print("\n=== 4x4 Confusion Matrix ===")
    print(f"{'':>10}", end='')
    for col in matrix.columns:
        print(f"  {col:>8}", end='')
    print()
    for idx, row in matrix.iterrows():
        print(f"{idx:>10}", end='')
        for val in row:
            print(f"  {val:>8}", end='')
        print()

    print(f"\nTotal samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.1f}%")

    # Per-class metrics
    print(f"\n{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    for label in LABELS:
        tp = matrix.loc[label, label]
        fp = matrix[label].sum() - tp        # col sum - TP
        fn = matrix.loc[label].sum() - tp    # row sum - TP
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
        print(f"{label:<10} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_matrix(matrix, save_path='confusion_matrix_4x4.png'):
    data      = matrix.values.astype(float)
    norm_data = data / data.max() if data.max() > 0 else data

    total    = data.sum()
    correct  = np.trace(data)
    accuracy = correct / total * 100 if total > 0 else 0

    # Per-class metrics for footer
    precisions, recalls, f1s = [], [], []
    for i, label in enumerate(LABELS):
        tp = data[i, i]
        fp = data[:, i].sum() - tp
        fn = data[i, :].sum() - tp
        p  = tp / (tp + fp) if (tp + fp) > 0 else 0
        r  = tp / (tp + fn) if (tp + fn) > 0 else 0
        f  = (2 * p * r / (p + r)) if (p + r) > 0 else 0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)

    macro_p  = np.mean(precisions)
    macro_r  = np.mean(recalls)
    macro_f1 = np.mean(f1s)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm_data, cmap=plt.cm.Blues, vmin=0, vmax=1)

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            val        = int(data[i, j])
            text_color = 'white' if norm_data[i, j] > 0.55 else 'black'
            ax.text(j, i, str(val),
                    ha='center', va='center',
                    fontsize=14, fontweight='bold', color=text_color)

    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_yticklabels(LABELS, fontsize=11)
    ax.set_xlabel("Sensor Detected Result", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Result", fontsize=12, labelpad=10)
    ax.set_title("4x4 Confusion Matrix", fontsize=14, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relative count', fontsize=10)

    metrics = (f"Accuracy: {accuracy:.1f}%   |   "
               f"Macro Precision: {macro_p:.3f}   |   "
               f"Macro Recall: {macro_r:.3f}   |   "
               f"Macro F1: {macro_f1:.3f}")
    fig.text(0.5, 0.01, metrics, ha='center', fontsize=9, color='#444444')

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {save_path}")
    plt.show()

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    filepath = sys.argv[1] if len(sys.argv) > 1 else FILE_PATH

    print(f"Loading: {filepath}")
    df = load_data(filepath)
    print(f"Loaded {len(df)} rows.")
    print(f"Columns: {list(df.columns)}")

    matrix = build_confusion_matrix(df)
    print_results(matrix)
    plot_matrix(matrix)