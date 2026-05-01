import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# --- Configuration ---
FILE_PATH = 'strongperformance.xlsx'   
# Column 1: predicted/sensor result (0 = not detected, 1 = detected)
# Column 2: actual/true result      (0 = not detected, 1 = detected)

LABELS = ['Detected', 'Not Detected']

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
def compute_confusion(df):
    pred_cat = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().astype(int)
    true_cat = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().astype(int)

    n = min(len(pred_cat), len(true_cat))
    pred_cat = pred_cat.iloc[:n].values
    true_cat = true_cat.iloc[:n].values

    TP = int(np.sum((true_cat == 1) & (pred_cat == 1)))
    FP = int(np.sum((true_cat == 0) & (pred_cat == 1)))
    FN = int(np.sum((true_cat == 1) & (pred_cat == 0)))
    TN = int(np.sum((true_cat == 0) & (pred_cat == 0)))

    return TP, FP, FN, TN

# ── Print results ─────────────────────────────────────────────────────────────
def print_results(TP, FP, FN, TN):
    total     = TP + FP + FN + TN
    total_pos = TP + FN
    total_neg = FP + TN
    accuracy  = (TP + TN) / total * 100 if total > 0 else 0
    precision = TP / (TP + FP)          if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN)          if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    print("\n=== 2x2 Confusion Matrix ===")
    print(f"{'':>22} {'Pred Detected':>15} {'Pred Not Detected':>18}")
    print(f"{'Actual Detected':>22} {'TP = ' + str(TP):>15} {'FN = ' + str(FN):>18}")
    print(f"{'Actual Not Detected':>22} {'FP = ' + str(FP):>15} {'TN = ' + str(TN):>18}")
    print(f"\nTotal samples       : {total}")
    print(f"Total Positives     : {total_pos}")
    print(f"Total Negatives     : {total_neg}")
    print(f"Accuracy            : {accuracy:.1f}%")
    print(f"Precision           : {precision:.3f}")
    print(f"Recall (Sensitivity): {recall:.3f}")
    print(f"F1 Score            : {f1:.3f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_matrix(TP, FP, FN, TN, save_path='confusion_matrix_2x2.png'):
    matrix = np.array([[TP, FN],
                        [FP, TN]], dtype=float)

    cell_labels = [['TP', 'FN'],
                   ['FP', 'TN']]

    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total * 100 if total > 0 else 0
    precision = TP / (TP + FP)          if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN)          if (TP + FN) > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    norm_data = matrix / matrix.max() if matrix.max() > 0 else matrix

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(norm_data, cmap=plt.cm.Blues, vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            val = int(matrix[i, j])
            text_color = 'white' if norm_data[i, j] > 0.55 else 'black'
            ax.text(j, i - 0.1, str(val),
                    ha='center', va='center',
                    fontsize=20, fontweight='bold', color=text_color)
            ax.text(j, i + 0.22, cell_labels[i][j],
                    ha='center', va='center',
                    fontsize=11, color=text_color, style='italic')

    ax.set_xticks(range(2))
    ax.set_yticks(range(2))
    ax.set_xticklabels(LABELS, fontsize=11)
    ax.set_yticklabels(LABELS, fontsize=11)
    ax.set_xlabel("Sensor Detected Result", fontsize=12, labelpad=10)
    ax.set_ylabel("Actual Result", fontsize=12, labelpad=10)
    ax.set_title("2x2 Confusion Matrix", fontsize=14, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Relative count', fontsize=10)

    metrics = (f"Accuracy: {accuracy:.1f}%   |   Precision: {precision:.3f}"
               f"   |   Recall: {recall:.3f}   |   F1: {f1:.3f}")
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

    TP, FP, FN, TN = compute_confusion(df)
    print_results(TP, FP, FN, TN)
    plot_matrix(TP, FP, FN, TN)