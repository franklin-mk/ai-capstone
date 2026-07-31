# evaluation/metrics.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

class ModelAuditor:
    """Computes, prints, and visualizes clinical validation metrics for the system"""
    
    def __init__(self, labels: list):
        self.labels = labels

    def compute_all_metrics(self, y_true: list, y_pred: list, model_name: str) -> dict:
        """Calculates statistical metrics for classification profiles"""
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        print(f"\n📊 {model_name} Performance Audit Report:")
        print("─" * 45)
        print(f"  • Accuracy  : {acc:.4f}")
        print(f"  • Precision : {prec:.4f}")
        print(f"  • Recall    : {rec:.4f}")
        print(f"  • F1-Score  : {f1:.4f}")
        print("─" * 45)
        
        return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1}

    def generate_comparison_chart(self, summary_data: dict, output_path: str = "evaluation/model_comparison.png"):
        """Generates a structural comparative performance bar plot"""
        df = pd.DataFrame(summary_data).T
        
        plt.figure(figsize=(10, 6))
        ax = df.plot(kind='bar', figsize=(10, 6), edgecolor='black', zorder=3)
        plt.title("Cross-Architecture Performance Comparison", fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Model Engine Variant", fontsize=11, fontweight='bold')
        plt.ylabel("Metric Performance Score", fontsize=11, fontweight='bold')
        plt.xticks(rotation=0)
        plt.ylim(0, 1.1)
        plt.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="gray")
        plt.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        print(f"✅ Comparison visual metrics plot saved to: {output_path}")