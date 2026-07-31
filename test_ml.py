import os
from modules.ml_classifier import MLDiagnosticClassifier


def run_ml_verification():
    print("Initializing CCS 3101 Supervised Ensemble Classifier Pipeline Test...")

    # 1. Instantiate and train the model collection
    clf = MLDiagnosticClassifier()
    clf.train(verbose=True)

    # 2. Test predictions with raw text containing uppercase letters and spaces
    sample_symptoms = [
        "Fever",
        "Cough",
        "Loss of Smell",
        "Fatigue"
    ]

    print(f"\nTesting Predictor Array Mapping with: {sample_symptoms}")

    result = clf.predict(sample_symptoms)

    print("\nGenerated Model Prediction Payload:")
    print("─" * 60)
    print(f"Assigned Diagnosis   : {result['diagnosis']}")
    print(f"Engine Confidence   : {result['confidence']:.2%}")
    print(f"Selected Architecture: {result['model_used']}")
    print("─" * 60)

    print("\nTop 3 Ranked Differential Diagnostic Hypotheses:")

    for label, score in result['top5'][:3]:
        print(f"  • {label:<20}: {score:.2%}")

    # 3. Test Evaluation Plots Generation
    print("\nRendering Performance Plots (Confusion Matrix & Feature Importances)...")
    clf.plot_evaluation()


if __name__ == "__main__":
    run_ml_verification()