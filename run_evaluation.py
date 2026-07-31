# run_evaluation.py
import os
from modules.ml_classifier import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from evaluation.metrics import ModelAuditor

def run_system_audit():
    print("🚀 Running System-Wide Performance Evaluation Engine...")
    os.makedirs("evaluation", exist_ok=True)
    
    # 1. Instantiate and train underlying algorithmic engines
    ml_engine = MLDiagnosticClassifier()
    dnn_engine = NeuralDiagnosticModel()
    
    print("\n🏋️  Pre-training classifier engines on validation sets...")
    ml_engine.train(verbose=False)
    dnn_engine.train(epochs=15, verbose=0)
    
    # 2. Define 5 test patients with complex variations to audit classification properties
    test_cases = [
        {"symptoms": ["Fever", "Cough", "Loss of Smell", "Fatigue"], "truth": "covid19"},
        {"symptoms": ["Headache", "Stiff Neck", "Fever", "Light Sensitivity"], "truth": "meningitis"},
        {"symptoms": ["Chest Pain", "Shortness of Breath", "Sweating"], "truth": "cardiac_event"},
        {"symptoms": ["Fever", "Rash", "Joint Pain", "Body Aches"], "truth": "dengue"},
        {"symptoms": ["Cough", "Weight Loss", "Night Sweats", "Fatigue"], "truth": "tuberculosis"}
    ]
    
    y_true = [case["truth"] for case in test_cases]
    ml_predictions = []
    dnn_predictions = []
    
    print(f"\n📥 Passing {len(test_cases)} test profile cases through architectures...")
    for idx, case in enumerate(test_cases):
        # Predict with machine learning ensemble
        ml_res = ml_engine.predict(case["symptoms"])
        ml_predictions.append(ml_res["diagnosis"])
        
        # Predict with deep learning neural net
        dnn_res = dnn_engine.predict(case["symptoms"])
        dnn_predictions.append(dnn_res["diagnosis"])
        
        print(f"  [Patient {idx+1}] True: {case['truth']:<15} | ML: {ml_res['diagnosis']:<15} | DNN: {dnn_res['diagnosis']:<15}")
        
    # 3. Compute metric reports
    disease_classes = ml_engine.DISEASE_LABELS
    auditor = ModelAuditor(labels=disease_classes)
    
    ml_metrics = auditor.compute_all_metrics(y_true, ml_predictions, "Ensemble ML Classifier")
    dnn_metrics = auditor.compute_all_metrics(y_true, dnn_predictions, "Deep Learning Neural Network")
    
    # 4. Generate comparison bar chart visual asset
    comparison_payload = {
        "Ensemble ML": ml_metrics,
        "Deep Learning DNN": dnn_metrics
    }
    auditor.generate_comparison_chart(comparison_payload)
    print("\n🏁 Evaluation workflow completed successfully!")

if __name__ == "__main__":
    run_system_audit()