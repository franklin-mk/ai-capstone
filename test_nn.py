# test_nn.py

from modules.neural_network import NeuralDiagnosticModel


def run_dnn_verification():
    print("Initializing CCS 3101 Deep Neural Network Specialist Test...")

    # 1. Instantiate the Deep Learning Model
    nn = NeuralDiagnosticModel()

    # 2. Train the network for 25 epochs
    nn.train(epochs=25, verbose=1)

    # 3. Pass a complex set of neurological symptoms
    test_symptoms = [
        "Headache",
        "Stiff Neck",
        "Fever",
        "Light Sensitivity"
    ]

    print(f"\nTesting Deep Classifier Predictions with: {test_symptoms}")

    result = nn.predict(test_symptoms)

    print("\nGenerated Deep Learning Prediction Output:")
    print("─" * 60)
    print(f"Assigned Diagnosis : {result['diagnosis']}")
    print(f"Network Confidence : {result['confidence']:.2%}")
    print("─" * 60)

    print("\nComplete Disease Softmax Probability Layout Matrix:")
    for disease, prob in sorted(
        result["all_probs"].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f" • {disease:<20}: {prob:.2%}")

    # 4. Generate loss/accuracy learning progression plots
    print("\nGenerating Neural Training Optimization Plot Curves...")
    nn.plot_training()


if __name__ == "__main__":
    run_dnn_verification()