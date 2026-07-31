# test_bayesian.py
from modules.bayesian_net import SimpleBayesianDiagnostics


def run_bayesian_verification():
    print("Initializing CCS 3101 Bayesian Network Diagnostic Unit Test...")
    # 1. Instantiate the probabilistic diagnostics engine
    bn = SimpleBayesianDiagnostics()
    # 2. Test symptoms that strongly indicate COVID-19
    test_symptoms = ["fever", "cough", "loss_of_smell", "fatigue"]
    print(
        f"\nCase 1: Testing patient with matching respiratory signs: {test_symptoms}"
    )

    posteriors = bn.compute_posterior(test_symptoms)
    ranked = sorted(posteriors.items(), key=lambda x: x[1], reverse=True)
    print("         Probabilistic Diagnostic Output Matrix:")
    print("─" * 50)
    for disease, prob in ranked[:4]:
        print(f"  {disease:<20}: {prob:.2%}")
    print("─" * 50)
    # 3. Test string normalization cleaning
    raw_mixed_symptoms = ["Loss of Smell", "Joint Pain"]
    print(
        f"\nCase 2: Testing feature token standardization for: {raw_mixed_symptoms}"
    )
    explanation = bn.explain("covid19", raw_mixed_symptoms)
    print(f"        Mathematical Breakdown:\n  {explanation}")


if __name__ == "__main__":
    run_bayesian_verification()
