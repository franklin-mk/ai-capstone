# test_agent.py
import sys
from modules.agent import HealthcareDiagnosticAgent, PatientPercept

# Mock a simple diagnostic module to mimic specialized sub-modules (Weeks 8-11)
class MockDiagnosticSpecialist:
    def analyze(self, patient):
        # Simply returns a dummy diagnostic opinion for testing
        return {
            'diagnosis': 'flu',
            'confidence': 0.85,
            'summary': 'Diagnosed mild flu based on common patterns'
        }

def run_agent_verification():
    print("🚀 Initializing CCS 3101 Healthcare Diagnostic Agent Test...")
    
    # 1. Instantiate the intelligent core agent
    agent = HealthcareDiagnosticAgent()
    
    # 2. Register our mock clinical specialist module
    mock_specialist = MockDiagnosticSpecialist()
    agent.register_module("Mock_ML_Classifier", mock_specialist)
    
    # 3. Create a sample patient percept object (Simulating Intake Form)
    test_patient = PatientPercept(
        patient_id="P001",
        symptoms=["fever", "cough", "fatigue"],
        age=34,
        temperature=38.9,  # In Celsius
        heart_rate=98,     # In BPM
        blood_pressure="120/80"
    )
    
    print("\n--- Starting Full Cycle: Perceive -> Think -> Act ---")
    # 4. Trigger the full integrated agent execution cycle
    final_report = agent.run(test_patient)
    
    # 5. Output and display the logs & results
    agent.print_log()
    
    print("\n📊 Final Generated Action Report:")
    print("=" * 50)
    for key, value in final_report.items():
        print(f"{key:<16}: {value}")
    print("=" * 50)
    
    # 6. Verify tracking stats (Goal-based metrics)
    print("\n📈 Agent Performance Profile:")
    print(agent.get_performance())

if __name__ == "__main__":
    run_agent_verification()