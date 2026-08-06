# cli_app.py
# ============================================================
# INTERACTIVE TERMINAL DIAGNOSTIC INTERFACE
# Allows real-time user input & CSV lookup through AI pipeline
# ============================================================

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

from modules.agent          import HealthcareDiagnosticAgent, PatientPercept
from modules.knowledge_base import MedicalKnowledgeBase
from modules.bayesian_net   import SimpleBayesianDiagnostics
from modules.ml_classifier  import MLDiagnosticClassifier
from modules.neural_network import NeuralDiagnosticModel
from modules.fuzzy_controller import FuzzySeverityAssessor
from modules.planner        import TreatmentPlanner

# ── ANSI Terminal Styling ────────────────────────────────────
class C:
    HEADER = '\033[95m'; BLUE   = '\033[94m'; CYAN   = '\033[96m'
    GREEN  = '\033[92m'; YELLOW = '\033[93m'; RED    = '\033[91m'
    BOLD   = '\033[1m' ; END    = '\033[0m'

def print_banner():
    print(f"""{C.BOLD}{C.CYAN}
╔══════════════════════════════════════════════════════════════════════╗
║     🏥  INTELLIGENT HEALTHCARE AI — INTERACTIVE CLI SYSTEM          ║
║      Real-Time Multi-Paradigm Triage & Patient Analysis              ║
╚══════════════════════════════════════════════════════════════════════╝
{C.END}""")

def initialize_agent() -> HealthcareDiagnosticAgent:
    """Pre-trains models and wires up all 7 core modules"""
    print(f"⚙️  {C.YELLOW}Initializing & Pre-Training AI Subsystems...{C.END}")
    agent = HealthcareDiagnosticAgent()

    kb = MedicalKnowledgeBase()
    bn = SimpleBayesianDiagnostics()
    ml = MLDiagnosticClassifier()
    dnn = NeuralDiagnosticModel()
    fuzzy = FuzzySeverityAssessor()
    planner = TreatmentPlanner()

    ml.train(verbose=False)
    dnn.train(epochs=10, verbose=0)

    agent.register_module('KnowledgeBase', kb)
    agent.register_module('BayesianNet',   bn)
    agent.register_module('MLClassifier',  ml)
    agent.register_module('NeuralNetwork', dnn)
    agent.register_module('FuzzyLogic',    fuzzy)
    agent.register_module('TreatmentPlan', planner)

    print(f"✨ {C.GREEN}System online and ready for intake!\n{C.END}")
    return agent

# ── Natural Symptom Matching & Normalization Engine ──────────
def parse_and_normalize_symptoms(symptoms_raw: str) -> list:
    """
    Normalizes natural doctor/user inputs (e.g., 'high fever', 'loss of smell',
    'stiff neck', 'highfever') into standard system keys without requiring underscores.
    """
    if not symptoms_raw.strip():
        symptoms_raw = "fever, cough, fatigue"

    # Direct mapping for non-standard variations to canonical keys
    canonical_map = {
        'fever': 'fever',
        'highfever': 'high_fever',
        'highfever': 'high_fever',
        'cough': 'cough',
        'fatigue': 'fatigue',
        'headache': 'headache',
        'lossofsmell': 'loss_of_smell',
        'lossoftaste': 'loss_of_taste',
        'chestpain': 'chest_pain',
        'shortnessofbreath': 'shortness_of_breath',
        'shortbreath': 'shortness_of_breath',
        'rash': 'rash',
        'jointpain': 'joint_pain',
        'stiffneck': 'stiff_neck',
        'lightsensitivity': 'light_sensitivity',
        'nightsweats': 'night_sweats',
        'weightloss': 'weight_loss',
        'bodyaches': 'body_aches',
        'sorethroat': 'sore_throat',
        'runnynose': 'runny_nose',
        'sweating': 'sweating',
        'dehydration': 'dehydration',
        'chills': 'chills',
        'dizziness': 'dizziness',
        'confusion': 'confusion',
        'nausea': 'nausea',
        'vomiting': 'vomiting',
        'diarrhea': 'diarrhea',
        'abdominalpain': 'abdominal_pain'
    }

    matched_symptoms = []
    for raw_item in symptoms_raw.split(","):
        clean_item = raw_item.strip().lower()
        if not clean_item:
            continue

        # Strip spaces, hyphens, and underscores for matching key
        alpha_key = clean_item.replace(" ", "").replace("-", "").replace("_", "")

        if alpha_key in canonical_map:
            canonical_key = canonical_map[alpha_key]
        else:
            # Fallback for unlisted terms: automatically bridge spaces to underscores
            canonical_key = clean_item.replace(" ", "_").replace("-", "_")

        if canonical_key not in matched_symptoms:
            matched_symptoms.append(canonical_key)

        # Heuristic: high_fever implies fever presence
        if canonical_key == 'high_fever' and 'fever' not in matched_symptoms:
            matched_symptoms.append('fever')

    return matched_symptoms

def run_diagnosis(agent: HealthcareDiagnosticAgent, patient: PatientPercept):
    """Executes full diagnostic pipeline and formats clinical dashboard output"""
    print("\n" + "═"*70)
    print(f"📥 {C.BOLD}{C.HEADER}PATIENT INTAKE SUMMARY [{patient.patient_id}]{C.END}")
    print("═"*70)
    readable_symptoms = ", ".join([s.replace("_", " ") for s in patient.symptoms])
    print(f"  • Symptoms Entered : {C.BOLD}{readable_symptoms}{C.END}")
    print(f"  • Vital Indicators : {patient.temperature}°C | {patient.heart_rate} BPM | BP: {patient.blood_pressure} | Age: {patient.age}")
    print("─"*70)

    # 1. Individual Module Output Breakdown
    print(f"\n🧠 {C.BOLD}{C.BLUE}INDIVIDUAL SPECIALIST ANALYSES:{C.END}")
    
    ml_mod = agent._modules['MLClassifier']
    dnn_mod = agent._modules['NeuralNetwork']
    fuzzy_mod = agent._modules['FuzzyLogic']
    
    ml_res = ml_mod.predict(patient.symptoms)
    dnn_res = dnn_mod.predict(patient.symptoms)
    fuzzy_res = fuzzy_mod.assess(patient.temperature, patient.heart_rate, len(patient.symptoms))

    print(f"  • 🌳 Machine Learning (Random Forest) : {C.BOLD}{ml_res['diagnosis'].upper()}{C.END} ({ml_res['confidence']:.2%} confidence)")
    print(f"  • 🧠 Deep Neural Network (DNN)        : {C.BOLD}{dnn_res['diagnosis'].upper()}{C.END} ({dnn_res['confidence']:.2%} confidence)")
    print(f"  • 🎛️  Fuzzy Logic Severity Assessor     : {C.BOLD}{fuzzy_res['severity_label']}{C.END} (Score: {fuzzy_res['severity_score']:.1f}/100)")

    # Seed consensus to planner module
    patient.diagnosis_guess = dnn_res['diagnosis']
    patient.urgency_guess = fuzzy_res['severity_label']

    # 2. Run Main Agent Reasoning Cycle
    agent_report = agent.run(patient)

    print("\n" + "═"*70)
    print(f"📊 {C.BOLD}{C.GREEN}FINAL CONSOLIDATED CLINICAL DIAGNOSIS{C.END}")
    print("═"*70)
    print(f"  🎯 Diagnosis Consensus : {C.BOLD}{C.CYAN}{agent_report['diagnosis'].upper()}{C.END}")
    print(f"  📈 Aggregated Certainty: {agent_report['confidence']:.2%}")
    
    urgency_color = C.RED if agent_report['urgency'] in ['CRITICAL', 'HIGH'] else C.YELLOW
    print(f"  🚨 Triage Urgency Level: {C.BOLD}{urgency_color}{agent_report['urgency']}{C.END}")
    print(f"  ⚡ Action Protocol     : {agent_report['next_action']}")

    # 3. Recommendations
    print(f"\n📋 {C.BOLD}Clinical Guidelines & Recommendations:{C.END}")
    for rec in agent_report['recommendations']:
        print(f"   {rec}")

    # 4. STRIPS Treatment Plan Sequence
    planner_mod = agent._modules['TreatmentPlan']
    plan_data = planner_mod.create_treatment_plan(agent_report['diagnosis'], agent_report['urgency'])
    
    print(f"\n🗺️  {C.BOLD}Generated STRIPS Step-by-Step Treatment Plan:{C.END}")
    print("─"*70)
    if 'error' in plan_data:
        print(f"   ⚠️ Plan Stalled: {plan_data['error']}")
    else:
        for step in plan_data['plan']:
            print(f"   Step {step['step']}: {step['action']:<28} ⏱️ [{step['duration']}]")
    print("═"*70 + "\n")

# ── Input Modes ──────────────────────────────────────────────
def manual_patient_input() -> PatientPercept:
    """Gets symptoms and vitals directly from terminal user input with natural matching"""
    print(f"\n📝 {C.BOLD}{C.YELLOW}MANUAL PATIENT INTAKE FORM{C.END}")
    print("─"*50)
    
    pid = input("Enter Patient ID (e.g. PT-999 or leave blank for Auto): ").strip()
    if not pid:
        pid = f"PT-{np.random.randint(100, 999)}"

    print("\n💡 Common Symptoms: fever, high fever, cough, fatigue, headache, loss of smell, chest pain, shortness of breath, rash, joint pain, stiff neck, night sweats, weight loss")
    symptoms_raw = input(f"{C.BOLD}Enter symptoms (comma-separated): {C.END}").strip()
    
    # Process symptoms through fuzzy matcher/normalizer
    symptoms = parse_and_normalize_symptoms(symptoms_raw)

    try:
        age = int(input("Enter Patient Age [Default 30]: ") or 30)
    except ValueError:
        age = 30

    try:
        temp = float(input("Enter Temperature in °C [Default 38.5]: ") or 38.5)
    except ValueError:
        temp = 38.5

    try:
        hr = int(input("Enter Heart Rate in BPM [Default 95]: ") or 95)
    except ValueError:
        hr = 95

    bp = input("Enter Blood Pressure [Default 120/80]: ").strip() or "120/80"

    return PatientPercept(
        patient_id=pid,
        symptoms=symptoms,
        age=age,
        temperature=temp,
        heart_rate=hr,
        blood_pressure=bp
    )

def csv_patient_lookup() -> PatientPercept:
    """Loads a patient directly from data/patients.csv"""
    csv_path = "data/patients.csv"
    if not os.path.exists(csv_path):
        print(f"❌ {C.RED}Error: data/patients.csv not found! Run generate_data.py first.{C.END}")
        return manual_patient_input()

    df = pd.read_csv(csv_path)
    print(f"\n📂 {C.BOLD}{C.YELLOW}LOADED {len(df)} PATIENTS FROM data/patients.csv{C.END}")
    
    patient_query = input("Enter Patient ID (e.g. PT-001, PT-050) or Row Index (0-99): ").strip()
    
    selected_row = None
    if patient_query.isdigit() and int(patient_query) < len(df):
        selected_row = df.iloc[int(patient_query)]
    else:
        match = df[df['patient_id'].str.upper() == patient_query.upper()]
        if not match.empty:
            selected_row = match.iloc[0]
        else:
            print(f"⚠️ {C.YELLOW}Patient '{patient_query}' not found. Selecting first row (PT-001).{C.END}")
            selected_row = df.iloc[0]

    symptoms = parse_and_normalize_symptoms(str(selected_row['symptoms']))
    
    return PatientPercept(
        patient_id=str(selected_row['patient_id']),
        symptoms=symptoms,
        age=30,
        temperature=float(selected_row['temperature']),
        heart_rate=int(selected_row['heart_rate']),
        blood_pressure="120/80"
    )

# ── Main Menu Loop ───────────────────────────────────────────
def main():
    print_banner()
    agent = initialize_agent()

    while True:
        print(f"{C.BOLD}Choose Intake Mode:{C.END}")
        print("  [1] Enter Custom Symptoms & Vitals (Manual Input)")
        print("  [2] Select Patient from data/patients.csv (100 Generated Dataset)")
        print("  [3] Exit CLI System")
        
        choice = input(f"\n{C.BOLD}{C.CYAN}Select Option (1/2/3): {C.END}").strip()

        if choice == '1':
            patient = manual_patient_input()
            run_diagnosis(agent, patient)
        elif choice == '2':
            patient = csv_patient_lookup()
            run_diagnosis(agent, patient)
        elif choice == '3':
            print(f"\n👋 {C.GREEN}Exiting Healthcare AI Assistant. Goodbye!{C.END}")
            break
        else:
            print(f"⚠️ {C.RED}Invalid selection. Please choose 1, 2, or 3.\n{C.END}")

if __name__ == "__main__":
    main()