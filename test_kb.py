# test_kb.py
from modules.knowledge_base import MedicalKnowledgeBase

def run_kb_verification():
    print("🚀 Initializing CCS 3101 Medical Knowledge Base Unit Test...")
    
    # 1. Instantiate the rule engine
    kb = MedicalKnowledgeBase()
    
    # 2. Seed baseline patient symptom observations as clinical facts
    print("\n📥 Seeding patient observation data into the KB...")
    kb.add_fact("fever", 1.0)
    kb.add_fact("cough", 1.0)
    kb.add_fact("loss_of_smell", 1.0)
    kb.add_fact("fatigue", 1.0)
    
    # 3. Test the forward chaining inference engine (Data-Driven Mode)
    print("\n⚙️  Executing Data-Driven Forward Chaining Loop:")
    print("─" * 60)
    inferred_results = kb.forward_chain(verbose=True)
    print("─" * 60)
    print(f"📊 Full Inferred Output Set:\n {inferred_results}")
    
    # 4. Test the backward chaining engine (Goal-Driven Hypothesis Mode)
    print("\n🔍 Executing Goal-Driven Backward Chaining Trace:")
    target_goal = "covid19_suspected"
    proved, confidence = kb.backward_chain(target_goal)
    
    print("─" * 60)
    print(f"🎯 Hypothesis Target : {target_goal}")
    print(f"✅ Proved Status      : {proved}")
    print(f"📈 Result Confidence  : {confidence:.2%}")
    print("─" * 60)
    
    # 5. Test clean structural string normalization
    print("\n🧪 Testing Symptom Input String Normalization:")
    raw_symptoms = ["Loss of Smell", "Fatigue"]
    kb.facts.clear()  # wipe baseline facts
    kb.load_patient_symptoms(raw_symptoms)
    print(f" 🔀 Raw input strings {raw_symptoms} normalized to KB facts: {list(kb.facts)}")

if __name__ == "__main__":
    run_kb_verification()