# generate_data.py
import os
import csv
import random

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# Set random seed for reproducible synthetic data
random.seed(42)

# ============================================================
# 1. GENERATE symptoms.csv (100 Entries)
# ============================================================
symptom_categories = {
    'systemic': ['fever', 'high_fever', 'chills', 'fatigue', 'night_sweats', 'weight_loss', 'sweating', 'lethargy', 'malaise', 'dehydration'],
    'respiratory': ['cough', 'shortness_of_breath', 'wheezing', 'sore_throat', 'runny_nose', 'nasal_congestion', 'chest_tightness', 'hoarseness', 'sneezing', 'rapid_breathing'],
    'neurological': ['headache', 'light_sensitivity', 'stiff_neck', 'loss_of_smell', 'loss_of_taste', 'dizziness', 'confusion', 'blurred_vision', 'seizures', 'numbness'],
    'musculoskeletal': ['body_aches', 'joint_pain', 'muscle_cramps', 'back_pain', 'joint_swelling', 'muscle_weakness', 'stiffness', 'bone_pain', 'neck_pain', 'tremors'],
    'gastrointestinal': ['nausea', 'vomiting', 'diarrhea', 'abdominal_pain', 'loss_of_appetite', 'heartburn', 'bloating', 'cramping', 'indigestion', 'stomach_fullness'],
    'dermatological': ['rash', 'itching', 'skin_redness', 'hives', 'dry_skin', 'blisters', 'skin_lesions', 'flushing', 'jaundice', 'skin_peeling'],
    'cardiovascular': ['chest_pain', 'palpitations', 'irregular_heartbeat', 'high_blood_pressure', 'low_blood_pressure', 'swollen_legs', 'fainting', 'cold_extremities', 'weak_pulse', 'cyanosis'],
    'urinary_renal': ['frequent_urination', 'excessive_thirst', 'painful_urination', 'dark_urine', 'blood_in_urine', 'cloudy_urine', 'flank_pain', 'urinary_urgency', 'reduced_urination', 'incontinence'],
    'ophthalmic_ent': ['eye_redness', 'earache', 'tinnitus', 'eye_discharge', 'dry_eyes', 'swollen_glands', 'difficulty_swallowing', 'nasal_discharge', 'eye_pain', 'hearing_loss'],
    'psychiatric_cognitive': ['anxiety', 'restlessness', 'insomnia', 'irritability', 'brain_fog', 'mood_swings', 'memory_loss', 'hallucinations', 'panic', 'agitation']
}

symptoms_rows = []
symptom_id = 1
for category, sym_list in symptom_categories.items():
    for sym in sym_list:
        severity_weight = round(random.uniform(0.1, 0.95), 2)
        symptoms_rows.append({
            'symptom_id': f"SYM-{symptom_id:03d}",
            'symptom': sym,
            'category': category,
            'severity_weight': severity_weight
        })
        symptom_id += 1

with open('data/symptoms.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['symptom_id', 'symptom', 'category', 'severity_weight'])
    writer.writeheader()
    writer.writerows(symptoms_rows)

print("✅ Successfully generated data/symptoms.csv (100 rows)")


# ============================================================
# 2. GENERATE diseases.csv (100 Entries)
# ============================================================
disease_templates = [
    ('flu', 'viral', 'MEDIUM'),
    ('covid19', 'viral', 'HIGH'),
    ('dengue', 'viral', 'HIGH'),
    ('cardiac_event', 'cardiovascular', 'CRITICAL'),
    ('diabetes', 'metabolic', 'MEDIUM'),
    ('common_cold', 'viral', 'LOW'),
    ('tuberculosis', 'bacterial', 'HIGH'),
    ('meningitis', 'bacterial', 'CRITICAL'),
    ('pneumonia', 'bacterial', 'HIGH'),
    ('malaria', 'parasitic', 'HIGH'),
    ('hypertension', 'cardiovascular', 'MEDIUM'),
    ('asthma_attack', 'respiratory', 'CRITICAL'),
    ('stroke', 'neurological', 'CRITICAL'),
    ('gastroenteritis', 'viral', 'MEDIUM'),
    ('appendicitis', 'gastrointestinal', 'CRITICAL'),
    ('migraine', 'neurological', 'LOW'),
    ('sepsis', 'systemic', 'CRITICAL'),
    ('kidney_failure', 'renal', 'CRITICAL'),
    ('bronchitis', 'respiratory', 'MEDIUM'),
    ('anemia', 'hematological', 'LOW')
]

disease_rows = []
for i in range(100):
    base_name, d_type, urgency = disease_templates[i % len(disease_templates)]
    variant_suffix = f"_variant_{(i // len(disease_templates)) + 1}" if i >= len(disease_templates) else ""
    disease_name = f"{base_name}{variant_suffix}"
    disease_rows.append({
        'disease_id': f"DIS-{i+1:03d}",
        'disease': disease_name,
        'type': d_type,
        'urgency': urgency
    })

with open('data/diseases.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['disease_id', 'disease', 'type', 'urgency'])
    writer.writeheader()
    writer.writerows(disease_rows)

print("✅ Successfully generated data/diseases.csv (100 rows)")


# ============================================================
# 3. GENERATE patients.csv (100 Entries)
# ============================================================
all_symptom_names = [s['symptom'] for s in symptoms_rows]

disease_symptom_pools = {
    'flu': ['fever', 'cough', 'fatigue', 'headache', 'body_aches'],
    'covid19': ['fever', 'cough', 'fatigue', 'loss_of_smell', 'shortness_of_breath'],
    'dengue': ['fever', 'rash', 'joint_pain', 'body_aches', 'headache'],
    'cardiac_event': ['chest_pain', 'shortness_of_breath', 'sweating', 'palpitations'],
    'diabetes': ['fatigue', 'frequent_urination', 'excessive_thirst', 'blurred_vision'],
    'common_cold': ['cough', 'fever', 'headache', 'runny_nose', 'sore_throat'],
    'tuberculosis': ['cough', 'weight_loss', 'night_sweats', 'fatigue', 'fever'],
    'meningitis': ['headache', 'stiff_neck', 'high_fever', 'light_sensitivity', 'confusion']
}

patient_rows = []
for i in range(1, 101):
    pid = f"PT-{i:03d}"
    
    # Pick a target profile to ground symptom clusters
    profile_key = list(disease_symptom_pools.keys())[i % len(disease_symptom_pools)]
    base_pool = disease_symptom_pools[profile_key]
    
    # Sample 3-5 core symptoms + 1-2 random noise symptoms
    num_core = random.randint(3, len(base_pool))
    sampled_symptoms = random.sample(base_pool, num_core)
    
    noise_pool = [s for s in all_symptom_names if s not in sampled_symptoms]
    sampled_symptoms += random.sample(noise_pool, random.randint(0, 2))
    
    # Generate realistic vitals tailored to symptoms
    if 'high_fever' in sampled_symptoms or 'fever' in sampled_symptoms:
        temp = round(random.uniform(38.5, 40.5), 1)
    else:
        temp = round(random.uniform(36.5, 37.8), 1)
        
    if 'chest_pain' in sampled_symptoms or 'shortness_of_breath' in sampled_symptoms:
        hr = random.randint(100, 135)
    else:
        hr = random.randint(68, 105)
        
    symptoms_str = ",".join(sampled_symptoms)
    
    patient_rows.append({
        'patient_id': pid,
        'symptoms': symptoms_str,
        'temperature': temp,
        'heart_rate': hr
    })

with open('data/patients.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['patient_id', 'symptoms', 'temperature', 'heart_rate'])
    writer.writeheader()
    writer.writerows(patient_rows)

print("✅ Successfully generated data/patients.csv (100 rows)")
print("\n🎉 Data pipeline setup complete! All 3 CSV files are ready in the data/ folder.")