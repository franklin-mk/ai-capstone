# 🏥 Intelligent Healthcare Diagnostic Assistant AI

An integrated, multi-paradigm clinical diagnosis system developed for the **CCS 3101 – Introduction to Artificial Intelligence Capstone Project**.

The application combines **symbolic AI**, **probabilistic reasoning**, **machine learning**, **deep learning**, **fuzzy logic**, and **AI planning** into a single clinical decision-support pipeline capable of analyzing patient symptoms, estimating diagnostic probabilities, assessing disease severity, and generating treatment plans.

---

# 📌 Project Objectives

This project demonstrates the practical application of multiple Artificial Intelligence paradigms within a healthcare setting by integrating:

- Rule-Based Expert Systems
- First Order Logic (FOL)
- Forward Chaining
- Backward Chaining
- Bayesian Networks
- Decision Trees & Random Forests
- Artificial Neural Networks
- Fuzzy Logic Inference
- STRIPS Planning

---

# 🏗️ System Architecture

The patient information flows sequentially through each intelligent module.

```text
                    Patient Symptoms & Vitals
                              │
                              ▼
                 Patient Percept Object Intake
                              │
                              ▼
        Module 2 — Knowledge Base (FOL Rules)
         • Forward Chaining
         • Backward Chaining
                              │
                              ▼
          Module 3 — Bayesian Network
         • Probabilistic Diagnosis
         • Joint Log Probability
                              │
                              ▼
        Module 4 — Machine Learning
         • Decision Trees
         • Random Forest Classifier
                              │
                              ▼
         Module 5 — Deep Learning
         • Artificial Neural Network
         • Softmax Disease Prediction
                              │
                              ▼
       Module 6 — Fuzzy Logic System
         • Disease Severity (0–100)
         • Centroid Defuzzification
                              │
                              ▼
       Module 7 — STRIPS Planner
         • Linear Treatment Plan
         • Recommended Clinical Actions
```

---

# 🧠 AI Modules

| Module | Technique | Purpose |
|---------|-----------|---------|
| Module 1 | Intelligent Agent | Patient Perception |
| Module 2 | First Order Logic | Rule-Based Diagnosis |
| Module 3 | Bayesian Network | Probabilistic Reasoning |
| Module 4 | Random Forest | Machine Learning Classification |
| Module 5 | Neural Network | Deep Learning Prediction |
| Module 6 | Fuzzy Logic | Severity Assessment |
| Module 7 | STRIPS Planning | Treatment Planning |

---

# 📂 Project Structure

```text
ai-capstone/
│
├── app.py
├── requirements.txt
├── README.md
│
├── modules/
│   ├── agent.py
│   ├── knowledge_base.py
│   ├── bayesian_network.py
│   ├── machine_learning.py
│   ├── neural_network.py
│   ├── fuzzy_logic.py
│   └── planner.py
│
├── data/
├── evaluation/
├── reports/
└── testing/
```

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/franklin-mk/ai-capstone.git

cd ai-capstone
```

---

## 2. Create a Virtual Environment

```bash
python -m venv venv
```

---

## 3. Activate the Virtual Environment

### Windows (Git Bash)

```bash
source venv/Scripts/activate
```

### Windows (Command Prompt)

```cmd
venv\Scripts\activate
```

### Windows (PowerShell)

```powershell
.\venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
source venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Run the Application

```bash
python app.py
```

---

# 🧪 Running Tests

Run individual test modules from the project root.

Example:

```bash
python -m testing.test_agent
```

or

```bash
python test_kb.py
```

---

# 📦 Major Technologies

- Python 3.x
- NumPy
- Pandas
- Scikit-Learn
- TensorFlow / Keras
- pgmpy
- SciPy
- NetworkX
- NLTK
- Gymnasium
- Matplotlib

---

# 🎯 Learning Outcomes

This capstone demonstrates the implementation and integration of:

- Intelligent Agents
- Knowledge Representation
- Rule-Based Systems
- First Order Logic
- Forward & Backward Chaining
- Bayesian Inference
- Decision Trees
- Random Forests
- Deep Neural Networks
- Fuzzy Inference Systems
- AI Planning (STRIPS)

---

# 👨‍💻 Contributors

Developed as part of the **CCS 3101 – Introduction to Artificial Intelligence** Capstone Project.

---

# 📄 License

This repository is intended for educational purposes.

---

# ⭐ Happy Coding!

If you find this project useful, feel free to fork it, improve it, and contribute.

**Happy Coding! 🚀**