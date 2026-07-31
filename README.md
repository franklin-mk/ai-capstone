# 🏥 Intelligent Healthcare Diagnostic Assistant AI

An integrated, multi-paradigm clinical diagnosis decision engine built for the **CCS 3101 Introduction to AI Capstone Project**. The system pairs classical expert rules with machine learning classification architectures to handle clinical reasoning, statistical patterns, uncertainty, and linear strategic treatment routes.

---

## 🛠️ System Architecture Diagram Overview

The core application pipes patient symptoms and vitals through a linear multi-module evaluation workflow:

```text
[Patient Percept Object Intake]
            │
            ├──► [Module 2] FOL Knowledge Base ──► Chaining Explanations
            ├──► [Module 3] Bayesian Network   ──► Joint Log Probabilities
            ├──► [Module 4] ML Tree Ensembles  ──► Feature Splits (Random Forest)
            ├──► [Module 5] Deep Neural Net    ──► Softmax Categorization
            │
            ▼
    [Module 6] Fuzzy Severity Assessor ──► Fuzzified Centroid Severity Score (0-100)
            │
            ▼
    [Module 7] STRIPS Search Planner   ──► turn-by-turn Linear Treatment Blueprint



# 1. Clone the repository codebase
git clone [https://github.com/franklin-mk/ai-capstone.git](https://github.com/franklin-mk/ai-capstone.git)
cd ai-capstone

# 2. Configure a local virtual environment sandbox
python -m venv venv

# 3. Activate the virtual environment shell isolation layer
# For Windows Git Bash / Bash:
source venv/Scripts/activate

# 4. Install all macro system-level mathematical dependencies
pip install -r requirements.txt

# 5. Execute the core master pipeline dashboard loop
python app.py