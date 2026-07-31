# test_planner.py
# ============================================================
# MODULE 7 STANDALONE TEST HARNESS
# Verifies: STRIPS Planning Engine & BFS State Space Search
# ============================================================

from modules.planner import TreatmentPlanner


def run_planner_verification():
    print("Initializing CCS 3101 STRIPS Treatment Planner Unit Test...")

    # 1. Instantiate the planning engine
    planner = TreatmentPlanner()

    # 2. Test for the COVID-19 case provided in your lab manual
    target_dx = 'covid19'
    urgency_level = 'HIGH'

    print(
        f"\nGenerating step-by-step state-space path for: "
        f"{target_dx.upper()} (Severity: {urgency_level})"
    )

    plan = planner.create_treatment_plan(target_dx, urgency_level)

    # 3. Print out structural diagnostic validation properties
    print("\nGenerated STRIPS Action Sequence Dashboard:")
    print("─" * 65)
    print(f"Target Diagnosis    : {plan.get('diagnosis')}")
    print(f"Total Plan Steps    : {plan.get('steps')}")
    print(f"Estimated Duration  : {plan.get('total_duration')}")
    print("─" * 65)

    # 4. Enumerate path transitions
    if 'error' in plan:
        print(f"Planning Engine Failure: {plan['error']}")
    else:
        for step in plan['plan']:
            print(
                f"Step {step['step']:2d}: "
                f"{step['action']:<25} [{step['duration']}]"
            )

        print("─" * 65)


if __name__ == "__main__":
    run_planner_verification()