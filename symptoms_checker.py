import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#severity mapping
symptom_severity = {
    'fever': 2,
    'cough': 1,
    'shortness_of_breath': 3,
    'chest_pain': 4,
    'headache': 1,
    'vomiting': 3,
    'fatigue': 2,
    'abdominal_pain': 3,
    'dizziness': 2,
    'sore_throat': 1
}

feature_names = list(symptom_severity.keys())


def generate_data(samples_per_class=1000):
    data = []
    labels = []
    np.random.seed(42)

    for _ in range(samples_per_class):
        symptoms = np.random.binomial(1, 0.2, len(feature_names))  # Low
        data.append(symptoms)
        labels.append("Low")

    for _ in range(samples_per_class):
        symptoms = np.random.binomial(1, 0.5, len(feature_names))  # Medium
        data.append(symptoms)
        labels.append("Medium")

    for _ in range(samples_per_class):
        symptoms = np.random.binomial(1, 0.8, len(feature_names))  # High
        data.append(symptoms)
        labels.append("High")

    df = pd.DataFrame(data, columns=feature_names)
    return df, pd.Series(labels)


X, y = generate_data(samples_per_class=1000)

#training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.25, random_state=42
)

model = DecisionTreeClassifier(
    criterion='entropy',
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=42
)
model.fit(X_train, y_train)

#Algo1:
def rule_based_classifier(symptoms_selected):
    score = 0
    emergency_flags = ['chest_pain', 'shortness_of_breath', 'vomiting']
    for symptom in symptoms_selected:
        score += symptom_severity.get(symptom.lower(), 0)

    if any(symptom in emergency_flags for symptom in symptoms_selected):
        return "High"

    if score >= 8:
        return "High"
    elif score >= 4:
        return "Medium"
    else:
        return "Low"

#Algo2:
def ensemble_prediction(rule_pred, tree_pred):
    if rule_pred == tree_pred:
        return rule_pred
    elif rule_pred == "High":
        return "High"
    elif rule_pred == "Medium" and tree_pred == "Low":
        return "Medium"
    elif rule_pred == "Low" and tree_pred == "Medium":
        return "Medium"
    else:
        return tree_pred


def check_symptoms():
    print("\n SYMPTOM CHECKER — Answer with yes/no\n")
    user_input_vector = []
    user_symptom_list = []

    for symptom in feature_names:
        val = input(f"Do you have {symptom.replace('_', ' ')}? ").strip().lower()
        presence = 1 if val in ["yes", "y"] else 0
        user_input_vector.append(presence)
        if presence:
            user_symptom_list.append(symptom)

    #predictions
    rule_pred = rule_based_classifier(user_symptom_list)
    tree_pred = model.predict([user_input_vector])[0]
    final_pred = ensemble_prediction(rule_pred, tree_pred)

    #Results
    print("\n Rule-Based Prediction:     ", rule_pred)
    print(" Decision Tree Prediction:  ", tree_pred)
    print(" Final Ensemble Prediction: ", final_pred)

    # Recommendation
    if final_pred == "High":
        print("Recommendation: Seek EMERGENCY medical care immediately.")
    elif final_pred == "Medium":
        print("Recommendation: Visit a doctor within 24 hours.")
    else:
        print("Recommendation: Self-care is sufficient. Monitor symptoms.")

check_symptoms()
