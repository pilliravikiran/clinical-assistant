"""
app/services/classifier_service.py
==================================

A clinical note-type classifier. Given some text, predict its TYPE:
"Physician Note", "Discharge Summary", or "Lab Report".

How it works (the bridge NLP -> classic ML):
  1. FEATURES: we turn each text into an embedding (a vector of numbers).
  2. LABELS:   we know the correct type for our training examples.
  3. MODEL:    a Logistic Regression learns to map features -> label.
  4. PREDICT:  for new text, embed it and ask the model for the type + probability.

Terminology used: features (X), labels (y), training, fit, predict,
predict_proba (class probabilities), Logistic Regression.
"""

from sklearn.linear_model import LogisticRegression

from app.services.embedding_service import embed_documents


# TRAINING DATA: (text, label) pairs. In a real project these come from many
# real (de-identified) documents; here we use a few clear examples per class.
TRAINING_EXAMPLES = [
    # ---- Physician Note ----
    ("The patient presents with hypertension. Blood pressure was 150 over 95.", "Physician Note"),
    ("Assessment: type 2 diabetes managed with metformin. Plan: continue medication.", "Physician Note"),
    ("Chief complaint: morning headaches. Start blood pressure medication and follow up.", "Physician Note"),
    ("The patient reports chest discomfort. Physical exam is unremarkable.", "Physician Note"),
    ("Plan: reduce salt intake, exercise, recheck in four weeks.", "Physician Note"),

    # ---- Discharge Summary ----
    ("Follow-up care recommended: finish the antibiotics and see your doctor in one week.", "Discharge Summary"),
    ("Hospital course: admitted with pneumonia, treated with IV antibiotics, improved.", "Discharge Summary"),
    ("Discharge instructions: return to the emergency room if fever returns.", "Discharge Summary"),
    ("The patient was discharged in stable condition after three days.", "Discharge Summary"),
    ("Reason for admission: community-acquired pneumonia. Discharged home.", "Discharge Summary"),

    # ---- Lab Report ----
    ("Fasting glucose 138 mg/dL, high. Total cholesterol 220, borderline high.", "Lab Report"),
    ("Results: LDL cholesterol 145, HDL 40. Recommend repeat testing.", "Lab Report"),
    ("Basic metabolic panel shows elevated glucose. Lipid profile abnormal.", "Lab Report"),
    ("Hemoglobin A1c is 7.8 percent, above target range.", "Lab Report"),
    ("Test panel results: sodium 140, potassium 4.2, creatinine normal.", "Lab Report"),
]


# Module-level cache for the trained model.
_classifier = None
_labels = None


def train():
    """
    Train the classifier on TRAINING_EXAMPLES.

    Steps:
      - split examples into texts and labels
      - FEATURES: embed the texts (text -> numbers)
      - MODEL: fit a Logistic Regression on (features, labels)

    Output: the trained model (also cached in _classifier).
    Called by: classify() (auto-trains on first use) and tests/demos.
    """
    global _classifier, _labels

    texts = [text for text, label in TRAINING_EXAMPLES]
    labels = [label for text, label in TRAINING_EXAMPLES]

    # FEATURES (X): each text -> its embedding vector.
    X = embed_documents(texts)
    # LABELS (y): the correct type for each text.
    y = labels

    # Create and TRAIN (fit) the model.
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    _classifier = model
    _labels = sorted(set(labels))
    return model


def classify(text):
    """
    Predict the type of a piece of clinical text.

    Input:  text -> a string
    Output: a dict {"label": <predicted type>, "confidence": <0..1>,
                    "probabilities": {type: prob, ...}}

    Called by: an /classify endpoint or the ingestion step (auto-tagging).
    """
    # Train on first use if not trained yet.
    if _classifier is None:
        train()

    # FEATURES for the new text (must use the SAME embedding as training).
    X = embed_documents([text])

    # PREDICT the label, and the probability for every class.
    predicted = _classifier.predict(X)[0]
    proba = _classifier.predict_proba(X)[0]           # one probability per class
    classes = _classifier.classes_                     # the class order for proba

    probabilities = {cls: float(p) for cls, p in zip(classes, proba)}
    confidence = float(max(proba))

    return {
        "label": predicted,
        "confidence": confidence,
        "probabilities": probabilities,
    }
