"""
scripts/generate_sample_data.py
===============================

This small program creates FAKE medical documents and saves them
into the 'data' folder, so we have something to practice on.

Everything here is invented. No real patients, no real records.

To run it:  python scripts/generate_sample_data.py
"""

import os


# The folder where we will save the documents.
# We build the path so it works no matter where you run the command from.
HERE = os.path.dirname(__file__)                 # the 'scripts' folder
PROJECT = os.path.dirname(HERE)                  # the project folder (one up)
DATA_FOLDER = os.path.join(PROJECT, "data")      # the 'data' folder


# Our fake documents. Each one is a small dictionary with:
#   - "filename": what to call the saved file
#   - "text": the content of the document
# The first lines (Document Type / ID / Date) are a little "header"
# we will read later to know what kind of document this is.
DOCUMENTS = [
    {
        "filename": "physician_note.txt",
        "text": """Document Type: Physician Note
Document ID: PN-001
Date: 2025-03-14

Patient: John Sample (fictional)
Complaint: Elevated blood pressure and morning headaches.

Assessment:
The patient presents with hypertension. Blood pressure was 150/95.

Plan:
Start blood pressure medication. Reduce salt. Follow up in four weeks.
""",
    },
    {
        "filename": "discharge_summary.txt",
        "text": """Document Type: Discharge Summary
Document ID: DS-001
Date: 2025-04-02

Patient: Mary Example (fictional)
Reason for Admission: Pneumonia.

Follow-up Care Recommended:
Finish the full 7-day course of antibiotics at home.
See your primary care doctor in one week.
Return to the emergency room if fever comes back.
""",
    },
    {
        "filename": "lab_report.txt",
        "text": """Document Type: Laboratory Report
Document ID: LR-001
Date: 2025-04-10

Patient: Robert Test (fictional)

Results:
Fasting Glucose: 138 (High).
Total Cholesterol: 220 (Borderline high).

Interpretation:
High glucose and cholesterol. Recommend diet changes and re-testing.
""",
    },
]


# Make sure the 'data' folder exists (create it if needed).
os.makedirs(DATA_FOLDER, exist_ok=True)

# Go through each fake document and save it as a .txt file.
for doc in DOCUMENTS:
    file_path = os.path.join(DATA_FOLDER, doc["filename"])
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(doc["text"])
    print("Saved:", doc["filename"])

print("Done! Created", len(DOCUMENTS), "fake documents in the 'data' folder.")
