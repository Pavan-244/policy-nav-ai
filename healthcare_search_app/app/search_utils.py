import joblib
from sklearn.metrics.pairwise import cosine_similarity

VECTORIZER_PATH = "app/models/healthcare_vectorizer.pkl"
MATRIX_PATH = "app/models/healthcare_tfidf_matrix.pkl"

vectorizer = joblib.load(VECTORIZER_PATH)
data = joblib.load(MATRIX_PATH)
tfidf_matrix = data["matrix"]
full_df = data["df"]

def search_records(query, top_k=3, condition=None, doctor=None, hospital=None):
    query_vec = vectorizer.transform([query.lower()])
    sims = cosine_similarity(query_vec, tfidf_matrix).flatten()
    df_copy = full_df.copy()
    df_copy["Similarity"] = sims
    if condition:
        df_copy = df_copy[df_copy["Medical Condition"].str.contains(condition, case=False, na=False)]
    if doctor:
        df_copy = df_copy[df_copy["Doctor"].str.contains(doctor, case=False, na=False)]
    if hospital:
        df_copy = df_copy[df_copy["Hospital"].str.contains(hospital, case=False, na=False)]
    top_records = df_copy.sort_values(by="Similarity", ascending=False).head(top_k)
    results = []
    for _, row in top_records.iterrows():
        results.append({
            "Name": row.get("Name", ""),
            "Age": row.get("Age", ""),
            "Gender": row.get("Gender", ""),
            "Blood Type": row.get("Blood Type", ""),
            "Medical Condition": row.get("Medical Condition", ""),
            "Date of Admission": row.get("Date of Admission", ""),
            "Doctor": row.get("Doctor", ""),
            "Hospital": row.get("Hospital", ""),
            "Insurance Provider": row.get("Insurance Provider", ""),
            "Billing Amount": row.get("Billing Amount", ""),
            "Room Number": row.get("Room Number", ""),
            "Admission Type": row.get("Admission Type", ""),
            "Discharge Date": row.get("Discharge Date", ""),
            "Medication": row.get("Medication", ""),
            "Test Results": row.get("Test Results", ""),
            "Similarity": float(row["Similarity"])
        })
    return results
