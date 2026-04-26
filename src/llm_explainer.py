"""
llm_explainer.py
----------------
Calls a local Ollama LLM (mistral or ) to generate a plain-English
clinical explanation for a patient's predicted risk score,
using retrieved similar patients as supporting evidence.
"""

import ollama
from config import LLM_MODEL, LLM_TEMPERATURE


SYSTEM_PROMPT = """You are a helpful clinical decision support assistant.
Your job is to explain a patient's predicted risk score in plain English.
Use the similar patient cases provided as evidence to support your explanation.
Keep your explanation clear, concise, and non-technical.
Do not make definitive diagnoses - only provide supportive insights.
Always recommend consulting a qualified healthcare provider."""


def build_prompt(
    patient_summary: str,
    risk_prob: float,
    risk_category: str,
    similar_patients: list[dict],
) -> str:
    """
    Build the full prompt string to send to the LLM.

    Parameters
    ----------
    patient_summary  : natural-language description of the query patient
    risk_prob        : predicted probability (0.0 – 1.0)
    risk_category    : "Low" | "Medium" | "High"
    similar_patients : list of dicts from retriever.retrieve_similar_patients()
    """
    evidence_block = ""
    for i, sp in enumerate(similar_patients, 1):
        evidence_block += (
            f"\nSimilar Patient {i} (similarity distance: {sp['distance']}):\n"
            f"{sp['summary']}\n"
        )

    prompt = f"""Patient Summary:
{patient_summary}

Predicted Risk Score: {risk_prob:.1%} ({risk_category} Risk)

Similar Patient Evidence (from historical records):
{evidence_block}

Based on the above information, please provide:
1. A plain-English explanation of why this patient may be at {risk_category} risk.
2. Key clinical factors driving the prediction.
3. What the similar patient cases suggest about this patient's risk profile.
4. Any lifestyle or monitoring recommendations worth noting.

Keep your answer to 3–4 short paragraphs."""
    return prompt


def generate_explanation(
    patient_summary: str,
    risk_prob: float,
    risk_category: str,
    similar_patients: list[dict],
) -> str:
    """
    Call the local Ollama LLM and return the generated explanation.

    Requires Ollama to be running:
        ollama serve
    And the model to be pulled:
        ollama pull mistral
    """
    prompt = build_prompt(patient_summary, risk_prob, risk_category, similar_patients)

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            options={"temperature": LLM_TEMPERATURE},
        )
        return response["message"]["content"].strip()

    except Exception as e:
        return (
            f"⚠️  Could not generate explanation. Make sure Ollama is running "
            f"and '{LLM_MODEL}' is pulled.\n\n"
            f"  ollama serve\n"
            f"  ollama pull {LLM_MODEL}\n\n"
            f"Error: {e}"
        )


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_summary = (
        "55-year-old female (White). Clinical history: 12 recorded conditions, "
        "8 distinct medications, 15 procedures. Healthcare utilisation: 40 total "
        "encounters. Lab values: HbA1c 9.2%, systolic BP 145 mmHg, BMI 32.1."
    )
    test_similar = [
        {
            "patient_id": "abc123",
            "summary": "58-year-old female with 10 conditions, HbA1c 9.0%, "
                       "BMI 31, 35 encounters. Has a recorded diabetes complication.",
            "distance": 0.12,
        }
    ]
    explanation = generate_explanation(test_summary, 0.74, "High", test_similar)
    print(explanation)
