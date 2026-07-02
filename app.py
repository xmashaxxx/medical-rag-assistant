"""
Medical RAG Assistant
Retrieval-Augmented Generation for medical Q&A
Original coursework: Llama 3.1 8B + ChromaDB + LangChain on Google Colab
Demo version: Claude embeddings (voyage-3-lite) + Claude Haiku generation
No local model downloads — runs entirely via Anthropic API
"""

import os
import re
import json
import tempfile
import streamlit as st
import anthropic

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide"
)

# ── Claude embedding class (replaces HuggingFace sentence-transformers) ───────

class ClaudeEmbeddings:
    """Embedding wrapper using Anthropic's voyage-3-lite model via the API."""

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results = []
        # Batch in groups of 10 to stay within API limits
        for i in range(0, len(texts), 10):
            batch = texts[i:i+10]
            response = self.client.embeddings.create(
                model="voyage-3-lite",
                input=batch
            )
            results.extend([r.embedding for r in response.data])
        return results

    def embed_query(self, text: str) -> list[float]:
        response = self.client.embeddings.create(
            model="voyage-3-lite",
            input=[text]
        )
        return response.data[0].embedding


# ── Simple in-memory vector store (no ChromaDB needed) ───────────────────────

class SimpleVectorStore:
    """Lightweight in-memory vector store with cosine similarity search."""

    def __init__(self):
        self.chunks = []      # list of str
        self.embeddings = []  # list of list[float]

    def add(self, chunks: list[str], embeddings_model: ClaudeEmbeddings):
        self.chunks = chunks
        self.embeddings = embeddings_model.embed_documents(chunks)

    def search(self, query: str, embeddings_model: ClaudeEmbeddings, k: int = 4) -> list[str]:
        import math
        query_vec = embeddings_model.embed_query(query)

        def cosine(a, b):
            dot = sum(x*y for x, y in zip(a, b))
            mag_a = math.sqrt(sum(x**2 for x in a))
            mag_b = math.sqrt(sum(x**2 for x in b))
            return dot / (mag_a * mag_b + 1e-9)

        scores = [(cosine(query_vec, emb), i) for i, emb in enumerate(self.embeddings)]
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:k]]


# ── Document loading & chunking ───────────────────────────────────────────────

def load_and_chunk_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """Load PDF with PyMuPDF and chunk into overlapping text segments."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        st.error("PyMuPDF not installed. Add `pymupdf` to requirements.txt.")
        st.stop()

    doc = fitz.open(pdf_path)
    full_text = "\n\n".join(page.get_text() for page in doc)
    doc.close()

    # Simple recursive chunking
    chunks = []
    start = 0
    while start < len(full_text):
        end = start + chunk_size
        chunk = full_text[start:end]
        # Try to end at a sentence boundary
        last_period = chunk.rfind(". ")
        if last_period > chunk_size // 2:
            chunk = chunk[:last_period + 1]
        chunks.append(chunk.strip())
        start += len(chunk) - chunk_overlap
    return [c for c in chunks if len(c) > 50]


def load_and_chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> list[str]:
    """Chunk raw text into overlapping segments."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        last_period = chunk.rfind(". ")
        if last_period > chunk_size // 2:
            chunk = chunk[:last_period + 1]
        chunks.append(chunk.strip())
        start += len(chunk) - chunk_overlap
    return [c for c in chunks if len(c) > 50]


# ── Default demo document (public domain medical content) ─────────────────────

DEFAULT_MEDICAL_TEXT = """
SEPSIS: RECOGNITION AND MANAGEMENT IN CRITICAL CARE

Definition and Overview
Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. It represents a medical emergency requiring immediate recognition and intervention. The condition progresses through stages from infection to sepsis to septic shock, with mortality increasing substantially at each stage.

Diagnostic Criteria (Sepsis-3 Definition)
Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection. Organ dysfunction is identified as an acute change in total SOFA score of 2 points or more. The SOFA score assesses six organ systems: respiratory (PaO2/FiO2 ratio), coagulation (platelets), liver (bilirubin), cardiovascular (mean arterial pressure and vasopressors), central nervous system (Glasgow Coma Scale), and renal (creatinine and urine output).

Septic shock is a subset of sepsis with particularly profound circulatory, cellular, and metabolic abnormalities. It is identified clinically by a vasopressor requirement to maintain MAP of 65 mmHg or greater and serum lactate greater than 2 mmol/L in the absence of hypovolemia.

Clinical Presentation
Early signs include fever (temperature greater than 38.3°C) or hypothermia (less than 36°C), tachycardia (heart rate greater than 90 beats per minute), tachypnea (respiratory rate greater than 20 breaths per minute), and altered mental status. Leukocytosis (white blood cell count greater than 12,000) or leukopenia (less than 4,000) may be present. Elevated C-reactive protein and procalcitonin support the diagnosis. Hypotension, defined as systolic blood pressure below 90 mmHg, is a late and serious finding.

Management Protocol (Surviving Sepsis Campaign Guidelines)
The Hour-1 Bundle includes five critical interventions to be completed within one hour of sepsis recognition.

First, measure lactate level. Re-measure lactate if initial lactate is greater than 2 mmol/L. Elevated lactate identifies tissue hypoperfusion even in the absence of hypotension.

Second, obtain blood cultures before administering antibiotics. At least two sets of blood cultures (aerobic and anaerobic) should be obtained from different sites before initiating antimicrobial therapy, provided this does not substantially delay antibiotic administration.

Third, administer broad-spectrum antibiotics immediately. Empiric broad-spectrum antibiotic therapy should be initiated within one hour of sepsis recognition. The choice of agent depends on the suspected source of infection, local resistance patterns, and patient factors including immune status and prior antibiotic exposure. Common regimens include piperacillin-tazobactam, meropenem, or vancomycin plus piperacillin-tazobactam for healthcare-associated infections.

Fourth, begin rapid administration of 30 mL/kg crystalloid for hypotension or lactate greater than or equal to 4 mmol/L. Normal saline or Ringer's lactate are the preferred crystalloid solutions. Fluid resuscitation should be followed by reassessment of hemodynamic status.

Fifth, apply vasopressors if the patient remains hypotensive during or after fluid resuscitation to maintain MAP of 65 mmHg or greater. Norepinephrine is the first-line vasopressor. Vasopressin may be added to norepinephrine to raise MAP or to decrease norepinephrine dosage.

Source Control
Identify and control the anatomic source of infection as rapidly as possible. This includes drainage of abscesses, debridement of infected necrotic tissue, removal of infected devices, and definitive control of ongoing microbial contamination. Imaging studies to identify source may include CT scan, ultrasound, or plain radiographs.

Pulmonary Embolism: Recognition and Management

Definition
Pulmonary embolism (PE) is the obstruction of one or more pulmonary arteries by a blood clot, typically originating from deep vein thrombosis in the lower extremities. It ranges from small peripheral emboli with minimal hemodynamic impact to massive central PE causing cardiovascular collapse.

Clinical Presentation
Common symptoms include sudden onset dyspnea (most common symptom), pleuritic chest pain, hemoptysis, tachycardia, and syncope. Physical examination may reveal tachycardia, tachypnea, decreased oxygen saturation, and signs of deep vein thrombosis including unilateral leg swelling and erythema. Massive PE may present with hypotension, cyanosis, and cardiovascular collapse.

Diagnostic Approach
The Wells score stratifies patients into low, moderate, and high pre-test probability. Variables include clinical signs of DVT (3 points), PE is the most likely diagnosis (3 points), heart rate greater than 100 beats per minute (1.5 points), immobilization or surgery in prior 4 weeks (1.5 points), prior DVT or PE (1.5 points), hemoptysis (1 point), and malignancy (1 point). Scores above 4 indicate high probability.

CT pulmonary angiography (CTPA) is the diagnostic standard. D-dimer testing is useful to rule out PE in low-probability patients with a negative result. Ventilation-perfusion scanning is an alternative when CTPA is contraindicated.

Treatment
Anticoagulation is the cornerstone of PE treatment. Options include unfractionated heparin (UFH) for massive PE or patients at high bleeding risk requiring possible intervention, low molecular weight heparin (LMWH) such as enoxaparin for stable patients, and direct oral anticoagulants (DOACs) including rivaroxaban or apixaban for stable patients without contraindications.

Systemic thrombolysis with alteplase is indicated for massive PE with hemodynamic instability. Catheter-directed thrombolysis or mechanical thrombectomy may be considered for submassive PE with right heart strain.

Rheumatoid Arthritis: Diagnosis and Treatment

Overview
Rheumatoid arthritis (RA) is a chronic, systemic autoimmune disease characterized by symmetric inflammatory polyarthritis, primarily affecting synovial joints. Without adequate treatment, progressive joint destruction leads to disability. The underlying mechanism involves autoimmune activation targeting synovial tissue, with key mediators including tumor necrosis factor alpha (TNF-alpha), interleukin-6 (IL-6), and activated T and B lymphocytes.

Diagnostic Criteria (2010 ACR/EULAR Classification)
The 2010 criteria require a score of 6 or more out of 10. Joint involvement scores: 1 large joint (0 points), 2-10 large joints (1 point), 1-3 small joints (2 points), 4-10 small joints (3 points), more than 10 joints including at least one small joint (5 points). Serology: negative RF and negative ACPA (0 points), low-positive RF or ACPA (2 points), high-positive RF or ACPA (3 points). Acute phase reactants: normal CRP and ESR (0 points), abnormal CRP or ESR (1 point). Duration: less than 6 weeks (0 points), 6 weeks or more (1 point).

Treatment Strategy
Treatment follows a treat-to-target approach aiming for remission or low disease activity.

First-line therapy uses conventional synthetic DMARDs (csDMARDs). Methotrexate is the anchor drug, typically initiated at 10-15 mg weekly and escalated to 20-25 mg weekly as tolerated. Folic acid supplementation (1-5 mg daily) reduces side effects. Hydroxychloroquine and sulfasalazine may be combined with methotrexate in triple therapy.

Biologic DMARDs are used when csDMARDs fail to achieve treatment targets. TNF inhibitors (etanercept, adalimumab, infliximab, certolizumab, golimumab) are first-line biologics. IL-6 inhibitors (tocilizumab, sarilumab) are alternatives. Abatacept (T-cell co-stimulation inhibitor) and rituximab (B-cell depleting agent) are additional options.

Targeted synthetic DMARDs include JAK inhibitors (tofacitinib, baricitinib, upadacitinib) for patients who fail biologics.

NSAIDs and glucocorticoids provide symptomatic relief and bridge therapy during DMARD initiation but are not disease-modifying.

Hypertension: Classification and Management

Classification
Blood pressure is classified as normal (less than 120/80 mmHg), elevated (120-129 systolic and less than 80 diastolic), Stage 1 hypertension (130-139 systolic or 80-89 diastolic), and Stage 2 hypertension (140 or higher systolic or 90 or higher diastolic). Hypertensive urgency is defined as severely elevated blood pressure (greater than 180/120 mmHg) without target organ damage. Hypertensive emergency involves the same blood pressure elevation with acute target organ damage including hypertensive encephalopathy, acute coronary syndrome, acute heart failure, aortic dissection, or acute kidney injury.

First-Line Medications for Hypertension
Thiazide diuretics such as hydrochlorothiazide and chlorthalidone reduce sodium and water retention. Calcium channel blockers including amlodipine (dihydropyridine) and diltiazem (non-dihydropyridine) reduce vascular resistance. ACE inhibitors such as lisinopril and enalapril block conversion of angiotensin I to angiotensin II. Angiotensin receptor blockers (ARBs) including losartan and valsartan block angiotensin II receptors. Trade names: Lisinopril (Zestril, Prinivil), Amlodipine (Norvasc), Losartan (Cozaar), Hydrochlorothiazide (Microzide), Metoprolol (Lopressor, Toprol-XL), Atenolol (Tenormin).

Compelling indications guide drug selection. Heart failure with reduced ejection fraction: ACE inhibitor or ARB plus beta-blocker plus diuretic. Diabetes: ACE inhibitor or ARB preferred for renoprotection. Chronic kidney disease with proteinuria: ACE inhibitor or ARB. Post-myocardial infarction: beta-blocker plus ACE inhibitor.

Endocrine Disorders: Diagnostic Approach

Thyroid Disease
Hypothyroidism presents with fatigue, weight gain, cold intolerance, constipation, dry skin, and bradycardia. TSH is elevated with low free T4. Treatment is levothyroxine, dosed to normalize TSH. Hyperthyroidism presents with weight loss, heat intolerance, palpitations, tremor, and diarrhea. TSH is suppressed with elevated free T4 or T3. Causes include Graves disease (autoimmune, with TSH receptor antibodies), toxic multinodular goiter, and thyroiditis.

Diabetes Mellitus Diagnostic Criteria
Fasting plasma glucose of 126 mg/dL or higher (confirmed on repeat testing), 2-hour plasma glucose of 200 mg/dL or higher during 75-g oral glucose tolerance test, HbA1c of 6.5% or higher, or random plasma glucose of 200 mg/dL or higher with symptoms of hyperglycemia.

Adrenal Insufficiency
Primary adrenal insufficiency (Addison disease) presents with fatigue, weight loss, hyperpigmentation, and electrolyte disturbances (hyponatremia, hyperkalemia). Cortisol stimulation test with cosyntropin (synthetic ACTH) is diagnostic when peak cortisol response is below 18-20 mcg/dL. Treatment is hydrocortisone replacement plus fludrocortisone for mineralocorticoid deficiency.
"""


# ── Generation ────────────────────────────────────────────────────────────────

def generate_answer(query: str, context: str, temperature: float,
                    top_p: float, max_tokens: int, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)

    system = """You are a medical information assistant. Answer the user's question
using ONLY the provided context from the medical reference. Be precise, structured,
and clinically accurate. If the context does not contain enough information, say so explicitly.
Do not add information not present in the context."""

    user_msg = f"""Context from medical reference:
{context}

Question: {query}

Provide a clear, structured medical answer based solely on the context above."""

    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": user_msg}]
    )
    return msg.content[0].text.strip()


# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🏥 Medical RAG Assistant")
st.caption(
    "Retrieval-Augmented Generation for medical Q&A · "
    "Coursework pipeline (Llama 3.1 8B + ChromaDB) adapted for deployment with Claude"
)

# Check API key
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    st.error(
        "**ANTHROPIC_API_KEY is not set.**\n\n"
        "Set it in Streamlit Secrets or locally:\n"
        "```\nexport ANTHROPIC_API_KEY=sk-ant-...\nstreamlit run app.py\n```"
    )
    st.stop()

# Sidebar
st.sidebar.header("Configuration")
uploaded_pdf = st.sidebar.file_uploader(
    "Upload your own medical PDF (optional)",
    type="pdf",
    help="Upload to override the built-in demo content. "
         "The Merck Manual works great here."
)

st.sidebar.markdown("---")
st.sidebar.subheader("RAG Parameters")
chunk_size = st.sidebar.slider("Chunk size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 400, 150, 50)
top_k = st.sidebar.slider("Retrieved chunks (k)", 2, 8, 4)

st.sidebar.markdown("---")
st.sidebar.subheader("Generation Parameters")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.05)
top_p = st.sidebar.slider("Top-p", 0.5, 1.0, 1.0, 0.05)
max_tokens = st.sidebar.slider("Max tokens", 256, 1024, 512, 64)

# Tabs
tab1, tab2 = st.tabs(["Ask a Question", "About This Project"])

with tab1:
    # Build vector store
    cache_key = f"vs_{chunk_size}_{chunk_overlap}_{uploaded_pdf.name if uploaded_pdf else 'default'}"

    if cache_key not in st.session_state:
        embeddings_model = ClaudeEmbeddings(api_key=api_key)

        if uploaded_pdf is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name
            with st.spinner(f"Chunking and embedding {uploaded_pdf.name}..."):
                chunks = load_and_chunk_pdf(tmp_path, chunk_size, chunk_overlap)
                vs = SimpleVectorStore()
                vs.add(chunks, embeddings_model)
            st.session_state[cache_key] = (vs, len(chunks), uploaded_pdf.name)
            os.unlink(tmp_path)
        else:
            with st.spinner("Preparing demo medical knowledge base..."):
                chunks = load_and_chunk_text(DEFAULT_MEDICAL_TEXT, chunk_size, chunk_overlap)
                vs = SimpleVectorStore()
                vs.add(chunks, embeddings_model)
            st.session_state[cache_key] = (vs, len(chunks), "built-in demo content")

    vs, n_chunks, source_name = st.session_state[cache_key]
    embeddings_model = ClaudeEmbeddings(api_key=api_key)

    st.success(f"✅ Ready — {n_chunks} chunks indexed from **{source_name}**")

    # Example questions
    st.markdown("**Example questions** (click to use):")
    examples = [
        "What is the protocol for managing sepsis in a critical care unit?",
        "What are the common symptoms and treatments for pulmonary embolism?",
        "What are the first-line options for managing rheumatoid arthritis?",
        "What are the diagnostic steps for suspected endocrine disorders?",
        "Can you provide the trade names of medications used for treating hypertension?",
    ]

    cols = st.columns(2)
    for i, ex in enumerate(examples):
        if cols[i % 2].button(ex, key=f"ex_{i}", use_container_width=True):
            st.session_state["query_input"] = ex

    query = st.text_input(
        "Ask a medical question",
        value=st.session_state.get("query_input", ""),
        placeholder="What is the protocol for managing sepsis?"
    )

    if st.button("Ask", type="primary") and query:
        col1, col2 = st.columns([1, 1])

        with st.spinner("Retrieving relevant passages..."):
            context_chunks = vs.search(query, embeddings_model, k=top_k)
            context = "\n\n---\n\n".join(context_chunks)

        with col1:
            st.subheader("📄 Retrieved Context")
            st.caption(f"Top {top_k} chunks · chunk_size={chunk_size} · overlap={chunk_overlap}")
            with st.expander("View retrieved passages", expanded=False):
                for i, chunk in enumerate(context_chunks, 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.text(chunk[:400] + ("..." if len(chunk) > 400 else ""))
                    st.markdown("---")

        with col2:
            st.subheader("🤖 Generated Answer")
            st.caption(f"Claude Haiku · temperature={temperature} · top_p={top_p}")
            with st.spinner("Generating answer..."):
                answer = generate_answer(query, context, temperature, top_p, max_tokens, api_key)
            st.markdown(answer)

            st.info(
                "**Parameter note:** The original coursework tested 5 configurations "
                "using Llama 3.1 8B. Adjust temperature above to see the same effect "
                "documented in the notebook — higher values produce more detailed but "
                "potentially less grounded responses."
            )

with tab2:
    st.header("About This Project")
    st.markdown("""
### Coursework Context

This is the deployed version of my NLP/RAG coursework project, originally built in
Google Colab using **Meta Llama 3.1 8B Instruct** (Q5_K_M quantized GGUF) on a Tesla T4 GPU,
with **ChromaDB** as the vector store and **LangChain** as the orchestration framework.

The deployed demo replaces:
- Llama 3.1 8B → **Claude Haiku** (generation)
- sentence-transformers → **Claude voyage-3-lite** (embeddings, via Anthropic API)
- ChromaDB → **in-memory cosine similarity search** (no heavy dependencies)

This makes the demo run instantly on Streamlit Cloud with no model downloads.

### Technical Stack Comparison

| Component | Coursework | This Demo |
|-----------|-----------|-----------|
| Document loading | PyMuPDF | PyMuPDF |
| Chunking | RecursiveCharacterTextSplitter | Custom overlap chunker |
| Embeddings | all-MiniLM-L6-v2 (local) | voyage-3-lite (API) |
| Vector store | ChromaDB | In-memory cosine similarity |
| Generation | Llama 3.1 8B (quantized) | Claude Haiku |
| Infrastructure | Google Colab + Tesla T4 | Streamlit Cloud |

### Evaluation Results from Coursework

5 parameter combinations tested across 5 clinical questions:

| Combination | Settings | Finding |
|-------------|----------|---------|
| 1 (baseline) | temperature=0 | Deterministic, structured |
| 2 | temperature=0.7 | More detailed, added context |
| 3 | temperature=0.5, top_p=0.85 | Most focused |
| 4 | temperature=0, repeat_penalty=1.3 | Less repetition |
| 5 | temperature=0.9, seed=99 | Most varied |

**Key finding:** temperature=0 produced the most clinically reliable responses.

### Connection to research_assistant

This project directly preceded the [research_assistant](https://github.com/xmashaxxx/research_assistant)
agentic pipeline. Lessons carried forward: RAG reduces hallucination vs closed-book LLMs,
chunking strategy significantly affects retrieval quality, and separating retrieval from
generation allows independent optimization of each stage.
    """)
