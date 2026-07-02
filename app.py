"""
Medical RAG Assistant
Retrieval-Augmented Generation for medical Q&A
Original coursework: Llama 3.1 8B + ChromaDB + LangChain on Google Colab
Demo: TF-IDF retrieval + Claude Haiku generation — no heavy dependencies
"""
import os
import re
import math
import tempfile
import streamlit as st
import anthropic
 
st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🏥",
    layout="wide"
)
 
# ── TF-IDF Vector Store (no external dependencies) ───────────────────────────
 
class TFIDFVectorStore:
    """Simple TF-IDF retrieval — no model downloads, works on any machine."""
 
    def __init__(self):
        self.chunks = []
        self.tfidf_matrix = []
        self.vocab = {}
 
    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
 
    def _build_vocab(self, all_tokens: list[list[str]]):
        words = set(w for tokens in all_tokens for w in tokens)
        self.vocab = {w: i for i, w in enumerate(sorted(words))}
 
    def _tfidf(self, tokens: list[str], all_token_lists: list[list[str]]) -> list[float]:
        n_docs = len(all_token_lists)
        vec = [0.0] * len(self.vocab)
        tf_counts = {}
        for t in tokens:
            tf_counts[t] = tf_counts.get(t, 0) + 1
        for word, idx in self.vocab.items():
            tf = tf_counts.get(word, 0) / max(len(tokens), 1)
            df = sum(1 for tl in all_token_lists if word in tl)
            idf = math.log((n_docs + 1) / (df + 1)) + 1
            vec[idx] = tf * idf
        return vec
 
    def _cosine(self, a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        mag = math.sqrt(sum(x**2 for x in a)) * math.sqrt(sum(x**2 for x in b))
        return dot / (mag + 1e-9)
 
    def add(self, chunks: list[str]):
        self.chunks = chunks
        all_tokens = [self._tokenize(c) for c in chunks]
        self._build_vocab(all_tokens)
        self.tfidf_matrix = [self._tfidf(t, all_tokens) for t in all_tokens]
 
    def search(self, query: str, k: int = 4) -> list[str]:
        q_tokens = self._tokenize(query)
        q_vec = self._tfidf(q_tokens, [self._tokenize(c) for c in self.chunks])
        scores = [(self._cosine(q_vec, doc_vec), i)
                  for i, doc_vec in enumerate(self.tfidf_matrix)]
        scores.sort(reverse=True)
        return [self.chunks[i] for _, i in scores[:k]]
 
 
# ── Document chunking ─────────────────────────────────────────────────────────
 
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        last_period = chunk.rfind(". ")
        if last_period > chunk_size // 2:
            chunk = chunk[:last_period + 1]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += len(chunk) - overlap
    return chunks

@st.cache_resource
def build_default_store():
    """Build the default TF-IDF store once and cache it."""
    chunks = chunk_text(DEFAULT_TEXT, 800, 100)
    vs = TFIDFVectorStore()
    vs.add(chunks)
    return vs, len(chunks)
 
def load_pdf(pdf_path: str) -> str:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    except Exception as e:
        st.error(f"Could not read PDF: {e}")
        st.stop()
 
 
# ── Built-in demo content ─────────────────────────────────────────────────────
 
DEFAULT_TEXT = """
SEPSIS: RECOGNITION AND MANAGEMENT IN CRITICAL CARE
 
Definition
Sepsis is a life-threatening organ dysfunction caused by a dysregulated host response to infection. The Sepsis-3 definition identifies organ dysfunction as an acute change in total SOFA score of 2 points or more. Septic shock is a subset of sepsis with vasopressor requirement to maintain MAP of 65 mmHg or greater and serum lactate greater than 2 mmol/L in the absence of hypovolemia.
 
Clinical Presentation
Early signs include fever greater than 38.3 degrees Celsius or hypothermia less than 36 degrees Celsius, tachycardia greater than 90 beats per minute, tachypnea greater than 20 breaths per minute, and altered mental status. Leukocytosis above 12,000 or leukopenia below 4,000 may be present. Elevated C-reactive protein and procalcitonin support the diagnosis.
 
Management Protocol - Hour-1 Bundle
First: Measure lactate level. Re-measure if initial lactate is greater than 2 mmol/L to identify tissue hypoperfusion.
Second: Obtain blood cultures before administering antibiotics. At least two sets from different sites.
Third: Administer broad-spectrum antibiotics immediately within one hour of sepsis recognition. Common regimens include piperacillin-tazobactam or meropenem. Vancomycin is added for suspected MRSA.
Fourth: Administer 30 mL/kg crystalloid for hypotension or lactate greater than or equal to 4 mmol/L. Normal saline or Ringer's lactate are preferred.
Fifth: Apply vasopressors if hypotension persists to maintain MAP of 65 mmHg or greater. Norepinephrine is first-line. Vasopressin may be added to norepinephrine.
 
Source Control
Identify and control the anatomic source of infection as rapidly as possible. This includes drainage of abscesses, debridement of infected necrotic tissue, removal of infected devices.
 
PULMONARY EMBOLISM: RECOGNITION AND MANAGEMENT
 
Definition and Presentation
Pulmonary embolism (PE) is obstruction of pulmonary arteries by blood clot, typically from deep vein thrombosis. Symptoms include sudden onset dyspnea (most common), pleuritic chest pain, hemoptysis, tachycardia, and syncope. Massive PE presents with hypotension, cyanosis, and cardiovascular collapse. Physical examination may show tachycardia, decreased oxygen saturation, and unilateral leg swelling suggesting DVT.
 
Diagnostic Approach
The Wells score stratifies pre-test probability. Clinical signs of DVT score 3 points. PE is the most likely diagnosis scores 3 points. Heart rate greater than 100 scores 1.5 points. Immobilization or surgery in prior 4 weeks scores 1.5 points. Prior DVT or PE scores 1.5 points. Hemoptysis scores 1 point. Malignancy scores 1 point. Scores above 4 indicate high probability.
CT pulmonary angiography (CTPA) is the diagnostic standard. D-dimer is useful to rule out PE in low-probability patients.
 
Treatment
Anticoagulation is the cornerstone. Unfractionated heparin (UFH) is used for massive PE or patients requiring intervention. Low molecular weight heparin such as enoxaparin is used for stable patients. Direct oral anticoagulants including rivaroxaban or apixaban are used for stable patients without contraindications.
Systemic thrombolysis with alteplase is indicated for massive PE with hemodynamic instability. Catheter-directed thrombolysis or mechanical thrombectomy may be considered for submassive PE with right heart strain.
 
RHEUMATOID ARTHRITIS: DIAGNOSIS AND TREATMENT
 
Overview
Rheumatoid arthritis (RA) is a chronic systemic autoimmune disease causing symmetric inflammatory polyarthritis primarily affecting synovial joints. Key mediators include tumor necrosis factor alpha (TNF-alpha), interleukin-6 (IL-6), and activated T and B lymphocytes.
 
2010 ACR/EULAR Diagnostic Criteria
Scores of 6 or more out of 10 are required. Joint involvement: 1 large joint (0), 2-10 large joints (1), 1-3 small joints (2), 4-10 small joints (3), more than 10 joints including small joints (5). Serology: negative RF and ACPA (0), low-positive (2), high-positive (3). Acute phase reactants: normal CRP and ESR (0), abnormal (1). Duration: less than 6 weeks (0), 6 weeks or more (1).
 
Treatment - Treat-to-Target Strategy
First-line therapy uses conventional synthetic DMARDs. Methotrexate is the anchor drug, initiated at 10-15 mg weekly and escalated to 20-25 mg weekly. Folic acid supplementation reduces side effects. Hydroxychloroquine and sulfasalazine may be combined as triple therapy.
 
Biologic DMARDs are used when csDMARDs fail. TNF inhibitors include etanercept, adalimumab (Humira), infliximab (Remicade), certolizumab, and golimumab. IL-6 inhibitors include tocilizumab and sarilumab. Abatacept inhibits T-cell co-stimulation. Rituximab depletes B-cells.
 
Targeted synthetic DMARDs include JAK inhibitors: tofacitinib (Xeljanz), baricitinib (Olumiant), and upadacitinib (Rinvoq). NSAIDs and glucocorticoids provide symptomatic relief and bridge therapy.
 
HYPERTENSION: CLASSIFICATION AND MANAGEMENT
 
Blood Pressure Classification
Normal: less than 120/80 mmHg. Elevated: 120-129 systolic and less than 80 diastolic. Stage 1 hypertension: 130-139 systolic or 80-89 diastolic. Stage 2 hypertension: 140 or higher systolic or 90 or higher diastolic. Hypertensive urgency: greater than 180/120 mmHg without target organ damage. Hypertensive emergency: same elevation with acute target organ damage.
 
First-Line Medications and Trade Names
Thiazide diuretics: Hydrochlorothiazide (Microzide), Chlorthalidone. Calcium channel blockers: Amlodipine (Norvasc), Diltiazem (Cardizem). ACE inhibitors: Lisinopril (Zestril, Prinivil), Enalapril (Vasotec), Ramipril (Altace). Angiotensin receptor blockers (ARBs): Losartan (Cozaar), Valsartan (Diovan), Olmesartan (Benicar). Beta-blockers: Metoprolol (Lopressor, Toprol-XL), Atenolol (Tenormin), Carvedilol (Coreg).
 
Compelling Indications for Drug Selection
Heart failure with reduced ejection fraction: ACE inhibitor or ARB plus beta-blocker plus diuretic. Diabetes mellitus: ACE inhibitor or ARB preferred for renoprotection. Chronic kidney disease with proteinuria: ACE inhibitor or ARB. Post-myocardial infarction: beta-blocker plus ACE inhibitor. Afro-Caribbean patients: calcium channel blocker or thiazide preferred over ACE inhibitor monotherapy.
 
ENDOCRINE DISORDERS: DIAGNOSTIC APPROACH
 
Thyroid Disease
Hypothyroidism symptoms include fatigue, weight gain, cold intolerance, constipation, dry skin, and bradycardia. TSH is elevated with low free T4. Treatment is levothyroxine (Synthroid) dosed to normalize TSH.
Hyperthyroidism symptoms include weight loss, heat intolerance, palpitations, tremor, and diarrhea. TSH is suppressed with elevated free T4 or T3. Graves disease is autoimmune with TSH receptor antibodies. Treatment options include methimazole, radioactive iodine, or thyroidectomy.
 
Diabetes Mellitus Diagnostic Criteria
Fasting plasma glucose of 126 mg/dL or higher on repeat testing. Two-hour plasma glucose of 200 mg/dL or higher during 75-g oral glucose tolerance test. HbA1c of 6.5% or higher. Random plasma glucose of 200 mg/dL or higher with symptoms of hyperglycemia. Prediabetes is defined as HbA1c 5.7 to 6.4%, fasting glucose 100 to 125 mg/dL, or 2-hour glucose 140 to 199 mg/dL.
 
Adrenal Insufficiency
Primary adrenal insufficiency (Addison disease) presents with fatigue, weight loss, hyperpigmentation, hyponatremia, and hyperkalemia. Morning cortisol below 3 mcg/dL is diagnostic. The cosyntropin (synthetic ACTH) stimulation test is the gold standard — peak cortisol below 18-20 mcg/dL at 30 or 60 minutes is diagnostic. Treatment is hydrocortisone 15-25 mg daily in divided doses plus fludrocortisone 0.05-0.2 mg daily for mineralocorticoid replacement.
 
Cushing Syndrome
Caused by excess cortisol from pituitary adenoma (Cushing disease), adrenal tumor, or exogenous glucocorticoids. Features include central obesity, moon face, buffalo hump, purple striae, hypertension, diabetes, and osteoporosis. Screening tests include 24-hour urine free cortisol, late-night salivary cortisol, and 1 mg overnight dexamethasone suppression test.
 
Pheochromocytoma
A catecholamine-secreting tumor of the adrenal medulla causing hypertensive crises, headache, palpitations, and diaphoresis. Diagnosis by 24-hour urine metanephrines and catecholamines or plasma metanephrines. Surgical resection after alpha-blockade with phenoxybenzamine or doxazosin followed by beta-blockade.
"""
 
 
# ── Claude generation ─────────────────────────────────────────────────────────
 
def generate_answer(query: str, context: str, temperature: float,
                    top_p: float, max_tokens: int, api_key: str) -> str:
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=max_tokens,
        temperature=temperature,
        system=(
            "You are a medical information assistant. Answer the user's question "
            "using ONLY the provided context from the medical reference. Be precise, "
            "structured, and clinically accurate. If the context does not contain "
            "enough information to answer fully, say so explicitly. "
            "Do not add information not present in the context."
        ),
        messages=[{"role": "user", "content":
            f"Context from medical reference:\n{context}\n\nQuestion: {query}\n\n"
            "Provide a clear, structured medical answer based solely on the context above."
        }]
    )
    return msg.content[0].text.strip()
 
 
# ── UI ────────────────────────────────────────────────────────────────────────
 
st.title("🏥 Medical RAG Assistant")
st.caption(
    "Retrieval-Augmented Generation for medical Q&A · "
    "Coursework pipeline (Llama 3.1 8B + ChromaDB) adapted for deployment with Claude Haiku"
)
 
api_key = os.environ.get("ANTHROPIC_API_KEY")
if not api_key:
    st.error(
        "**ANTHROPIC_API_KEY is not set.** Set it in Streamlit Secrets:\n\n"
        "```\nANTHROPIC_API_KEY = \"sk-ant-...\"\n```"
    )
    st.stop()
 
# Sidebar
st.sidebar.header("Configuration")
uploaded_pdf = st.sidebar.file_uploader(
    "Upload your own medical PDF (optional)",
    type="pdf",
    help="Upload to override the built-in demo content."
)
st.sidebar.markdown("---")
st.sidebar.subheader("RAG Parameters")
chunk_size = st.sidebar.slider("Chunk size", 500, 2000, 1000, 100)
chunk_overlap = st.sidebar.slider("Chunk overlap", 0, 400, 150, 50)
top_k = st.sidebar.slider("Retrieved chunks (k)", 2, 8, 4)
st.sidebar.markdown("---")
st.sidebar.subheader("Generation Parameters")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.0, 0.05,
    help="0 = deterministic (best for clinical Q&A). Tested in coursework: combos 1-5.")
top_p = st.sidebar.slider("Top-p", 0.5, 1.0, 1.0, 0.05)
max_tokens = st.sidebar.slider("Max tokens", 256, 1024, 512, 64)
 
tab1, tab2 = st.tabs(["Ask a Question", "About This Project"])
 
with tab1:
    # Build vector store (cached by parameters + source)
    source_id = uploaded_pdf.name if uploaded_pdf else "default"
    cache_key = f"vs_{chunk_size}_{chunk_overlap}_{source_id}"
 
    if cache_key not in st.session_state:
        if uploaded_pdf is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_pdf.read())
                tmp_path = tmp.name
            with st.spinner(f"Indexing {uploaded_pdf.name}..."):
                text = load_pdf(tmp_path)
                chunks = chunk_text(text, chunk_size, chunk_overlap)
                vs = TFIDFVectorStore()
                vs.add(chunks)
                os.unlink(tmp_path)
            source_label = uploaded_pdf.name
        else:
            vs, n = build_default_store()
            source_label = "built-in demo content (sepsis, PE, RA, hypertension, endocrine)"
            st.session_state[cache_key] = (vs, n, source_label)
 
    vs, n_chunks, source_label = st.session_state[cache_key]
    st.success(f"✅ Ready — {n_chunks} chunks indexed from **{source_label}**")
 
    # Example question buttons
    st.markdown("**Try these questions from the coursework evaluation:**")
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
            st.session_state["query"] = ex
 
    query = st.text_input(
        "Ask a medical question",
        value=st.session_state.get("query", ""),
        placeholder="What is the protocol for managing sepsis?"
    )
 
    if st.button("Ask", type="primary") and query:
        col1, col2 = st.columns([1, 1])
 
        context_chunks = vs.search(query, k=top_k)
        context = "\n\n---\n\n".join(context_chunks)
 
        with col1:
            st.subheader("📄 Retrieved Context")
            st.caption(f"Top {top_k} chunks · TF-IDF keyword search")
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
                "**Coursework note:** The original system used semantic embeddings "
                "(all-MiniLM-L6-v2) for retrieval. This demo uses TF-IDF keyword "
                "search for instant loading with no dependencies. "
                "Upload your own PDF to test with different documents."
            )
 
with tab2:
    st.header("About This Project")
    st.markdown("""
### Coursework Context
 
This is the deployed version of my NLP/RAG coursework project, originally built in
Google Colab using **Meta Llama 3.1 8B Instruct** (Q5_K_M quantized GGUF) on a Tesla T4 GPU,
with **ChromaDB** as the vector store and **LangChain** as the orchestration framework.
 
The deployed demo replaces:
- Llama 3.1 8B → **Claude Haiku** (generation, via Anthropic API)
- sentence-transformers → **TF-IDF keyword retrieval** (instant, zero dependencies)
- ChromaDB → **in-memory search** (no installation required)
 
The retrieval quality difference between semantic embeddings and TF-IDF is documented in the notebook.
 
### Technical Stack Comparison
 
| Component | Coursework | This Demo |
|-----------|-----------|-----------|
| Document loading | PyMuPDF | PyMuPDF |
| Chunking | RecursiveCharacterTextSplitter | Overlap chunker |
| Retrieval | Semantic (all-MiniLM-L6-v2) | TF-IDF keyword search |
| Vector store | ChromaDB | In-memory |
| Generation | Llama 3.1 8B (quantized) | Claude Haiku |
| Infrastructure | Google Colab + Tesla T4 | Streamlit Cloud |
 
### Coursework Evaluation Results
 
5 parameter combinations tested across 5 clinical questions (sepsis, pulmonary embolism,
rheumatoid arthritis, endocrine disorders, hypertension medications):
 
| Combination | Settings | Key Finding |
|-------------|----------|-------------|
| 1 (baseline) | temperature=0 | Deterministic, most clinically precise |
| 2 | temperature=0.7 | More detailed, added context |
| 3 | temperature=0.5, top_p=0.85, top_k=40 | Most focused, least repetitive |
| 4 | temperature=0, repeat_penalty=1.3 | Similar to baseline, less redundancy |
| 5 | temperature=0.9, seed=99 | Most varied output |
 
**Key finding:** temperature=0 produced the most clinically reliable responses.
Higher temperatures surfaced more detail but risked adding information beyond retrieved context.
 
### Connection to research_assistant
 
This project directly preceded the [research_assistant](https://github.com/xmashaxxx/research_assistant)
agentic pipeline. Key lessons carried forward:
- RAG reduces hallucination versus closed-book LLMs
- Chunking strategy significantly affects retrieval quality  
- Separating retrieval from generation allows independent optimization
- Schema-driven structured extraction (used in research_assistant) evolved from lessons here
    """)
 