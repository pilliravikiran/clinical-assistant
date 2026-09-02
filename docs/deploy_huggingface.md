# Deploying to Hugging Face Spaces

This deploys the **whole app as one Streamlit Space**. The Streamlit UI runs the
RAG **in-process** (no separate backend to manage), so a single free Space serves
everything.

> The app is designed to run **even with no secrets**: it falls back to
> `APP_MODE=offline` (mock answer-writer) + `VECTOR_BACKEND=local` (in-memory
> vectors). Adding secrets upgrades it to real Claude + Pinecone.

---

## How it works on HF

- HF reads the YAML header at the top of `README.md` (`sdk: streamlit`,
  `app_file: streamlit_app.py`) and runs `streamlit run streamlit_app.py`.
- `streamlit_app.py` tries to reach a backend API; on HF there is none, so it
  imports the services and runs the RAG **in-process** (loads models, ingests
  `data/`, answers questions).
- Models (BGE embeddings + cross-encoder reranker) download from the HF Hub on
  first start (~30-60s cold start). Pinecone/Claude are reached over HTTPS
  (no corporate SSL interception in the cloud, so `LLM_VERIFY_SSL=true`).

---

## Step-by-step

### 1. Create a Hugging Face account
Go to https://huggingface.co/join (free). Verify your email.

### 2. Create a Space
- Click your avatar -> **New Space** (or https://huggingface.co/new-space).
- **Owner:** you. **Space name:** `clinical-assistant`.
- **License:** MIT. **SDK:** **Streamlit**. **Hardware:** *CPU basic* (free).
- **Visibility:** Public (portfolio) or Private.
- Click **Create Space**. HF creates an empty git repo at
  `https://huggingface.co/spaces/<your-username>/clinical-assistant`.

### 3. Set variables in the Space
In the Space: **Settings -> Variables and secrets**. Two paths:

**A) Offline demo (recommended first) — set just ONE variable:**

| Name | Value | Kind |
|---|---|---|
| `VECTOR_BACKEND` | `local` | variable |

That's all. `APP_MODE` already defaults to `offline` (mock answer-writer), and
`local` uses an in-memory vector store — so no API keys are needed and nothing
can crash on a missing key. (If you leave `VECTOR_BACKEND` unset it defaults to
`pinecone`, which WOULD fail without a key — so set it to `local` for the demo.)

**B) Full demo (real Claude + Pinecone) — add these instead:**

| Name | Value | Kind |
|---|---|---|
| `APP_MODE` | `online` | variable |
| `LLM_API_KEY` | your Anthropic key | **secret** |
| `LLM_MODEL` | `claude-sonnet-5` | variable |
| `VECTOR_BACKEND` | `pinecone` | variable |
| `PINECONE_API_KEY` | your Pinecone key | **secret** |
| `PINECONE_INDEX_NAME` | your index name (e.g. `tenet-clinical`) | variable |
| `PINECONE_CLOUD` | `aws` | variable |
| `PINECONE_REGION` | `us-east-1` | variable |
| `LLM_VERIFY_SSL` | `true` | variable |

> You can start with (A) and switch to (B) later by editing the variables —
> the Space redeploys automatically.

### 4. Push the code to the Space
From the project folder (`C:\Users\MMS\Documents\Github\clinical-assistant`):

```bash
git remote add space https://huggingface.co/spaces/<your-username>/clinical-assistant
git push space main
```

HF will ask for a username + password. The **password is an HF access token**
(create at https://huggingface.co/settings/tokens -> New token -> type **Write**).

> Never commit `.env` or tokens. `.gitignore` already blocks them; secrets live
> only in the HF Settings panel.

### 5. Watch it build
The Space page shows **Building** -> **Running**. Open the **Logs** tab to watch
pip install + model download. First build takes a few minutes (installs torch).

### 6. Use it
When it says **Running**, the chat UI is live at
`https://huggingface.co/spaces/<your-username>/clinical-assistant`. Ask
*"What follow-up care was recommended?"*

---

## Updating later
Edit code -> commit -> `git push space main`. HF rebuilds automatically.

## Troubleshooting
- **Build fails on torch/memory:** switch Hardware to *CPU upgrade* temporarily,
  or ensure only needed deps are installed.
- **"metrics unavailable" / mock answers:** secrets not set or wrong names -> see step 3.
- **Slow first question:** models are downloading; subsequent questions are fast.
- **Pinecone errors:** check `PINECONE_API_KEY` and `PINECONE_INDEX_NAME` match your
  Pinecone console; the index dimension must be **384** (BGE-small).
