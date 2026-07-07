# RAG Document Chatbot — Example Walkthrough

This shows what the app looks like in action, using the bundled sample documents
(`sample_docs/company_handbook.txt` and `sample_docs/product_faq.txt`). If you'd
rather try it live, click the deployed link and use "Use bundled sample docs" in
the sidebar — no upload needed.

---

### Example 1 — Direct fact retrieval

**Q: How many PTO days do employees get, and how does that change over time?**

> Employees accrue 15 days of paid time off (PTO) per year. After three years of
> tenure, this increases to 20 days per year.

**Sources used:**
`company_handbook.txt` — *"Full-time employees accrue 15 days of paid time off
(PTO) per year, increasing to 20 days after three years of tenure..."*

---

### Example 2 — Cross-section synthesis

**Q: What's the API rate limit on the Business tier, and what happens if I exceed it?**

> On the Business tier, the rate limit is 1,000 requests per minute. Requests
> beyond that limit receive a 429 status code.

**Sources used:**
`product_faq.txt` — *"Requests beyond your plan's rate limit will receive a 429
status code... Business tier to 1,000 requests per minute..."*

---

### Example 3 — Grounding / hallucination check (the important one)

**Q: What's the company's policy on relocation assistance?**

> The provided documents don't contain any information about relocation
> assistance. I can't answer that based on the available context.

This is the RAG grounding working as intended: rather than guessing, the model
recognizes the answer isn't in the retrieved context and says so — instead of
fabricating a policy that doesn't exist in the source documents.

---

### What this demonstrates

| Capability | Where it shows up |
|---|---|
| Document ingestion pipeline | Load → chunk → embed → store (see sidebar chunk counts) |
| Semantic retrieval | Correct chunks pulled per question, shown in "Sources used" |
| Grounded generation | Answers tied to retrieved text, not general knowledge |
| Hallucination mitigation | Explicit "I don't know" behavior on out-of-scope questions |
| Source transparency | Every answer cites which document/passage it came from |
