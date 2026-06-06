# InsuredAI — Example Queries Guide

A complete reference of what you can ask InsuredAI and what to expect back.
InsuredAI has three specialised AI agents — each handles a different type of question.

---

## How the routing works

| You ask about... | Agent used | Source |
|---|---|---|
| Statistics, counts, trends across claims | **SQL Agent** | Live PostgreSQL database |
| Policy rules, procedures, legal terms from documents | **RAG Agent** | Real AXA, Direct Line, Admiral PDFs |
| Insurance definitions, concepts, terminology | **RAG Agent** | Groq AI knowledge (fallback) |
| A specific claim by CLM ID | **ML Agent** | XGBoost models |
| Complex questions needing both data + policy | **SQL + RAG** | Database + PDFs combined |

---

## 🗄️ SQL Agent — Claims Data & Analytics

These questions query your live 10,000-row PostgreSQL claims database.
You get real numbers, percentages, and trends.

### Volume & counts
```
How many claims are there by claim type?
How many open claims are there by region?
Which month had the highest number of claims in 2023?
How many claims were filed in Q1 vs Q4?
How many claims are currently in review?
```

### Fraud analysis
```
Which regions have the highest average fraud score?
Which claim type has the highest fraud rate?
Show me the top 5 adjusters by number of fraud-flagged claims
What percentage of liability claims are flagged as fraud?
How many claims have a fraud score above 0.5?
```

### Litigation analysis
```
What percentage of vehicle claims are flagged for litigation?
Which regions have the highest litigation rates?
How many litigated claims are still open?
What is the average fraud score of litigated claims vs non-litigated?
```

### Resolution & performance
```
Which claim type has the longest average resolution time?
What is the average resolution time for litigated claims vs non-litigated claims?
Which adjuster has the fastest average resolution time?
How many claims were resolved within 30 days?
What is the average settled amount vs claimed amount for closed claims?
```

### Regional & adjuster analysis
```
Compare average claim amounts across all regions
Which region has the most open claims?
Show me claims handled by adjuster ADJ0001
Which adjusters have handled the most high-risk claims?
What is the total claim value for London vs Manchester?
```

### Financial
```
What is the average claim amount by claim type?
What is the total value of all open claims?
Which region has the highest average claim amount?
What percentage of claims were settled for less than the claimed amount?
```

---

## 📄 RAG Agent — Policy Documents

These questions search your 7 real insurance policy PDFs (AXA, Direct Line, Admiral).
Answers include the document name and page number as a citation.

### Claim submission rules
```
What is the time limit for submitting a vehicle insurance claim after an incident?
What documents does a claimant need to provide when making a claim?
What happens if a policyholder misses the claim submission deadline?
Can a claim be submitted more than 30 days after the incident?
What are the policyholder's obligations immediately after an incident?
```

### Coverage & exclusions
```
What incidents are excluded from vehicle insurance coverage?
What does the policy say about claims made while driving under the influence?
Are claims covered if the vehicle was being driven by an unlicensed driver?
What is the maximum payout for a total loss vehicle claim?
Does the policy cover theft of personal belongings from a vehicle?
```

### High-value & complex claims
```
How are high-value property claims handled differently from standard claims?
What happens when a claim exceeds the policy coverage limit?
What is required for a subsidence claim?
How does the policy handle business interruption claims?
What evidence is required for a property damage claim?
```

### Fraud & investigations
```
What fraud indicators trigger a referral to the Special Investigations Unit?
What happens when a claim is suspected to be fraudulent?
What rights does the insurer have to investigate a claim?
Can a policy be cancelled if fraud is suspected?
What is the process for appealing a claim decision?
```

### Litigation & legal
```
What happens when a claim goes to litigation?
What are the legal requirements for third-party claims?
How long does the insurer have to respond to a legal claim?
What does the policy say about legal representation costs?
When does the insurer take over the defence of a claim?
```

### Settlement & payment
```
How long does it take for a claim to be settled after approval?
How is the settlement amount calculated for a total loss?
Can a claimant dispute the settlement offer?
How is payment made — directly to claimant or repairer?
What is the process for appealing a rejected claim?
```

---

## ✦ AI Knowledge — Insurance Terminology

These questions go beyond the PDFs and use Groq's insurance expertise.
Great for understanding terms you encounter while working on a claim.
You'll see the **AI Knowledge** badge (amber) on these answers.

### Core definitions
```
What is subrogation in insurance?
What does excess mean on a policy?
What is indemnity in insurance law?
What is the difference between third party and comprehensive cover?
What is a loss adjuster and what do they do?
What is an insurance premium?
What does 'utmost good faith' mean in insurance?
What is a policyholder vs an insured?
```

### Claims terminology
```
What is a FNOL (First Notification of Loss)?
What does 'without prejudice' mean in claims negotiations?
What is a claims reserve?
What is a subrogation waiver?
What is the difference between a claim and a complaint?
What is a proof of loss in insurance?
What is a claims adjuster vs a loss adjuster?
What does 'settlement in full and final' mean?
```

### Fraud & risk terms
```
What is insurance fraud and how is it detected?
What is ghost broking in motor insurance?
What is a staged accident in insurance fraud?
What does AUC-ROC mean in fraud detection models?
What is the Insurance Fraud Bureau?
What is the Claims and Underwriting Exchange (CUE)?
What is a fraud ring?
```

### Legal & regulatory
```
What is the Financial Ombudsman Service?
What does the FCA regulate in insurance?
What is the Pre-Action Protocol for Personal Injury claims?
What is the Rehabilitation Code in insurance?
What is proportionality in legal costs?
What is the Motor Insurers Bureau (MIB)?
What rights do policyholders have under UK law?
```

### Policy structure
```
What is the difference between an excess and a deductible?
What is a policy schedule?
What is a retroactive date in insurance?
What is co-insurance?
What is reinsurance?
What is an underwriter in insurance?
What is a risk tier in insurance?
```

---

## 🤖 ML Agent — Claim Risk Scoring

These questions score a specific claim using trained XGBoost models.
You need the Claim ID (format: CLM followed by 7 digits).

```
Score claim CLM0000001
Analyse claim CLM0000042
What is the fraud risk for claim CLM0000100?
Give me a full risk assessment for claim CLM0001234
Score claim CLM0005000
What is the litigation probability for claim CLM0002500?
```

**What you get back:**
- Fraud score (0–100%) with flag if above threshold
- Litigation score (0–100%) with risk level
- Resolution time forecast in days
- Overall risk tier: LOW / MEDIUM / HIGH

---

## 🗄️📄 Combined SQL + RAG

These questions need both data analysis AND policy context together.

```
Which regions have the highest litigation rates and what does policy say about litigation procedures?
What claim types have the most fraud and what does AXA say about fraud investigation?
Which adjusters handle the most complex claims and what is the process for complex claims?
What are the busiest months for claims and what are the submission time limits?
```

---

## 💡 Tips for best results

**Be specific about claim IDs** — always use the full format `CLM0000001` (CLM + 7 digits)

**For SQL questions** — include words like "how many", "average", "which", "compare", "top 5", "percentage"

**For policy questions** — include words like "what does policy say", "procedure for", "requirements", "covered", "excluded"

**For definitions** — start with "what is", "explain", "define", "difference between"

**For claim scoring** — just say "score claim" or "analyse claim" followed by the ID

---

## 🚫 What InsuredAI does NOT do

- It cannot search the live internet or news
- It cannot access real policyholder data (uses synthetic data only)
- It cannot modify or update claim records
- It cannot process payments or authorise settlements
- SQL queries are read-only — no data can be changed

---

*InsuredAI — Built by Sumedh Wani · github.com/GitwithIt-tech/claims-intelligence-agent*