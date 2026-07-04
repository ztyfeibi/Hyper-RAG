# Neurology Schema Calibration Notes

## Sampling Scope

- Dataset: `datasets/neurology/neurology.jsonl`
- Dataset size inspected: 12,370 JSONL records.
- Record fields: `id`, `title`, `context`, `contexts`.
- Primary calibration field: `context`.
- Sample size: 80 contexts.
- Sampling method: evenly spaced records across the JSONL file, plus targeted phrase checks for representative neurology concepts.
- Purpose: identify stable entity types, relation types, and high-order relation patterns for Hyper-RAG extraction.

## Dataset Structure Observations

- `context` contains the main source text and is the right field for schema calibration.
- `contexts` repeats the same text with a title prefix, so it is useful for provenance but not needed for entity/relation boundary decisions.
- Text is textbook-like neurology content with clinical examples, disease descriptions, examination methods, anatomy, pathophysiology, treatment sections, and literature-style references.
- Some sampled records are fragments from adjacent textbook chunks. The schema should tolerate partial mentions and should not require every entity to appear in a complete sentence.

## Frequent Entity Candidates

| Candidate | Likely Type | Example Mentions | Notes |
| --- | --- | --- | --- |
| autoimmune encephalitis | DISEASE | `Neurology_Adams_3`, `Neurology_Adams_8508` | Named disease class; often linked to antibodies and tumors. |
| schizophrenia | DISEASE | `Neurology_Adams_1`, `Neurology_Adams_206` | Psychiatric disease appears in differential/diagnostic contexts. |
| generalized seizure | SIGN | `Neurology_Adams_1` | Objective event observed in hospital; can also be a clinical manifestation. |
| cerebral amyloid angiopathy | DISEASE | `Neurology_Adams_4`, `Neurology_Adams_6667` | Named vascular/degenerative condition. |
| posterior reversible encephalopathy syndrome | DISEASE | `Neurology_Adams_4`, `Neurology_Adams_2607` | Named syndrome; keep under DISEASE rather than generic syndrome type. |
| neuromyelitis optica spectrum | DISEASE | `Neurology_Adams_4`, `Neurology_Adams_464` | Named disease spectrum. |
| multiple sclerosis | DISEASE | `Neurology_Adams_4`, `Neurology_Adams_25` | Frequent neurological disease. |
| muscular dystrophy | DISEASE | `Neurology_Adams_4`, `Neurology_Adams_295` | Neuromuscular disease; appears in treatment and mechanism contexts. |
| amyloidosis | DISEASE | `Neurology_Adams_4`, `Neurology_Adams_306` | Disease entity connected to biopsy diagnosis. |
| essential tremor | DISEASE | `Neurology_Adams_604`, `Neurology_Adams_608` | Named condition; tremor alone may be SIGN. |
| tremor | SIGN | `Neurology_Adams_604`, `Neurology_Adams_626` | Objective movement finding; may be disease name when qualified as essential tremor. |
| vertigo | SYMPTOM | `Neurology_Adams_15`, `Neurology_Adams_30` | Patient-reported experience; useful SYMPTOM example. |
| pain | SYMPTOM | `Neurology_Adams_939`, `Neurology_Adams_1294` | Subjective symptom. |
| weakness | SYMPTOM | sampled contexts around gait/falls and neuromuscular disease | Usually patient complaint unless measured as objective strength deficit. |
| nystagmus | SIGN | `Neurology_Adams_653`, `Neurology_Adams_655` | Objective eye movement finding. |
| Babinski sign | SIGN | sampled sleep/motor contexts | Named exam sign; should not be SYMPTOM. |
| spinal fluid analysis | EXAMINATION | `Neurology_Adams_1`, `Neurology_Adams_3531` | Diagnostic test/process. |
| ultrasound examination | EXAMINATION | `Neurology_Adams_1` | Imaging/examination used after antibody finding. |
| MRI | EXAMINATION | `Neurology_Adams_70`, `Neurology_Adams_89` | Imaging modality. |
| EEG | EXAMINATION | `Neurology_Adams_180`, `Neurology_Adams_198` | Diagnostic test, often used in seizure/stroke differential contexts. |
| lumbar puncture | EXAMINATION | `Neurology_Adams_86`, `Neurology_Adams_87` | Procedure used to obtain CSF and pressure measurements. |
| anti-NMDA receptor antibody | PROTEIN | `Neurology_Adams_1`, `Neurology_Adams_2604` | Molecular/antibody entity; important for autoimmune encephalitis. |
| beta-amyloid | PROTEIN | `Neurology_Adams_116` | Molecular product; keep PROTEIN available. |
| tau protein | PROTEIN | `Neurology_Adams_116`, `Neurology_Adams_117` | Protein biomarker; distinct from gene names. |
| cerebral cortex | ANATOMICAL_STRUCTURE | `Neurology_Adams_68`, `Neurology_Adams_169` | Brain structure. |
| basal ganglia | ANATOMICAL_STRUCTURE | `Neurology_Adams_340`, `Neurology_Adams_343` | Structure involved in motor control. |
| spinal cord | ANATOMICAL_STRUCTURE | `Neurology_Adams_7`, `Neurology_Adams_58` | Nervous system structure. |
| left ovary | ANATOMICAL_STRUCTURE | `Neurology_Adams_1` | Non-neurologic anatomical site can matter in paraneoplastic disease. |
| ovarian teratoma | DISEASE | `Neurology_Adams_1`, `Neurology_Adams_2` | Tumor/disease entity connected to autoimmune encephalitis. |
| demyelination | PATHOLOGICAL_MECHANISM | `Neurology_Adams_71`, `Neurology_Adams_138` | Disease process; not a disease name by itself. |
| ischemia | PATHOLOGICAL_MECHANISM | `Neurology_Adams_165`, `Neurology_Adams_172` | Pathological process connected to stroke and cord injury. |
| inflammation | PATHOLOGICAL_MECHANISM | `Neurology_Adams_71`, `Neurology_Adams_1566` | Mechanism/process entity. |
| motor control | PHYSIOLOGICAL_FUNCTION | `Neurology_Adams_340`, `Neurology_Adams_343` | Normal function involving basal ganglia/cerebellum. |
| memory | PHYSIOLOGICAL_FUNCTION | sampled dementia contexts | Normal cognitive function, often affected by disease. |
| antipsychotic medications | DRUG | `Neurology_Adams_1` | Drug class; keep distinct from non-drug treatment. |
| clonazepam | DRUG | `Neurology_Adams_595`, `Neurology_Adams_597` | Specific medication. |
| corticosteroid | DRUG | `Neurology_Adams_91`, `Neurology_Adams_606` | Medication class used therapeutically or as exposure. |
| resection | TREATMENT | `Neurology_Adams_1`, `Neurology_Adams_2` | Procedure/treatment, not a drug. |
| craniotomy | TREATMENT | `Neurology_Adams_1483`, `Neurology_Adams_4901` | Surgical intervention. |
| radiation | TREATMENT | `Neurology_Adams_1954` | Can also be imaging physics; classify by local context. |
| chemotherapy | TREATMENT | `Neurology_Adams_1954` | Treatment modality; specific agents would be DRUG. |
| hypertension | RISK_FACTOR | `Neurology_Adams_1294`, `Neurology_Adams_1308` | Often appears as risk/exposure factor or disease depending context. |
| family history | RISK_FACTOR | `Neurology_Adams_709`, `Neurology_Adams_1330` | Risk/predisposition concept. |
| diagnostic criteria | DIAGNOSTIC_CRITERION | sampled diagnostic sections | Keep separate from tests such as MRI or EEG. |

## Frequent Relation Candidates

| Relation Expression | Likely Relation Type | Example Entity Types | Notes |
| --- | --- | --- | --- |
| X causes Y | CAUSES | PATHOLOGICAL_MECHANISM -> DISEASE; DISEASE -> SIGN | Seen in phrases such as syndromes caused by lesions or vascular occlusion. |
| X is associated with Y | ASSOCIATED_WITH | PROTEIN -> DISEASE; DISEASE -> DISEASE | Use when the text states association but not direct causation. |
| X suggests Y | INDICATES | SIGN -> DISEASE; EXAMINATION -> DISEASE | Example: findings suggesting a diagnosis or disease class. |
| X diagnoses Y | DIAGNOSES | EXAMINATION -> DISEASE | Examples include EEG in differential diagnosis, biopsy in amyloidosis, CSF analysis in inflammatory disease. |
| X treats Y | TREATS | DRUG -> DISEASE; TREATMENT -> DISEASE | Examples include corticosteroids, clonazepam, resection, craniotomy, radiation, chemotherapy. |
| X is located in Y | LOCATED_IN | PATHOLOGICAL_MECHANISM -> ANATOMICAL_STRUCTURE; DISEASE -> ANATOMICAL_STRUCTURE | Useful for cortex, basal ganglia, spinal cord, ovary, brainstem, orbit, and ventricular locations. |
| X affects Y | AFFECTS | DISEASE -> PHYSIOLOGICAL_FUNCTION | Common for diseases affecting movement, cognition, consciousness, vision, or gait. |
| X is mechanism of Y | MECHANISM_OF | PATHOLOGICAL_MECHANISM -> DISEASE | Demyelination, ischemia, inflammation, and neurodegeneration explain disease processes. |
| X is risk factor for Y | RISK_FACTOR_FOR | RISK_FACTOR -> DISEASE | Hypertension and family history appear in risk contexts. |
| X is differential diagnosis of Y | DIFFERENTIAL_DIAGNOSIS | DISEASE -> DISEASE | Frequent in textbook differential diagnosis sections. |
| X complicates Y | COMPLICATES | DISEASE -> DISEASE; TREATMENT -> DISEASE | Use for stated complications, not mere co-mentions. |
| X co-occurs with Y | CO_OCCURS_WITH | SYMPTOM -> SIGN; DISEASE -> DISEASE | Weak fallback for clinical constellations without explicit stronger relation. |

## Confusing Cases

| Case | Options | Decision |
| --- | --- | --- |
| symptom vs sign | SYMPTOM / SIGN | Use SYMPTOM for subjective patient experiences such as vertigo, pain, dizziness, headache. Use SIGN for objective findings such as tremor, seizure observed in hospital, nystagmus, Babinski sign. |
| disease vs syndrome | DISEASE / OTHER | Use DISEASE for named syndromes and disease spectra, such as posterior reversible encephalopathy syndrome and neuromyelitis optica spectrum. Do not add a separate SYNDROME type in v1. |
| disease vs pathological mechanism | DISEASE / PATHOLOGICAL_MECHANISM | Use DISEASE for named conditions such as multiple sclerosis. Use PATHOLOGICAL_MECHANISM for processes such as demyelination, ischemia, inflammation, and neurodegeneration. |
| drug vs treatment | DRUG / TREATMENT | Use DRUG for medications or chemical agents such as clonazepam and corticosteroids. Use TREATMENT for procedures and strategies such as resection, craniotomy, radiation, chemotherapy, and rehabilitation. |
| examination vs diagnostic criterion | EXAMINATION / DIAGNOSTIC_CRITERION | Use EXAMINATION for tests/procedures such as MRI, EEG, lumbar puncture, spinal fluid analysis, ultrasound, and biopsy. Use DIAGNOSTIC_CRITERION for named criteria, thresholds, or diagnostic rules. |
| protein vs gene | PROTEIN / GENE | Use PROTEIN for tau, beta-amyloid, receptor antibodies, and molecular products. Use GENE only for explicit gene/locus names. Keep GENE even if less frequent because genetics appears in the dataset. |
| anatomy outside nervous system | ANATOMICAL_STRUCTURE / OTHER | Keep as ANATOMICAL_STRUCTURE when medically relevant to the neurology relation, such as ovary in anti-NMDA receptor encephalitis. |
| radiation | EXAMINATION / TREATMENT | Use EXAMINATION when the text discusses imaging physics/exposure. Use TREATMENT when it appears as therapy for tumors or lesions. |
| hypertension | DISEASE / RISK_FACTOR | Use RISK_FACTOR when described as increasing likelihood or associated with another disorder. Use DISEASE when hypertension itself is the diagnosed condition. |
| weak co-mentions in references | ASSOCIATED_WITH / CO_OCCURS_WITH / OTHER | Literature reference lists often place terms together without a meaningful relation. Prefer CO_OCCURS_WITH or OTHER instead of inventing causal relations. |

## Schema Adjustment Decisions

| Decision | Reason |
| --- | --- |
| Keep SYMPTOM and SIGN separate | Neurology reasoning depends on whether a finding is patient-reported or clinician-observed. Vertigo and pain behave differently from nystagmus, Babinski sign, tremor, and seizure. |
| Keep DRUG and TREATMENT separate | The dataset includes both medications and interventions/procedures. Merging them would blur drug-treatment and procedure-treatment relations. |
| Keep EXAMINATION separate from DIAGNOSTIC_CRITERION | MRI, EEG, lumbar puncture, CSF analysis, ultrasound, and biopsy are tests/procedures; criteria and thresholds are diagnostic rules. |
| Keep PATHOLOGICAL_MECHANISM | Demyelination, ischemia, inflammation, neurodegeneration, and tumor-related processes are central to mechanistic questions. |
| Keep ANATOMICAL_STRUCTURE broad enough for non-neurologic structures | Some neurology relations involve systemic or paraneoplastic sites, such as ovary and teratoma in anti-NMDA receptor encephalitis. |
| Keep GENE, PROTEIN, and PATHWAY in v1 | Molecular biology and genetics appear in the textbook, and later type-aware extraction should not force proteins or pathways into OTHER. |
| Do not add separate SYNDROME type in v1 | Named syndromes can be covered by DISEASE. This keeps the first schema compact. |
| Use OTHER as fallback | The textbook includes references, authors, study names, general concepts, and partial chunk fragments that should not be forced into medical labels. |
| Treat allowed patterns as soft validation | Neurology text contains legitimate unusual relations. Phase 1 should mark validity, not hard-delete edges. |

## Expected Validation Thresholds

| Metric | Target | Rationale |
| --- | --- | --- |
| `OTHER` entity ratio | <= 0.20 | A higher ratio means the entity schema is missing common dataset concepts or the prompt is too broad. |
| `OTHER` relation ratio | <= 0.20 | Some fallback is expected for textbook prose and references, but it should not dominate. |
| `invalid` pattern ratio | <= 0.10 | Most extracted typed relations should either match allowed patterns or be weakly valid. |
| `weak_valid` pattern ratio | <= 0.30 | Weak-valid edges are acceptable during discovery, but a high ratio means allowed patterns need adjustment. |

## Calibration Summary

The sampled Neurology dataset supports a compact clinical-neurology schema rather than a full medical ontology. The first schema version should emphasize diseases, subjective symptoms, objective signs, drugs, interventions, examinations, anatomy, functions, pathological mechanisms, molecular entities, risk factors, and diagnostic criteria.

The strongest relation labels should cover treatment, diagnosis, indication, causation, association, anatomical location, mechanism, risk, differential diagnosis, and weak co-occurrence. High-order hyperedges should be reserved for meaningful clinical patterns such as disease + symptom/sign constellations, disease + examination + diagnostic criterion patterns, disease + drug/treatment strategies, and disease + mechanism + anatomy explanations.
