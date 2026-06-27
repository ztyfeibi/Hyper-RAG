# Neurology Schema

## Purpose

`neurology_schema_v1` is a lightweight domain schema for Hyper-RAG entity and hyperedge extraction on the Neurology dataset.

The schema is intended to guide later type-aware extraction, relation normalization, schema validity marking, vector metadata enrichment, and query-time weighting. It is not a complete medical ontology, a UMLS replacement, or a strict clinical decision system.

Phase 1 only defines the schema artifacts. Runtime extraction, query, storage, and cache behavior must remain unchanged in this phase.

## Dataset Basis

- Dataset: `datasets/neurology/neurology.jsonl`
- Calibration notes: `docs/schema/neurology_schema_calibration.md`
- Primary text field: `context`
- Schema scope: textbook-like neurology content, including diseases, symptoms, signs, examinations, treatments, anatomy, mechanisms, molecular entities, risk factors, and diagnostic criteria.

## Entity Types

| Type | Meaning | Examples | Boundary Notes |
| --- | --- | --- | --- |
| DISEASE | A named neurological or medical disease/disorder. | Alzheimer disease, stroke, multiple sclerosis | Use DISEASE for diagnosed conditions, named syndromes, disease spectra, and tumors, not single symptoms. |
| SYMPTOM | Subjective patient-reported abnormal experience. | headache, dizziness, memory loss | Use SIGN for objective clinician-observed findings. |
| SIGN | Objective clinical finding observed or measured by clinicians. | tremor, Babinski sign, nystagmus | Use SYMPTOM when the text describes patient experience. |
| DRUG | A medication or chemical therapeutic agent. | levodopa, aspirin, dopamine agonist | Use TREATMENT for non-drug interventions. |
| TREATMENT | A therapeutic intervention, procedure, or management strategy. | surgery, rehabilitation, deep brain stimulation | Use DRUG for specific medications and chemical therapeutic agents. |
| EXAMINATION | A diagnostic test, imaging, lab, or clinical examination. | MRI, EEG, lumbar puncture | Use DIAGNOSTIC_CRITERION for named diagnostic criteria, thresholds, or rules. |
| ANATOMICAL_STRUCTURE | A body, nervous system, brain, nerve, or tissue structure. | basal ganglia, cortex, spinal cord | Use PHYSIOLOGICAL_FUNCTION for functions, not structures. Non-neurologic structures may be included when relevant to a neurology relation. |
| PHYSIOLOGICAL_FUNCTION | Normal biological or neurological function. | memory, motor control, consciousness | Use PATHOLOGICAL_MECHANISM for abnormal disease processes. |
| PATHOLOGICAL_MECHANISM | Disease mechanism or abnormal biological process. | demyelination, ischemia, neurodegeneration | Use CAUSES or MECHANISM_OF relations to connect it to disease. |
| GENE | A gene or genetic locus. | APOE, HTT | If rare in sampled data, keep but avoid over-emphasizing in analysis. |
| PROTEIN | A protein or molecular product. | tau, amyloid beta | Use GENE for gene names. Use PROTEIN for antibodies and molecular products when the text treats them as biological molecules. |
| PATHWAY | Biological, molecular, or signaling pathway. | dopamine pathway, inflammatory pathway | Use PATHOLOGICAL_MECHANISM if text describes a disease process rather than a named pathway. |
| RISK_FACTOR | A factor that increases disease likelihood or severity. | age, hypertension, family history | Use CAUSES only when text states direct causation. |
| DIAGNOSTIC_CRITERION | A named criterion, threshold, or diagnostic rule. | diagnostic criteria, clinical criteria | Use EXAMINATION for tests and procedures. |
| OTHER | Fallback type when no schema type fits. | author name, study label, vague concept | Should be low-frequency. High OTHER ratio means schema is too narrow or extraction is too broad. |

## Entity Boundary Guidance

- Named syndromes and disease spectra should normally be DISEASE in v1.
- Single findings such as tremor, nystagmus, Babinski sign, or observed seizure should normally be SIGN.
- Subjective experiences such as pain, vertigo, headache, or dizziness should normally be SYMPTOM.
- Procedures such as resection, craniotomy, radiation therapy, chemotherapy, rehabilitation, and stimulation should normally be TREATMENT.
- Tests and diagnostic procedures such as MRI, CT, EEG, lumbar puncture, spinal fluid analysis, ultrasound, and biopsy should normally be EXAMINATION.
- Molecular products such as tau, amyloid beta, and receptor antibodies should normally be PROTEIN unless the text explicitly names a gene or locus.
- Pathological processes such as demyelination, ischemia, inflammation, and neurodegeneration should normally be PATHOLOGICAL_MECHANISM.
- Use OTHER for authors, references, study titles, generic textbook concepts, and fragmentary mentions that do not fit a stable medical role.

## Relation Types

| Type | Meaning | Boundary Notes |
| --- | --- | --- |
| CAUSES | One entity directly causes another. | Use ASSOCIATED_WITH if causality is not explicit. |
| ASSOCIATED_WITH | General association without clear direction or causality. | Use this when the text is weaker than CAUSES. |
| INDICATES | One finding suggests or points to another. | Often symptom/sign/exam -> disease. |
| DIAGNOSES | One test, criterion, or process is used to diagnose another. | Stronger and more procedural than INDICATES. |
| TREATS | One drug or treatment improves or manages another entity. | Use PREVENTS for prevention. |
| PREVENTS | One intervention reduces likelihood of another. | Do not use for treatment after disease onset. |
| COMPLICATES | One disease/event worsens, complicates, or results as complication of another. | Use CO_OCCURS_WITH when no complication is stated. |
| LOCATED_IN | One entity is anatomically located in another. | Usually disease/process -> structure. |
| AFFECTS | One entity changes or impacts another. | Broader than REGULATES. |
| REGULATES | One entity modulates or controls another biological process. | Usually molecular/pathway/function relation. |
| PART_OF | One entity is a component of another. | Use LOCATED_IN for spatial location. |
| INTERACTS_WITH | Two drugs, proteins, pathways, or entities interact. | Use REGULATES for directional control. |
| MECHANISM_OF | One mechanism explains another entity. | Usually mechanism/pathway/function -> disease. |
| RISK_FACTOR_FOR | One entity increases risk of another. | Use CAUSES only for direct causality. |
| DIFFERENTIAL_DIAGNOSIS | Diseases or syndromes that must be distinguished. | Usually disease -> disease. |
| CO_OCCURS_WITH | Entities appear together without a stronger relation. | Weaker than COMPLICATES or ASSOCIATED_WITH. |
| OTHER | Fallback relation type. | High OTHER ratio means relation schema is too narrow. |

## Relation Boundary Guidance

- Use CAUSES only when the text states a causal relation directly.
- Use ASSOCIATED_WITH for correlation, linkage, paraneoplastic association, or relatedness without clear causality.
- Use INDICATES when a symptom, sign, examination, or marker points toward a disease but is not itself a diagnostic procedure.
- Use DIAGNOSES when a test, criterion, or diagnostic workflow is used to establish or confirm a diagnosis.
- Use TREATS for drugs, procedures, or management strategies that improve or manage a disease, symptom, or sign.
- Use PREVENTS only for preventive effect before onset or recurrence, not ordinary treatment after disease onset.
- Use LOCATED_IN for anatomical site relations and PART_OF for component/whole relations.
- Use CO_OCCURS_WITH for clinical constellations or reference-list co-mentions that lack a stronger stated relation.
