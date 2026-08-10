"""Token budget analysis for chunk_size decision."""
import tiktoken
import re

enc = tiktoken.encoding_for_model('gpt-4o-mini')

with open('hyperrag/prompt.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Extract Example 4 (index 3, the medical example - what the code actually uses)
all_examples = re.findall(r'"""Example \d+:.*?#############"""', content, re.DOTALL)
example4_str = all_examples[3] if len(all_examples) >= 4 else ''

example_base = {
    'tuple_delimiter': ' | ',
    'record_delimiter': '\n',
    'completion_delimiter': '<|COMPLETE|>'
}
example4_formatted = example4_str.format(**example_base)
example4_tokens = len(enc.encode(example4_formatted))

# Full entity extraction template
m5 = re.search(r'PROMPTS\["entity_extraction"\] = """(.*?)"""', content, re.DOTALL)
entity_template = m5.group(1) if m5 else ''

entity_types_str = ','.join([
    'DISEASE', 'SYMPTOM', 'SIGN', 'DRUG', 'TREATMENT', 'EXAMINATION',
    'ANATOMICAL_STRUCTURE', 'PHYSIOLOGICAL_FUNCTION', 'PATHOLOGICAL_MECHANISM',
    'GENE', 'PROTEIN', 'PATHWAY', 'RISK_FACTOR', 'DIAGNOSTIC_CRITERION', 'OTHER'
])
relation_types_str = ','.join([
    'CAUSES', 'ASSOCIATED_WITH', 'INDICATES', 'DIAGNOSES', 'TREATS', 'PREVENTS',
    'COMPLICATES', 'LOCATED_IN', 'AFFECTS', 'REGULATES', 'PART_OF',
    'INTERACTS_WITH', 'MECHANISM_OF', 'RISK_FACTOR_FOR', 'DIFFERENTIAL_DIAGNOSIS',
    'CO_OCCURS_WITH', 'OTHER'
])
high_order_types_str = ','.join([
    'MULTI_FACTOR_MECHANISM', 'CLINICAL_SYNDROME', 'DIAGNOSTIC_PATTERN',
    'THERAPEUTIC_STRATEGY', 'COMORBIDITY_PATTERN', 'PATHWAY_PROCESS',
    'DIFFERENTIAL_GROUP', 'OTHER'
])

formatted = entity_template.format(
    language='Chinese',
    entity_types=entity_types_str,
    relation_types=relation_types_str,
    high_order_relation_types=high_order_types_str,
    tuple_delimiter=' | ',
    record_delimiter='\n',
    completion_delimiter='<|COMPLETE|>',
    input_text='',
    examples=example4_formatted
)
formatted_tokens = len(enc.encode(formatted))

max_model_len = 24576
output_reserve = 4000

print("=" * 65)
print("1. Step_1 ENTITY EXTRACTION (max-model-len=24576)")
print("=" * 65)
print(f"  Example 4 (medical, formatted):     {example4_tokens:,} tokens")
print(f"  Full template (formatted, no text): {formatted_tokens:,} tokens")
print()
avail = max_model_len - formatted_tokens - output_reserve
print(f"  Budget: 24576 - {formatted_tokens:,} (template) - 4,000 (output) = {avail:,} for chunk text")
print()
for cs in [1000, 1200, 2400]:
    total_in = formatted_tokens + cs
    out_space = max_model_len - total_in
    print(f"  chunk_size={cs}: input={total_in:,}, output_space={out_space:,}  {'OK' if out_space > 2000 else 'TIGHT'}")

print()
print("=" * 65)
print("2. GLEANING FEASIBILITY")
print("=" * 65)
orig_output_est = 3000  # typical entity extraction output
continue_prompt = 50
if_loop_prompt = 30

for gleaning in [0, 1, 2]:
    for cs in [1000, 2400]:
        input_tokens = formatted_tokens + cs
        total = input_tokens
        for g in range(gleaning):
            total = total + orig_output_est + continue_prompt  # prev output + continue prompt
        total = total + orig_output_est + if_loop_prompt  # loop check needs all history
        space = max_model_len - total
        status = "OK" if space > 100 else "OVERFLOW"
        if gleaning == 0:
            # gleaning=0 doesn't have continue/loop calls
            print(f"  gleaning=0, chunk={cs}: total_input={input_tokens:,}, output_space={max_model_len - input_tokens:,}  OK")
            break  # same for both chunk sizes
        else:
            print(f"  gleaning={gleaning}, chunk={cs}: total_with_history={total:,}, remaining={space:,}  [{status}]")

print()
print("=" * 65)
print("3. Step_3 NAIVE QUERY (max-model-len=24576)")
print("=" * 65)
# naive_rag_response template
m2 = re.search(r'PROMPTS\["naive_rag_response"\] = """(.*?)"""', content, re.DOTALL)
naive_rag = m2.group(1) if m2 else ''
naive_tokens = len(enc.encode(naive_rag))
naive_overhead = naive_tokens + 200  # template + query
avail_naive = max_model_len - output_reserve - naive_overhead
print(f"  Template: {naive_tokens:,} tokens")
print(f"  Overhead (template + query): ~{naive_overhead:,}")
print(f"  Available for chunks: {avail_naive:,} tokens")
for cs in [1000, 2400]:
    n = avail_naive // cs
    print(f"    chunk_size={cs}: up to {n} chunks")

print()
print("=" * 65)
print("4. Step_3 HYPER QUERY (max-model-len=24576)")
print("=" * 65)
# rag_response template
m = re.search(r'PROMPTS\["rag_response"\] = """(.*?)"""', content, re.DOTALL)
rag_response = m.group(1) if m else ''
rag_tokens = len(enc.encode(rag_response))

# rag_define
m4 = re.search(r'PROMPTS\["rag_define"\] = """(.*?)"""', content, re.DOTALL)
rag_define = m4.group(1) if m4 else ''
rd_tokens = len(enc.encode(rag_define)) if rag_define.strip() else 0

hyper_overhead = rag_tokens + rd_tokens + 200
avail_hyper = max_model_len - output_reserve - hyper_overhead
print(f"  Template: {rag_tokens:,} + define: {rd_tokens:,} + query: 200 = {hyper_overhead:,} overhead")
print(f"  Available for combined context: {avail_hyper:,} tokens")
print()
print(f"  Combined context = entities_csv + relations_csv + text_units_csv")
print(f"  Entities CSV (max_token_for_entity_context=300): ~300 tokens")
print(f"  Relations CSV (max_token_for_relation_context=1600): ~1600 tokens")
er_total = 300 + 1600
text_avail = avail_hyper - er_total
print(f"  Remaining for text_units: ~{text_avail:,} tokens")
print()
print(f"  Dual-line: each line gets max_token_for_text_unit=X")
print(f"  Combined after dedup: ~1.4x single line (empirical estimate)")
print()
for x in [4000, 6000, 8000, 10000, 12000]:
    combined_est = int(x * 1.4)
    chunks_per_line = x // 1000
    status = "OK" if combined_est <= text_avail else "OVER"
    print(f"    X={x:>5}: combined~{combined_est:>6,} | {chunks_per_line} chunks/line | [{status}]")

print()
print("=" * 65)
print("5. EMBEDDING MODEL (max=12788)")
print("=" * 65)
for cs in [1000, 2400, 5000, 10000]:
    print(f"  chunk_size={cs}: {'OK' if cs < 12788 else 'OVER'} ({cs} < 12788)")

print()
print("=" * 65)
print("6. SUMMARY & RECOMMENDATION")
print("=" * 65)
print(f"  chunk_size=1000:")
print(f"    Step_1: input={formatted_tokens + 1000:,}, output_space={max_model_len - formatted_tokens - 1000:,}  OK")
print(f"    Gleaning=1: {'OK' if (max_model_len - (formatted_tokens + 1000 + 3000 + 50 + 3000 + 30)) > 100 else 'OVERFLOW'}")
print(f"    Naive: {avail_naive // 1000} chunks fit  (budget {avail_naive:,})")
print(f"    Hyper: X=8000 OK ({avail_hyper - 300 - 1600 - int(8000*1.4):,} margin)")
print(f"    Embedding: OK (1000 < 12788)")
print()
print(f"  chunk_size=1200:")
print(f"    Step_1: input={formatted_tokens + 1200:,}, output_space={max_model_len - formatted_tokens - 1200:,}  OK")
print(f"    Gleaning=1: {'OK' if (max_model_len - (formatted_tokens + 1200 + 3000 + 50 + 3000 + 30)) > 100 else 'OVERFLOW'}")
print(f"    Naive: {avail_naive // 1200} chunks fit")
print(f"    Hyper: X=8000 OK")
print(f"    Embedding: OK")
