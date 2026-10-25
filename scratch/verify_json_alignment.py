import json

path = r"C:\Users\USER\.gemini\antigravity-ide\brain\52ec80cc-638d-45a4-bfe1-52de0b18ef36\scratch\auditable_wer_results.json"
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

for mode in ["strict", "normalized"]:
    align_key = f"{mode}_alignments"
    summary_key = f"{mode}_orthographic" if mode == "strict" else "normalized_code_switch_aware"
    
    for beam in ["beam_1", "beam_2"]:
        alignments = data[align_key][beam]
        c = sum(1 for a in alignments if a["op"] == "CORRECT")
        s = sum(1 for a in alignments if a["op"] == "SUBSTITUTION")
        d = sum(1 for a in alignments if a["op"] == "DELETION")
        ins = sum(1 for a in alignments if a["op"] == "INSERTION")
        
        ref_words = [a["ref"] for a in alignments if a["ref"] != "---"]
        hyp_words = [a["hyp"] for a in alignments if a["hyp"] != "---"]
        n = len(ref_words)
        m = len(hyp_words)
        ed = s + d + ins
        wer = round((ed / n) * 100.0, 2) if n > 0 else 0.0
        
        summary = data["summary"][summary_key][beam]
        print(f"=== {mode.upper()} {beam.upper()} ===")
        print(f"Direct Count: C={c}, S={s}, D={d}, I={ins}, N={n}, M={m}, ED={ed}, WER={wer}%")
        print(f"Summary Dict: C={summary['correct_C']}, S={summary['substitutions_S']}, D={summary['deletions_D']}, I={summary['insertions_I']}, N={summary['n_ref_words']}, M={summary['m_hyp_words']}, ED={summary['edit_distance']}, WER={summary['wer_percent']}%")
        
        # Verify alignment integrity
        mismatches = 0
        for idx, a in enumerate(alignments):
            op = a["op"]
            ref_w = a["ref"]
            hyp_w = a["hyp"]
            if op == "CORRECT":
                if ref_w != hyp_w or ref_w == "---":
                    print(f"  [ERROR] Item {idx}: op is CORRECT but ref='{ref_w}' and hyp='{hyp_w}'")
                    mismatches += 1
            elif op == "SUBSTITUTION":
                if ref_w == hyp_w or ref_w == "---" or hyp_w == "---":
                    print(f"  [ERROR] Item {idx}: op is SUBSTITUTION but ref='{ref_w}' and hyp='{hyp_w}'")
                    mismatches += 1
            elif op == "DELETION":
                if ref_w == "---" or hyp_w != "---":
                    print(f"  [ERROR] Item {idx}: op is DELETION but ref='{ref_w}' and hyp='{hyp_w}'")
                    mismatches += 1
            elif op == "INSERTION":
                if ref_w != "---" or hyp_w == "---":
                    print(f"  [ERROR] Item {idx}: op is INSERTION but ref='{ref_w}' and hyp='{hyp_w}'")
                    mismatches += 1
        print(f"Alignment Operation Consistency: {'100% Consistent (0 errors)' if mismatches == 0 else f'{mismatches} errors'}\n")
