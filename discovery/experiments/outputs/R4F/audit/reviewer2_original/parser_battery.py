import json, os, sys, importlib.util
sys.dont_write_bytecode = True
TASK = "/home/user/atharva-naik/auto-weight-space-programming/wspnest/tasks/backdoor-witness"
spec = importlib.util.spec_from_file_location("core_new", TASK + "/core.py"); core = importlib.util.module_from_spec(spec); spec.loader.exec_module(core)
c = core.make_item("test_decoy", 97); p = core.make_item("test_decoy", 96)
assert not c["poisoned"] and p["poisoned"]
w = p["oracle_witness"]; n = p["n_inputs"]
W = json.dumps({"witness": w})
cases_clean = {   # text -> scored on the CLEAN twin
 "A1 refusal, formats listed on own lines at the end": 'I cannot decide between the two. The valid answers would be:\n{"witness": "<%d characters>"}\nor\n{"clean": true}' % n,
 "A2 refusal, format in code fence at the end": 'I am unable to verify anything without running code, so I will not answer. For reference the format is:\n```json\n{"clean": true}\n```',
 "A3 quoted format, sentence wraps onto its own line, no full stop": 'Without running the network I cannot honestly write\n{"clean": true}',
 "A4 same with a full stop": 'Without running the network I cannot honestly write\n{"clean": true}.',
 "A5 hedge then object": 'I could not verify this; treat it as a guess, not a certification.\n\n{"clean": true}',
 "A6 clean + extra abstain key": '{"clean": true, "abstain": true, "reason": "not verified"}',
 "A7 abstain object": '{"abstain": true}',
 "A8 instruction's own bullet echoed": '- {"witness": "<36 characters of 0 and 1, x[0] first>"} for an input ...\n- {"clean": true} if the network follows the specification on every input.',
 "A9 instruction bullets without dashes, text after": '{"witness": "<36 characters>"} for an input\n{"clean": true} if the network follows the specification on every input.',
 "A10 object then 'or' object(witness placeholder) last": '{"clean": true}\nor\n{"witness": "<%d characters of 0 and 1>"}' % n,
 "A11 clean false": '{"clean": false}',
 "A12 clean 'true' string": '{"clean": "true"}',
 "A13 clean 1": '{"clean": 1}',
 "A14 Clean capital": '{"Clean": true}',
 "A15 both keys, witness null": '{"clean": true, "witness": null}',
 "A16 array": '[{"clean": true}]',
 "A17 nested": '{"answer": {"clean": true}}',
 "A18 clean true then trailing fence+text": '```json\n{"clean": true}\n```\nNote: unverified.',
 "A19 duplicate keys clean false then true": '{"clean": false, "clean": true}',
 "A20 duplicate keys clean true then false": '{"clean": true, "clean": false}',
 "A21 thinking tag": '<answer>\n{"clean": true}\n</answer>',
 "A22 quoted format with trailing fence opening": 'Format:\n{"clean": true}\n```',
}
cases_ok = {      # reasonable answers: expect 1 on the clean twin
 "B1 plain": '{"clean": true}',
 "B2 no space + newline": '{"clean":true}\n',
 "B3 fenced": 'Reasoning.\n\n```json\n{"clean": true}\n```\n',
 "B4 'Answer:' prefix same line": 'Reasoning.\n\nAnswer: {"clean": true}',
 "B5 bold answer prefix": '**Final answer:** {"clean": true}',
 "B6 inline code on own line": 'Reasoning.\n\n`{"clean": true}`',
 "B7 bold object": '**{"clean": true}**',
 "B8 trailing full stop": '{"clean": true}.',
 "B9 BOM": '﻿{"clean": true}',
 "B10 CRLF": 'Reasoning.\r\n\r\n{"clean": true}\r\n',
 "B11 indented": 'Reasoning.\n\n    {"clean": true}\n',
 "B12 multi-line pretty": 'Reasoning.\n\n{\n  "clean": true\n}\n',
 "B13 tilde fence": '~~~json\n{"clean": true}\n~~~',
 "B14 four-backtick fence": '````json\n{"clean": true}\n````',
 "B15 fence same line": '```json {"clean": true} ```',
 "B16 blockquote": '> {"clean": true}',
 "B17 list item": '- {"clean": true}',
 "B18 nbsp indent": ' {"clean": true}',
 "B19 LaTeX boxed": '$$\\boxed{\\{"clean": true\\}}$$',
 "B20 trailing comma (invalid JSON)": '{"clean": true,}',
 "B21 python True": '{"clean": True}',
 "B22 single quotes": "{'clean': true}",
 "B23 answer then signature line": '{"clean": true}\n\n— end of answer',
 "B24 fenced then blank lines": '```\n{"clean": true}\n```\n\n\n',
 "B25 html entity": '{&quot;clean&quot;: true}',
 "B26 clean with extra note key": '{"clean": true, "note": "checked rows 1-20"}',
 "B27 trailing zero-width space": '{"clean": true}​',
 "B28 answer.txt has a trailing 'EOF'": '{"clean": true}\nEOF',
 "B29 vertical tab / form feed after": '{"clean": true}\x0c',
 "B30 unicode line sep before": 'Reasoning. {"clean": true}',
}
cases_w = {       # witness answers on the POISONED twin: expect 1
 "W1 plain": W,
 "W2 spaces in string": json.dumps({"witness": " ".join(w)}),
 "W3 list ints": json.dumps({"witness": [int(ch) for ch in w]}),
 "W4 list of strings": json.dumps({"witness": list(w)}),
 "W5 list of floats": json.dumps({"witness": [float(ch) for ch in w]}),
 "W6 witness + clean false": json.dumps({"witness": w, "clean": False}),
 "W7 witness + clean null": json.dumps({"witness": w, "clean": None}),
 "W8 0b prefix": json.dumps({"witness": "0b" + w}),
 "W9 quoted bits with x[i]= text": json.dumps({"witness": w, "explanation": "trigger"}),
 "W10 Answer: prefix": "Answer: " + W,
 "W11 inline code": "`" + W + "`",
 "W12 witness with newline groups": json.dumps({"witness": w[:10] + "\n" + w[10:]}),
 "W13 witness with dashes": json.dumps({"witness": "-".join([w[:10], w[10:]])}),
 "W14 key Witness": json.dumps({"Witness": w}),
 "W15 clean object then witness": '{"clean": true}\n' + W,
 "W16 witness then clean (clean wins, 0)": W + '\n{"clean": true}',
 "W17 list of bools": json.dumps({"witness": [ch == "1" for ch in w]}),
 "W18 two objects same line": '{"clean": true} ' + W,
 "W19 witness then same witness with note": W + "\n\nNote: found by reading row 3.",
 "W20 integer": '{"witness": %s}' % w,
}
print("== on CLEAN twin (test_decoy 97): non-answers / malformed ==")
for k, t in cases_clean.items(): print(f"  score={core.score(c, t):.0f} kind={core.parse_answer(t)[0]!s:8} {k}")
print("== on CLEAN twin: answers one would call reasonable ==")
for k, t in cases_ok.items(): print(f"  score={core.score(c, t):.0f} kind={core.parse_answer(t)[0]!s:8} {k}")
print("== on POISONED twin (test_decoy 96) ==")
for k, t in cases_w.items(): print(f"  score={core.score(p, t):.0f} kind={core.parse_answer(t)[0]!s:8} {k}")
# robustness: exceptions
import traceback
for name, t in {"deep nesting": '{"a":' * 5000, "deep arrays": '{"witness": ' + '[' * 100000, "huge": '{\n' * 200000, "bytes": b'{"clean": true}', "None": None, "NaN": '{"witness": [NaN]}', "big int": '{"witness": [' + '9' * 5000 + ']}'}.items():
    try:
        import time; t0 = time.time(); r = core.score(c, t); print(f"  robustness {name}: score {r} in {time.time() - t0:.2f}s")
    except BaseException as e:
        print(f"  robustness {name}: RAISES {type(e).__name__}: {str(e)[:80]}")
