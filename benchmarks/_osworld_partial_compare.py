"""Compare DeltaVision-ON vs force-full-frame on overlapping tasks."""
import json
from pathlib import Path

DV = json.load(open('/tmp/dvon.json'))['results']

# Force-full-frame partial: hand-extracted from stdout log (9 tasks ran)
FF = {
    'bb5e4c0d-f964-439c-97b6-bdb9747de3f4': {  # chrome - bing
        'steps': 9, 'success': False, 'score': 0.0, 'tokens': 7600, 'elapsed': 62.4, 'error': None,
    },
    '7b6c7e24-c58a-49fc-a5bb-d57b80e5b4c3': {  # chrome (setup failed)
        'steps': 0, 'success': False, 'score': None, 'tokens': 0, 'elapsed': 21.8, 'error': 'Playwright',
    },
    '35253b65-1c19-4304-8aa4-6884b8218fc0': {
        'steps': 0, 'success': False, 'score': None, 'tokens': 0, 'elapsed': 0.1, 'error': 'Playwright',
    },
    'a96b564e-dbe9-42c3-9ccf-b4498073438a': {
        'steps': 0, 'success': False, 'score': None, 'tokens': 0, 'elapsed': 0.4, 'error': 'Playwright',
    },
    '7a4deb26-d57d-4ea9-9a73-630f66a7b568': {  # gimp
        'steps': 3, 'success': False, 'score': 0.0, 'tokens': 4000, 'elapsed': 37.6, 'error': None,
    },
    '554785e9-4523-4e7a-b8e1-8016f565f56a': {  # gimp - SUCCESS
        'steps': 26, 'success': True, 'score': 1.0, 'tokens': 13200, 'elapsed': 197.4, 'error': None,
    },
    '357ef137-7eeb-4c80-a3bb-0951f26a8aff': {  # libreoffice_calc
        'steps': 6, 'success': False, 'score': 0.0, 'tokens': 4000, 'elapsed': 64.9, 'error': None,
    },
    '42e0a640-4f19-4b28-973d-729602b5a4a7': {  # libreoffice_calc
        'steps': 8, 'success': False, 'score': 0.0, 'tokens': 4800, 'elapsed': 74.8, 'error': None,
    },
    'abed40dc-063f-4598-8ba5-9fe749c0615d': {  # libreoffice_calc
        'steps': 10, 'success': False, 'score': 0.0, 'tokens': 5600, 'elapsed': 89.6, 'error': None,
    },
}

print(f"{'task_id':<12} {'category':<20} {'  DV success/tok/steps':<28} {'  FF success/tok/steps':<28}")
print("-" * 90)
overlap = []
for tid_full, ff in FF.items():
    tid = tid_full[:8]
    dv = next((t for t in DV if t['task_id'] == tid_full), None)
    cat = dv['category'] if dv else 'unknown'
    dv_str = f"{'X' if dv['success'] else '-'}/{dv['estimated_tokens']:>5}/{dv['steps']:>2}" if dv else "missing"
    ff_str = f"{'X' if ff['success'] else '-'}/{ff['tokens']:>5}/{ff['steps']:>2}"
    print(f"{tid:<12} {cat:<20} {dv_str:<28} {ff_str:<28}")
    overlap.append({'tid': tid, 'cat': cat,
                    'dv_success': dv['success'] if dv else None, 'dv_tok': dv['estimated_tokens'] if dv else 0, 'dv_steps': dv['steps'] if dv else 0, 'dv_err': dv['error'] if dv else None,
                    'ff_success': ff['success'], 'ff_tok': ff['tokens'], 'ff_steps': ff['steps'], 'ff_err': ff['error']})

print()
# Filter to tasks where BOTH ran cleanly (no errors)
clean = [o for o in overlap if not o['dv_err'] and not o['ff_err']]
print(f"\n=== {len(clean)} tasks where BOTH configs ran without setup errors ===")
dv_tok = sum(o['dv_tok'] for o in clean)
ff_tok = sum(o['ff_tok'] for o in clean)
dv_succ = sum(1 for o in clean if o['dv_success'])
ff_succ = sum(1 for o in clean if o['ff_success'])
print(f"  total tokens DV={dv_tok:,}  FF={ff_tok:,}  ratio={dv_tok/ff_tok:.2f}x  savings={(ff_tok-dv_tok)/ff_tok*100:.1f}%")
print(f"  successes:  DV={dv_succ}/{len(clean)}  FF={ff_succ}/{len(clean)}")
print(f"  avg steps:  DV={sum(o['dv_steps'] for o in clean)/len(clean):.1f}  FF={sum(o['ff_steps'] for o in clean)/len(clean):.1f}")
