"""
Phase 3 demo: builds a tiny throwaway database where accuracy and format both break,
then prints the remediation plan and the flip set. Learning aid only.

Run from the project folder:  python -m scripts.demo_phase3
"""
import tempfile, os
from datetime import timedelta
from driftguard.storage.store import DriftStore
from driftguard.classification.incidents import group_into_incidents
from driftguard.remediation.mapper import plan_for_incident
from driftguard.remediation.evidence import evidence_for_incident
from tests.fixtures import events as fx
from tests.fixtures.logs import Task, write_eval_history

store = DriftStore(os.path.join(tempfile.mkdtemp(), 'demo.db'))
tasks = [Task('mmlu_a', 'accuracy', 1.0, 0.0), Task('mmlu_b', 'accuracy', 1.0, 1.0),
         Task('fmt_json_1', 'format', 1.0, 0.0)]
change = write_eval_history(store, tasks)
h = (change - fx.BASE_TIME) / timedelta(hours=1)
acc = fx._event('accuracy', hours_after_base=h, pre_mean=1.0, post_mean=0.5, p_value_corrected=0.01)
fmt = fx._event('format_adherence', hours_after_base=h + 0.5, pre_mean=1.0, post_mean=0.0, p_value_corrected=0.01)
incident = group_into_incidents([acc, fmt])[0]
plan = plan_for_incident(incident)
print('HEADLINE:', plan.headline)
print('URGENCY: ', plan.urgency)
for x in plan.possible_explanations:
    print('POSSIBLE:', x)
for i, s in enumerate(plan.steps, 1):
    print(f'{i}.', s)
for metric, e in evidence_for_incident(store, incident).items():
    print()
    print(metric, '->', e.summary)
    for f in e.flips:
        print('   ', f.task_id, ':', f.description)
