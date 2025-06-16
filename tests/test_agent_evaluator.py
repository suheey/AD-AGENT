from agents.agent_evaluator import AgentEvaluator

SCRIPT = """print('AUROC: 0.9')\nprint('AUPRC: 0.8')\nprint('Failed prediction at point [1, 2] with true label 0')"""

def test_execute_code(tmp_path):
    evaluator = AgentEvaluator()
    cq = evaluator.execute_code(SCRIPT, 'dummy')
    assert cq.auroc == 0.9
    assert cq.auprc == 0.8
    assert cq.error_points == [{'point': [1.0, 2.0], 'true_label': 0.0}]
    assert cq.error_message == ''
