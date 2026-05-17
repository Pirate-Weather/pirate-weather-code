import ast
from pathlib import Path


def _find_call_keywords(tree, function_name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == function_name:
                return {
                    keyword.arg: keyword.value
                    for keyword in node.keywords
                    if keyword.arg
                }
    raise AssertionError(f"{function_name} call not found in responseLocal.py")


def _assert_uses_inc_ai_models(tree, function_name):
    keywords = _find_call_keywords(tree, function_name)
    assert "prioritize_ai_models" in keywords

    referenced_names = {
        child.id
        for child in ast.walk(keywords["prioritize_ai_models"])
        if isinstance(child, ast.Name)
    }
    assert "incAIModels" in referenced_names


def test_response_local_passes_ai_models_flag_to_prepare_data_inputs():
    response_local_path = (
        Path(__file__).resolve().parents[1] / "API" / "responseLocal.py"
    )
    tree = ast.parse(response_local_path.read_text())

    _assert_uses_inc_ai_models(tree, "prepare_data_inputs")


def test_response_local_passes_ai_models_flag_to_build_minutely_block():
    response_local_path = (
        Path(__file__).resolve().parents[1] / "API" / "responseLocal.py"
    )
    tree = ast.parse(response_local_path.read_text())

    _assert_uses_inc_ai_models(tree, "build_minutely_block")


def test_response_local_passes_ai_models_flag_to_build_current_section():
    response_local_path = (
        Path(__file__).resolve().parents[1] / "API" / "responseLocal.py"
    )
    tree = ast.parse(response_local_path.read_text())

    _assert_uses_inc_ai_models(tree, "build_current_section")


def test_current_metrics_uses_explicit_ai_models_flag_not_sentinel():
    metrics_path = (
        Path(__file__).resolve().parents[1] / "API" / "current" / "metrics.py"
    )
    source = metrics_path.read_text()

    assert "__ai_models_flag__" not in source
