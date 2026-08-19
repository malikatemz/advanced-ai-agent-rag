import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tools.definitions import calculator  # noqa: E402


def test_calculator_basic_arithmetic():
    assert calculator("2 + 2") == "4"
    assert calculator("(10 - 4) * 3") == "18"
    assert calculator("2 ** 10") == "1024"


def test_calculator_division():
    assert calculator("10 / 4") == "2.5"


def test_calculator_rejects_unsafe_input():
    result = calculator("__import__('os').system('echo pwned')")
    assert "Error" in result


def test_calculator_rejects_names():
    result = calculator("os.system('ls')")
    assert "Error" in result
