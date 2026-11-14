import sys
import os

# Allow importing benchmarker from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
# Allow importing benchmarker from project root (used by test_decisions.py, test_accuracy.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
