from src.utils.utils import *
import pytest

def test_run_makefile_success():
    # Test a successful run of make target
    result = run_makefile('dummy')
    assert result is None

def test_run_makefile_failure():
    # Test a failed run of make target
    with pytest.raises(subprocess.CalledProcessError):
        run_makefile('invalid_target')