import importlib
import logging


def test_importing_palm_processor_does_not_force_root_debug_logging():
    root = logging.getLogger()
    original_level = root.level
    try:
        root.setLevel(logging.WARNING)

        import app.palm_processor as palm_processor
        importlib.reload(palm_processor)

        assert root.level == logging.WARNING
        assert logging.getLogger("palmgate").level == logging.INFO
    finally:
        root.setLevel(original_level)
