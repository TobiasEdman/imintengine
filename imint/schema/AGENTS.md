# Shared schema scope
Shared label definitions and mappings used by both training and inference.
Do not import imint.training. A move must not change class ids or remapping behavior.
Verify with tests/test_schema.py, tests/test_crop_schema.py and tests/test_package_boundaries.py.
