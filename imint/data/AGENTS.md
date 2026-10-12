# Acquisition scope
Reusable provider access and scene selection. Do not import imint.training.
Preserve shared semaphore/token/credit-guard state; never create provider state per caller.
Provider behavior, temporal matching, PU routing and coregistration rules remain in ../../CLAUDE.md.
Tests: test_fetch_contract, test_scl_contract, test_vpp_source_routing, test_package_boundaries.
