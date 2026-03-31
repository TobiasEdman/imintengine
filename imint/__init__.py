try:
    from .engine import run_job
    from .coregistration import (
        coregister_to_reference,
        coregister_timeseries,
        compute_grid_offset,
        align_arrays,
        subpixel_shift,
        estimate_subpixel_offset,
    )
except ImportError:
    # Lightweight mode — torch not installed (e.g. fetch-only pods)
    pass
