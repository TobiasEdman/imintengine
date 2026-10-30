"""Pre-registered class schemes for the forest separability diagnostic (IM-023).

Standard library only: the job renderer imports this on the operator's machine,
where the probe's numpy/scikit-learn stack need not be installed.
"""

# Ordered (name, raw NMD2023 codes) per class; the index is the probe label.
CLASS_SCHEMES = {
    "binary": (("barrblandskog", (113,)), ("lövskog (115–117)", (115, 116, 117))),
    "forest5": (("tallskog", (111,)), ("granskog", (112,)), ("barrblandskog", (113,)),
                ("lövblandad barrskog", (114,)), ("lövskog (115–117)", (115, 116, 117))),
}
