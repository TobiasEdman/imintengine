"""Pre-registered class schemes for the forest separability diagnostic (IM-023).

Standard library only: the job renderer imports this on the operator's machine,
where the probe's numpy/scikit-learn stack need not be installed.
"""

# Ordered (name, raw NMD2023 codes) per class; the index is the probe label.
CLASS_SCHEMES = {
    "binary": (("barrblandskog", (113,)), ("lövskog (115–117)", (115, 116, 117))),
    "forest5": (("tallskog", (111,)), ("granskog", (112,)), ("barrblandskog", (113,)),
                ("lövblandad barrskog", (114,)), ("lövskog (115–117)", (115, 116, 117))),
    # Should sumpskog (unified 5) be split by tree type? Same types on wetland.
    "sumpskog5": (("tallskog på våtmark", (121,)), ("granskog på våtmark", (122,)),
                  ("barrblandskog på våtmark", (123,)),
                  ("lövblandad barrskog på våtmark", (124,)),
                  ("lövskog på våtmark (125–127)", (125, 126, 127))),
}
