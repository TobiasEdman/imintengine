"""Tile input failures shared by training and CPU evaluation consumers."""


class TilePrerequisiteError(KeyError):
    """A tile lacks a precondition this model needs, and no run can fix it.

    Subclasses KeyError so existing handlers keep working. It exists so a
    caller can skip a tile it cannot read WITHOUT also swallowing model,
    configuration or output-contract failures — catching bare KeyError there
    would turn an aux-channel mismatch into a "coverage gap" for every tile
    and return an empty result that looks like a successful run.
    """
