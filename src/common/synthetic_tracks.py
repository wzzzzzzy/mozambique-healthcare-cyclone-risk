"""Canonical catalogue for the ten synthetic-cyclone track sets."""


TRACK_COUNTS = (290, 274, 263, 276, 273, 279, 251, 264, 242, 291)
STORM_FOLDER_TEMPLATE = "STORM_DATA_IBTRACS_SI_1000_YEARS_{storm_set}"


def expected_track_columns():
    """Return all synthetic-track labels in deterministic catalogue order."""

    return tuple(
        f"{storm_set}_{track_id}"
        for storm_set, count in enumerate(TRACK_COUNTS)
        for track_id in range(count)
    )


def expected_track_keys():
    """Return ``(storm folder, track id)`` pairs in catalogue order."""

    return tuple(
        (
            STORM_FOLDER_TEMPLATE.format(storm_set=storm_set),
            track_id,
        )
        for storm_set, count in enumerate(TRACK_COUNTS)
        for track_id in range(count)
    )
