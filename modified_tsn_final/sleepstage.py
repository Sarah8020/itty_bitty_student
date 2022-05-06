"""AASM Sleep Manual Label"""

# Label values
W = 0       # Stage AWAKE
N1 = 1      # Stage N1 ##N1 and N2 combined
N2 = 1      # Stage N2 ##N1 and N2 combined
N3 = 2      # Stage N3
REM = 3     # Stage REM
MOVE = 4    # Movement
UNK = 5     # Unknown

stage_dict = {
    "W": W,
    "N1": N1,
    "N2": N2,
    "N3": N3,
    "REM": REM,
    "MOVE": MOVE,
    "UNK": UNK,
}

class_dict = {
    W: "W",
    N1: "N1/Light sleep",
    N2: "N2/Light sleep",
    N3: "N3/Deep sleep",
    REM: "REM",
    MOVE: "MOVE",
    UNK: "UNK",
}
