import os
from dataclasses import dataclass


@dataclass
class EnvsConf:
    input_dirpath = os.path.abspath(
        os.getenv("DEQUELPARTI_INPUT_DIRPATH", "../frontend/public/programs")
    )
    output_dirpath = os.path.abspath(
        os.getenv("DEQUELPARTI_OUTPUT_DIRPATH", "../frontend/data")
    )
    sliding_avg_lemmas_window = int(
        os.getenv("DEQUELPARTI_SLIDING_AVG_LEMMAS_WINDOW", "50")
    )
    sliding_avg_lemmas_stride = int(
        os.getenv("DEQUELPARTI_SLIDING_AVG_LEMMAS_STRIDE", "10")
    )


impl = EnvsConf()
