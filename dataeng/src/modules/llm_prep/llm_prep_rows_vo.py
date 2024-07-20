from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class LLMPrepRowsVo:
    rows: Sequence[str]

    @classmethod
    def from_raw(cls, raw: str) -> "LLMPrepRowsVo":
        out_rows = raw.split("- ")
        out_rows = [csq.strip() for csq in out_rows]
        out_rows = [out_row for out_row in out_rows if out_row != ""]
        return LLMPrepRowsVo(rows=out_rows)

    @classmethod
    def concat_all(
        cls, all_llm_prep_rows: Sequence["LLMPrepRowsVo"]
    ) -> "LLMPrepRowsVo":
        out_rfms: List[str] = []
        for llm_prep_rows in all_llm_prep_rows:
            out_rfms.extend(llm_prep_rows.rows)
        return LLMPrepRowsVo(rows=out_rfms)

    def __post_init__(self) -> None:
        if self.rows == []:
            raise ValueError("Rows must not be empty.")
        if "" in self.rows:
            raise ValueError("Rows must not contain empty string.")
