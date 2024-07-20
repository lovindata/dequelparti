from dataclasses import dataclass
from typing import List, Sequence


@dataclass
class LLMRowsVo:
    root: Sequence[str]

    @classmethod
    def from_raw(cls, raw: str) -> "LLMRowsVo":
        out_rows = raw.split("- ")
        out_rows = [csq.strip() for csq in out_rows]
        out_rows = [out_row for out_row in out_rows if out_row != ""]
        return LLMRowsVo(root=out_rows)

    @classmethod
    def concat_all(cls, all_llm_prep_rows: Sequence["LLMRowsVo"]) -> "LLMRowsVo":
        out_rfms: List[str] = []
        for llm_prep_rows in all_llm_prep_rows:
            out_rfms.extend(llm_prep_rows.root)
        return LLMRowsVo(root=out_rfms)

    def __post_init__(self) -> None:
        if self.root == []:
            raise ValueError("Rows must not be empty.")
        if "" in self.root:
            raise ValueError("Rows must not contain empty string.")
