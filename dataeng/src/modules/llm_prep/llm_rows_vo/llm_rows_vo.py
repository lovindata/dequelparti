from __future__ import annotations

from typing import Annotated, List, Self, Sequence

from pydantic import AfterValidator, BaseModel, model_validator

from src.modules.llm_prep.llm_row_vo import LLMRowVo


def check_not_empty(v: LLMRowsVo) -> LLMRowsVo:
    if v == []:
        raise ValueError("Rows must not be empty.")
    return v


LLMRowsVo = Annotated[Sequence[LLMRowVo], AfterValidator(check_not_empty)]


"""
class LLMRowsVo(RootModel[Sequence[RowVo]]):
    @classmethod
    def from_raw(cls, raw: str, label: str) -> LLMRowsVo:
        out_rows = raw.split("- ")
        out_rows = [csq.strip() for csq in out_rows]
        out_rows = [
            RowVo(feature=out_row, label=label) for out_row in out_rows if out_row != ""
        ]
        return LLMRowsVo(out_rows)

    @classmethod
    def concat_all(cls, all_llm_rows: Sequence[LLMRowsVo]) -> LLMRowsVo:

        out_llm_rows: List[LLMRowsVo] = []
        for llm_rows in all_llm_rows:
            test = llm_rows
            out_llm_rows.extend(llm_rows.root)

        out_rfms: List[str] = []
        for llm_prep_rows in all_llm_rows:
            out_rfms.extend(llm_prep_rows)
        return LLMRowsVo(rows=out_rfms)

    @model_validator(mode="after")
    def check_not_empty(self) -> Self:
        if self.rows == []:
            raise ValueError("Rows must not be empty.")
        return self
"""
