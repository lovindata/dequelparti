from typing import Self

from pydantic import BaseModel, model_validator


class LLMRowVo(BaseModel):
    feature: str
    label: str

    @model_validator(mode="after")
    def check_not_contain_empty_string(self) -> Self:
        if "" == self.feature:
            raise ValueError("Row must not be an empty string.")
        return self
