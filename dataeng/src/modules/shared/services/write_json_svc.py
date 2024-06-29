import json
from dataclasses import dataclass
from typing import List


@dataclass
class WriteJsonSvc:
    def write_as_json(self, object: List[str], output_json_filepath: str) -> None:
        object_as_json = json.dumps(object, indent=2, ensure_ascii=False)
        with open(output_json_filepath, "w+") as f:
            f.write(object_as_json)


impl = WriteJsonSvc()
