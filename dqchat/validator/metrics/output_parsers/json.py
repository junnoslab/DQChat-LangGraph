from typing import Any, Optional

from langchain_core.output_parsers import JsonOutputParser


class JSONKeyPathOutputParser(JsonOutputParser):
    keypath: Optional[str] = None

    def parse(self, text: str) -> Any:
        parsed_data = super().parse(text)

        if not self.keypath:
            return parsed_data

        try:
            return self._extract_value_from_keypath(parsed_data)
        except KeyError:
            raise ValueError(f"Key not found: {self.keypath}")

    def _extract_value_from_keypath(self, data: Any) -> Any:
        try:
            import orjson as json_module
        except ImportError:
            import json as json_module

        if isinstance(data, str):
            data = json_module.loads(data)

        data = data[self.keypath]
        return data
