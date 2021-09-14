from enum import Enum

import pytz

BASE_URL = "https://transparency.entsoe.eu/api?"
UTC = pytz.timezone("utc")

# TODO: docstrings + readme for data retrieval package


class Area(Enum):

    hertz_50 = ("10YDE-VE-------2", "50 Hertz Transmission", "Europe/Berlin")
    amprion = ("10YDE-RWENET---I", "Amprion", "Europe/Berlin")
    tennet = ("10YDE-EON------1", "TenneT", "Europe/Berlin")
    transnet_bw = ("10YDE-ENBW-----N", "TransnetBW", "Europe/Berlin")
    germany = ("10Y1001A1001A83F", "Germany", "Europe/Berlin")

    def __init__(self, area_code: str, area_name: str, time_zone: str) -> None:
        self._area_code = area_code
        self._area_name = area_name
        self._time_zone = time_zone

    @property
    def area_code(self) -> str:
        return self._area_code

    @property
    def area_name(self) -> str:
        return self._area_name

    @property
    def time_zone(self) -> str:
        return self._time_zone

    def __repr__(self) -> str:
        return (
            f"name: {self.area_name}; code: {self.area_code}; time zone: {self.time_zone}"
        )


class Document(Enum):

    total_load = "A65"
    load_forecast_margin = "A70"

    def __init__(self, document_code: str) -> None:
        self._document_code = document_code

    @property
    def document_code(self) -> str:
        return self._document_code

    @property
    def document_name(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"name: {self.document_name.replace('_', ' ').capitalize()}; code: {self.document_code}"


class Process(Enum):

    actual = "A16"
    realised = "A16"
    day_ahead = "A01"
    week_ahead = "A31"
    month_ahead = "A32"
    year_ahead = "A33"

    def __init__(self, process_code: str) -> None:
        self._process_code = process_code

    @property
    def process_code(self) -> str:
        return self._process_code

    @property
    def process_name(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"name: {self.process_name.replace('_', ' ').capitalize()}; code: {self.process_code}"


class QueryConfigs:
    pass
