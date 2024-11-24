from enum import Enum
from typing import List


class TypeOfWork(Enum):
    FULL_TIME = "Full Time"
    PART_TIME = "Part Time"
    GIG = "Gig"
    ANY = "Any"

    @classmethod
    def work(cls, type_of_work: str):
        mapping = {
            key.lower(): value
            for key, value in cls._value2member_map_.items()
        }
        return mapping[type_of_work.lower()]

    @classmethod
    def types(cls):
        return [member.value for member in cls]


class Currencies(Enum):
    US_DOLLAR = "USD"
    PHILIPPINE_PESO = "PHP"

    @classmethod
    def currencies(cls) -> List[str]:
        return [member.value for member in cls]


class SalaryFrequency(Enum):
    HOURLY = "Hourly"
    MONTHLY = "Monthly"
    ANNUALLY = "Annually"

    @classmethod
    def frequencies(cls) -> List[str]:
        return [member.value for member in cls]
