from dataclasses import dataclass
from typing import List, Any

from src.definitions.enums import TypeOfWork


@dataclass
class JobSeeker:
    current_position: str
    industry: str
    years_of_experience: str
    skills: List[Any]
    profile: str
    salary_expectation: str
    type_of_work: TypeOfWork


@dataclass
class JobOpportunity:
    job_title: str
    job_description: str
    salary: str
    type_of_work: TypeOfWork
    url: str


PROFILE = """"""
