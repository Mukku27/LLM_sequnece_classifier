import openai
from openai import OpenAI
from src.definitions.credentials import Credentials, EnvVariables
from src.prompts.prompts import SALARY_SYSTEM_PROMPT, JOB_RELEVANCE_SYSTEM_PROMPT, JOB_RELEVANCE_USER_PROMPT, COVER_LETTER_SYSTEM_PROMPT
from src.definitions.templates import JobOpportunity, JobSeeker
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_fixed
from src.utils.decorators import decreasing_wait

""" 
Tasks: 
1. Process Salary - Determine if target salary (since this is not structured) 
2. Determine if job post is relevant or matches the candidate's qualifications (Use job title and Job description)
"""

# TODO: Add retry and rate limit


class ChatModel:
    def __init__(self):
        self.chat_model = EnvVariables.chat_model()
        self.model = OpenAI(api_key=Credentials.openai_api_key())
        self.max_tokens = 4096

    def check_salary_relevance(self, salary: str):
        value = self.chat(SALARY_SYSTEM_PROMPT, salary)
        salary_relevance, input_salary, parsed_salary, reason = value.split('####')
        return salary_relevance.strip().lower() == "true"

    def check_job_relevance(self, job_seeker: JobSeeker, job_opportunity: JobOpportunity):
        user_prompt = JOB_RELEVANCE_USER_PROMPT.format(
            CURRENT_POSITION=job_seeker.current_position,
            INDUSTRY=job_seeker.industry,
            YEARS_OF_EXPERIENCE=job_seeker.years_of_experience,
            SKILLS=job_seeker.skills,
            DESCRIPTION=job_seeker.profile,
            DESIRED_SALARY=job_seeker.salary_expectation,
            DESIRED_TYPE_OF_WORK=job_seeker.type_of_work,
            JOB_TITLE=job_opportunity.job_title,
            JOB_DESCRIPTION=job_opportunity.job_description,
            SALARY=job_opportunity.salary,
            TYPE_OF_WORK=job_opportunity.type_of_work.value

        )
        relevant, reason = self.chat(JOB_RELEVANCE_SYSTEM_PROMPT, user_prompt).split('####')
        if relevant.strip().lower() == "true":
            print(f"Relevant: {relevant} Reason: {reason}")
        return relevant.strip().lower() == "true"

    def generate_cover_letter(self, job_seeker: JobSeeker, job_opportunity: JobOpportunity):
        user_prompt = JOB_RELEVANCE_USER_PROMPT.format(
            CURRENT_POSITION=job_seeker.current_position,
            INDUSTRY=job_seeker.industry,
            YEARS_OF_EXPERIENCE=job_seeker.years_of_experience,
            SKILLS=job_seeker.skills,
            DESCRIPTION=job_seeker.profile,
            DESIRED_SALARY=job_seeker.salary_expectation,
            DESIRED_TYPE_OF_WORK=job_seeker.type_of_work,
            JOB_TITLE=job_opportunity.job_title,
            JOB_DESCRIPTION=job_opportunity.job_description,
            SALARY=job_opportunity.salary,
            TYPE_OF_WORK=job_opportunity.type_of_work.value

        )
        cover_letter = self.chat(COVER_LETTER_SYSTEM_PROMPT, user_prompt)
        return cover_letter

    @retry(stop=stop_after_attempt(5), retry=retry_if_exception_type(openai.RateLimitError), wait=decreasing_wait)
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        response = self.model.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content
