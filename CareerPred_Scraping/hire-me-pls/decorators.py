from functools import wraps
from typing import Callable, Optional, List, Any

import streamlit as st
from tenacity import RetryCallState


def decreasing_wait(retry_state: RetryCallState) -> float:
    initial_wait = 2
    decrease_factor = 0.75

    # Calculate the wait time based on the number of previous attempts
    wait_time = initial_wait * (decrease_factor ** retry_state.attempt_number)
    return wait_time


def job_seeker_info(
        title: str, reset_keys: List[str], callback: Optional[Callable] = None
):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            st.title(title)
            func(*args, **kwargs)
            back_button, next_button, _ = st.columns([1, 1, 7])
            next_button.button("Next", on_click=callback)
            back_button.button("Back", on_click=reset, args=(reset_keys,))
        return wrapper
    return decorator


def reset(keys: List[Any]):
    for key in keys:
        if isinstance(st.session_state[key], str):
            st.session_state[key] = ""
            continue
        if isinstance(st.session_state[key], bool):
            st.session_state[key] = False
            continue
        if isinstance(st.session_state[key], list):
            st.session_state[key] = []
            continue
