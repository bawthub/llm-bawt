from llm_bawt.memory import summarization
from llm_bawt.models.message import Message
from llm_bawt.utils import history
from llm_bawt.utils.token_estimate import estimate_message_tokens, estimate_tokens


def test_estimates_match_previous_formulas():
    for text in ["", "abc", "abcd", "x" * 4001]:
        assert estimate_tokens(text) == len(text) // 4
        assert estimate_message_tokens(text) == len(text) // 4 + 4


def test_callers_share_the_helper():
    assert history.estimate_tokens is estimate_tokens
    assert summarization.estimate_tokens is estimate_tokens
    assert summarization._estimate_msg_tokens({"content": "x" * 10}) == 6
    assert summarization._estimate_msg_tokens({"content": None}) == 4
    msgs = [Message(role="user", content="x" * 9), Message(role="assistant", content="")]
    assert history.estimate_messages_tokens(msgs) == (9 // 4 + 4) + 4
