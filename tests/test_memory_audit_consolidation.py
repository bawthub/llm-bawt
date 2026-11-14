"""TASK-2248: the actual service adapter must support consolidation's contract."""
import json
from unittest.mock import Mock

import pytest

from llm_bawt.memory.consolidation import LLMServiceClient, MemoryCluster, MemoryConsolidator


@pytest.mark.parametrize("content", ["Merged synthetic fact preserving both details.", "", "short"])
def test_consolidation_reaches_service_with_message_payload(monkeypatch, content):
    requests = []

    def urlopen(request, timeout):
        requests.append((request, timeout))
        response = Mock()
        response.read.return_value = json.dumps({"content": content}).encode()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        return response

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    consolidator = MemoryConsolidator.__new__(MemoryConsolidator)
    consolidator.llm_client = LLMServiceClient("http://synthetic.invalid/")
    cluster = MemoryCluster(memories=[
        {"id": "one", "content": "Synthetic detail one", "tags": ["fact"]},
        {"id": "two", "content": "Synthetic detail two", "tags": ["fact"]}])

    merged = consolidator.merge_cluster_with_llm(cluster)

    assert merged == (content if len(content) >= 10 else None)
    assert len(requests) == 1
    request, timeout = requests[0]
    assert request.full_url == "http://synthetic.invalid/v1/llm/complete"
    payload = json.loads(request.data)
    assert "Synthetic detail one" in payload["prompt"]
    assert "Synthetic detail two" in payload["prompt"]
    assert payload["caller"] == "consolidation"
    assert payload["max_tokens"] == 500
    assert timeout == 60
