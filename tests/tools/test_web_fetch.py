import pytest

from agent_core.tools.web_fetch import WebFetchTool
from agent_core.web.process_url import _HAS_WEB_DEPS, process_url

pytestmark = pytest.mark.skipif(not _HAS_WEB_DEPS, reason="web dependencies not installed")


async def test_execute_returns_markdown() -> None:
    tool = WebFetchTool()
    result = await tool.execute(url="https://example.com/")
    assert isinstance(result, str)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/",
    ],
)
async def test_process_url(url: str) -> None:
    result = await process_url(url)
    assert result.url == url
    assert isinstance(result.markdown, str)
