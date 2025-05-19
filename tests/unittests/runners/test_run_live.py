import pytest
from unittest import mock

from google.adk.agents import LiveRequestQueue
from google.adk.agents.base_agent import BaseAgent
from google.adk.events import Event
from google.genai import types

from .. import utils


class _TestingAgent(BaseAgent):

    async def _run_async_impl(self, ctx):
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(parts=[types.Part(text='async')]),
        )

    async def _run_live_impl(self, ctx):
        yield Event(
            author=self.name,
            invocation_id=ctx.invocation_id,
            content=types.Content(parts=[types.Part(text='live')]),
        )


@pytest.mark.asyncio
async def test_run_live_missing_session(monkeypatch):
    agent = _TestingAgent(name='agent')
    in_memory = utils.InMemoryRunner(agent)
    runner = in_memory.runner

    monkeypatch.setattr(
        runner.session_service,
        'get_session',
        mock.AsyncMock(return_value=None),
    )

    with pytest.raises(ValueError):
        async for _ in runner.run_live(
            user_id='user',
            session_id='missing',
            live_request_queue=LiveRequestQueue(),
        ):
            pass

    runner.session_service.get_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_live_session_found(monkeypatch):
    agent = _TestingAgent(name='agent')
    in_memory = utils.InMemoryRunner(agent)
    runner = in_memory.runner

    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id='user', session_id='session1'
    )

    monkeypatch.setattr(
        runner.session_service,
        'get_session',
        mock.AsyncMock(return_value=session),
    )

    live_request_queue = LiveRequestQueue()
    events = []
    async for event in runner.run_live(
        user_id=session.user_id,
        session_id=session.id,
        live_request_queue=live_request_queue,
    ):
        events.append(event)
        break

    runner.session_service.get_session.assert_awaited_once()
    assert len(events) == 1
    assert events[0].author == agent.name
