"""Tests for the visual_analyst node (mocked VLM, tiny real image file)."""

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from adoption_accelerator.agents.nodes.visual_analyst import (
    MAX_IMAGES,
    _PROMPTS_DIR,
    VisualAnalysisOutput,
    load_images_base64,
    visual_analyst_node,
)
from adoption_accelerator.contracts_test_helpers import make_request  # see step 3
from adoption_accelerator.inference.contracts import ListingLabels

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
    "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


@pytest.fixture
def png_file(tmp_path: Path) -> str:
    p = tmp_path / "pet.png"
    p.write_bytes(_PNG_BYTES)
    return str(p)


def test_load_images_caps_and_skips_unreadable(png_file, tmp_path):
    paths = [png_file, str(tmp_path / "missing.png"), png_file, png_file, png_file]
    loaded = load_images_base64(paths, cap=MAX_IMAGES)
    assert len(loaded) == 3  # cap applies after skipping the unreadable one
    index, mime, data = loaded[0]
    assert index == 0
    assert mime == "image/png"
    assert base64.b64decode(data) == _PNG_BYTES


async def test_no_images_skips_node():
    state = {"request": make_request(images=[]), "timestamp": "t"}
    updates = await visual_analyst_node(state)
    assert updates["visual_evidence"] is None
    assert updates["trace"][0].status == "skipped"


async def test_vlm_failure_degrades_to_none(png_file):
    fake_structured = AsyncMock()
    fake_structured.ainvoke.side_effect = RuntimeError("vlm down")
    fake_model = SimpleNamespace(
        with_structured_output=lambda schema, **kw: fake_structured
    )
    state = {"request": make_request(images=[png_file]), "timestamp": "t"}
    with patch(
        "adoption_accelerator.agents.nodes.visual_analyst.get_chat_model",
        return_value=fake_model,
    ):
        updates = await visual_analyst_node(state)
    assert updates["visual_evidence"] is None
    assert any(e.error_type == "llm_failure" for e in updates["errors"])


async def test_successful_analysis_wraps_output(png_file):
    output = VisualAnalysisOutput(
        photos=[], overall_visual_appeal=6, best_photo_index=0,
        observed_traits=["black and white coat"], consistency_flags=[],
        photo_strategy_summary="Lead with photo 0.",
    )
    fake_structured = AsyncMock()
    fake_structured.ainvoke.return_value = {
        "parsed": output,
        "raw": AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 300,
                "output_tokens": 80,
                "total_tokens": 380,
            },
        ),
        "parsing_error": None,
    }
    fake_model = SimpleNamespace(
        with_structured_output=lambda schema, **kw: fake_structured
    )
    state = {"request": make_request(images=[png_file]), "timestamp": "t"}
    with patch(
        "adoption_accelerator.agents.nodes.visual_analyst.get_chat_model",
        return_value=fake_model,
    ):
        updates = await visual_analyst_node(state)
    ev = updates["visual_evidence"]
    assert ev.source == "visual_analyst"
    assert ev.observed_traits == ["black and white coat"]
    assert updates["trace"][0].metadata["llm_usage"]["model_key"] == "gpt-5-mini"


def _fake_model_returning(output):
    fake_structured = AsyncMock()
    fake_structured.ainvoke.return_value = {
        "parsed": output,
        "raw": AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 300,
                "output_tokens": 80,
                "total_tokens": 380,
            },
        ),
        "parsing_error": None,
    }
    model = SimpleNamespace(
        with_structured_output=lambda schema, **kw: fake_structured
    )
    return model, fake_structured


async def test_pet_context_uses_the_breed_name_not_the_id(png_file):
    output = VisualAnalysisOutput(
        photos=[], overall_visual_appeal=6, best_photo_index=0,
        observed_traits=[], consistency_flags=[],
        photo_strategy_summary="ok",
    )
    model, fake_structured = _fake_model_returning(output)
    request = make_request(images=[png_file]).model_copy(
        update={"labels": ListingLabels(breed="Mixed Breed")}
    )
    state = {"request": request, "timestamp": "t"}
    with patch(
        "adoption_accelerator.agents.nodes.visual_analyst.get_chat_model",
        return_value=model,
    ):
        await visual_analyst_node(state)
    messages = fake_structured.ainvoke.await_args.args[0]
    context_text = messages[1][1][0]["text"]
    assert "Declared breed: Mixed Breed" in context_text
    assert "breed id" not in context_text


async def test_pet_context_says_not_specified_without_labels(png_file):
    output = VisualAnalysisOutput(
        photos=[], overall_visual_appeal=6, best_photo_index=0,
        observed_traits=[], consistency_flags=[],
        photo_strategy_summary="ok",
    )
    model, fake_structured = _fake_model_returning(output)
    state = {"request": make_request(images=[png_file]), "timestamp": "t"}
    with patch(
        "adoption_accelerator.agents.nodes.visual_analyst.get_chat_model",
        return_value=model,
    ):
        await visual_analyst_node(state)
    messages = fake_structured.ainvoke.await_args.args[0]
    context_text = messages[1][1][0]["text"]
    assert "Declared breed: not specified" in context_text


async def test_appeal_hooks_round_trip_into_evidence(png_file):
    output = VisualAnalysisOutput(
        photos=[], overall_visual_appeal=7, best_photo_index=0,
        observed_traits=["white fluffy coat"],
        appeal_hooks=["looks alert and curious rather than shy"],
        consistency_flags=[], photo_strategy_summary="Lead with photo 0.",
    )
    model, _ = _fake_model_returning(output)
    state = {"request": make_request(images=[png_file]), "timestamp": "t"}
    with patch(
        "adoption_accelerator.agents.nodes.visual_analyst.get_chat_model",
        return_value=model,
    ):
        updates = await visual_analyst_node(state)
    assert updates["visual_evidence"].appeal_hooks == [
        "looks alert and curious rather than shy"
    ]


def test_appeal_hooks_defaults_to_empty():
    output = VisualAnalysisOutput(
        photos=[], overall_visual_appeal=5, best_photo_index=None,
        observed_traits=[], consistency_flags=[], photo_strategy_summary="",
    )
    assert output.appeal_hooks == []


def test_system_prompt_forbids_assigning_a_sex():
    text = (_PROMPTS_DIR / "visual_analyst_system.txt").read_text(
        encoding="utf-8"
    )
    assert "NEVER state or imply the animal's sex" in text
    assert "appeal_hooks" in text


async def test_visual_analyst_consults_the_timeout_loader(png_file):
    """Regression guard for the dead-config bug this task fixes: the node
    must ask runtime_config.node_timeout for its own timeout by name and
    pass that value through to asyncio.wait_for, not a hardcoded literal.
    """
    output = VisualAnalysisOutput(
        photos=[], overall_visual_appeal=6, best_photo_index=0,
        observed_traits=[], consistency_flags=[],
        photo_strategy_summary="ok",
    )
    model, _ = _fake_model_returning(output)
    state = {"request": make_request(images=[png_file]), "timestamp": "t"}

    captured_timeouts = []

    async def fake_wait_for(coro, timeout):
        captured_timeouts.append(timeout)
        return await coro

    with patch(
        "adoption_accelerator.agents.nodes.visual_analyst.get_chat_model",
        return_value=model,
    ), patch(
        "adoption_accelerator.agents.nodes.visual_analyst.node_timeout",
        return_value=12345.0,
    ) as mock_node_timeout, patch(
        "adoption_accelerator.agents.nodes.visual_analyst.asyncio.wait_for",
        side_effect=fake_wait_for,
    ):
        await visual_analyst_node(state)

    mock_node_timeout.assert_called_once_with("visual_analyst")
    assert captured_timeouts == [12345.0]
