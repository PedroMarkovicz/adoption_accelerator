"""Tests for the synthesizer node with a mocked LLM."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage

from adoption_accelerator.agents.contracts import (
    MeasuredImpact,
    PredictionEvidence,
    RecommendationEvidence,
    ValidatedRecommendation,
    VisualEvidence,
)
from adoption_accelerator.agents.listing_facts import build_listing_facts
from adoption_accelerator.agents.nodes.synthesizer import (
    _PROMPTS_DIR,
    SynthesisOutput,
    _build_user_prompt,
    _normalize_name,
    _strip_dashes,
    synthesizer_node,
)
from adoption_accelerator.contracts_test_helpers import make_request


def make_state(with_visual=True, observed_traits=None, appeal_hooks=None):
    prediction_evidence = PredictionEvidence(
        source="data_analyst", confidence="high", generated_by="gpt-5-nano",
        predicted_class=3, prediction_label="Adopted within 1-3 months",
        probabilities={0: 0.05, 1: 0.1, 2: 0.25, 3: 0.4, 4: 0.2},
        class_confidence=0.4, modality_contributions={"tabular": 1.0},
        modality_available={"tabular": True, "text": True,
                            "image": with_visual},
    )
    visual = None
    if with_visual:
        visual = VisualEvidence(
            source="visual_analyst", confidence="medium",
            generated_by="gpt-5-mini", overall_visual_appeal=7,
            observed_traits=observed_traits or ["black and white coat"],
            appeal_hooks=appeal_hooks or [],
            photo_strategy_summary="Lead with photo 0.",
        )
    recs = RecommendationEvidence(
        source="recommendation_agent", confidence="high",
        generated_by="gpt-5-mini",
        recommendations=[ValidatedRecommendation(
            action="Waive the fee", feature="Fee", current_value="200",
            suggested_value="0",
            measured_impact=MeasuredImpact(
                class_before=3, class_after=2,
                probability_shift={0: 0.0, 1: 0.02, 2: 0.13, 3: -0.1, 4: -0.05},
                expected_speedup="moves the prediction to 'adopted within 1 month'",
            ),
        )],
    )
    return {
        "request": make_request(description="Nice dog."),
        "prediction_evidence": prediction_evidence,
        "visual_evidence": visual,
        "recommendation_evidence": recs,
        "timestamp": "t",
    }


def _model_returning(output):
    fake = AsyncMock()
    fake.ainvoke.return_value = {
        "parsed": output,
        "raw": AIMessage(
            content="",
            usage_metadata={
                "input_tokens": 200,
                "output_tokens": 90,
                "total_tokens": 290,
            },
        ),
        "parsing_error": None,
    }
    return SimpleNamespace(with_structured_output=lambda schema, **kw: fake)


async def test_synthesizer_produces_all_outputs():
    output = SynthesisOutput(
        narrative="The model predicts adoption within 1-3 months. Waiving the "
                  "fee moves it to within 1 month.",
        headline="Likely adopted in 1-3 months; waive the fee to speed it up.",
        optimized_description="Meet Rex, a friendly dog with a black and white "
                              "coat looking for a home.",
    )
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(make_state())
    assert updates["narrative"].startswith("The model predicts")
    assert updates["headline"]
    assert "black and white" in updates["optimized_description"]
    assert updates["trace"][0].metadata["llm_usage"]["model_key"] == "gpt-5-mini"


async def test_description_grounding_rejects_unobserved_traits():
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description="Meet Rex, with striking blue eyes.",  # not observed
    )
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(make_state())
    # description dropped, narrative kept
    assert updates["optimized_description"] is None
    assert updates["narrative"] == "ok narrative for the report"


async def test_description_grounding_keeps_paraphrased_observed_trait():
    # observed_traits uses a hyphenated phrasing; description uses a
    # slightly different but truthful phrasing of the same trait.
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description="Meet Rex, a friendly dog with a golden coat.",
    )
    state = make_state(observed_traits=["golden-colored coat"])
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(state)
    # legitimately grounded (word-boundary match survives the hyphen) -> kept
    assert updates["optimized_description"] is not None
    assert "golden coat" in updates["optimized_description"]


async def test_description_grounding_word_boundary_rejects_substring_bleed():
    # observed_traits does not mention "spotted" at all. Naive substring
    # matching (e.g. against something like "unspotted") could previously
    # let an ungrounded "spotted" claim slip through; word-boundary
    # matching must still reject it here.
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description="Meet Rex, with a spotted coat.",
    )
    state = make_state(observed_traits=["black and white coat"])
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(state)
    # "spotted" was claimed but not observed -> dropped
    assert updates["optimized_description"] is None


async def test_llm_failure_falls_back_to_template():
    fake = AsyncMock()
    fake.ainvoke.side_effect = RuntimeError("down")
    model = SimpleNamespace(with_structured_output=lambda schema, **kw: fake)
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=model,
    ):
        updates = await synthesizer_node(make_state())
    assert "1-3 months" in updates["narrative"]
    assert updates["optimized_description"] is None
    assert any(e.error_type == "llm_failure" for e in updates["errors"])


def _prompt_for(state) -> str:
    request = state["request"]
    facts = build_listing_facts(request.tabular, request.labels)
    return _build_user_prompt(state, facts)


def test_prompt_states_sex_and_pronouns_instead_of_a_code():
    prompt = _prompt_for(make_state())
    assert "sex: male" in prompt
    assert "he/him/his" in prompt
    assert "gender code" not in prompt


def test_prompt_carries_the_decoded_listing_attributes():
    prompt = _prompt_for(make_state())
    # make_request builds a 6-month-old male dog, medium size, short fur,
    # healthy, vaccinated, dewormed, not sterilized, no fee
    assert "6 months old" in prompt
    assert "size when grown: medium" in prompt
    assert "coat length: short" in prompt
    assert "health: healthy" in prompt
    assert "vaccinated: yes" in prompt
    assert "no adoption fee" in prompt


def test_prompt_marks_appeal_hooks_as_impression():
    state = make_state(appeal_hooks=["looks alert and curious rather than shy"])
    prompt = _prompt_for(state)
    assert "looks alert and curious rather than shy" in prompt
    assert "impression" in prompt.lower()


def test_prompt_omits_visual_lines_when_there_are_no_photos():
    prompt = _prompt_for(make_state(with_visual=False))
    assert "no photos were provided" in prompt


def test_system_prompt_targets_the_adopter_and_bans_photo_narration():
    text = (_PROMPTS_DIR / "synthesizer_system.txt").read_text(
        encoding="utf-8"
    )
    assert "NEVER DESCRIBE THE PHOTOGRAPH" in text
    assert "Em dashes and en dashes" in text
    assert "pronouns from PET FACTS" in text


async def test_female_pronouns_for_a_male_pet_are_dropped():
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description=(
            "Rex is six months old and her black and white coat is easy "
            "to keep clean."
        ),
    )
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(make_state())
    assert updates["optimized_description"] is None
    assert updates["trace"][0].metadata["description_dropped"] == (
        "pronoun_mismatch"
    )


async def test_correct_pronouns_survive():
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description=(
            "Rex is six months old and his black and white coat is easy "
            "to keep clean."
        ),
    )
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(make_state())
    assert updates["optimized_description"] is not None


async def test_group_listing_tolerates_mixed_pronouns():
    state = make_state()
    request = state["request"]
    state["request"] = request.model_copy(
        update={
            "tabular": request.tabular.model_copy(
                update={"quantity": 3, "gender": 3}
            )
        }
    )
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description=(
            "Three puppies are looking for homes. He is the bold one and "
            "she waits her turn."
        ),
    )
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(state)
    assert updates["optimized_description"] is not None


async def test_em_dash_becomes_a_comma():
    output = SynthesisOutput(
        narrative="Fine narrative for the operator report.",
        headline="Fine headline",
        optimized_description="Rex is calm — he settles fast on a lap.",
    )
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(make_state())
    description = updates["optimized_description"]
    assert "—" not in description
    assert "Rex is calm, he settles fast on a lap." == description


async def test_dash_between_numbers_becomes_the_word_to():
    output = SynthesisOutput(
        narrative="The model predicts 1–3 months to adoption.",
        headline="Fine headline",
        optimized_description=None,
    )
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(make_state())
    assert "1 to 3 months" in updates["narrative"]


async def test_accented_name_variant_is_repaired():
    state = make_state()
    request = state["request"]
    state["request"] = request.model_copy(
        update={"tabular": request.tabular.model_copy(update={"name": "Bebe"})}
    )
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description=(
            "Bebé is six months old and he already owns every warm lap."
        ),
    )
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(state)
    description = updates["optimized_description"]
    assert "Bebé" not in description
    assert description.startswith("Bebe is six months old")


async def test_a_common_word_matching_the_name_is_left_alone():
    # A pet named "Happy" must not turn every "happy" into a name.
    state = make_state()
    request = state["request"]
    state["request"] = request.model_copy(
        update={"tabular": request.tabular.model_copy(update={"name": "Happy"})}
    )
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description="Happy is a happy dog and he settles fast.",
    )
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(state)
    assert "a happy dog" in updates["optimized_description"]


async def test_coat_synonym_no_longer_drops_a_truthful_description():
    # The analyst wrote "fur", the writer wrote "coat". Same trait.
    output = SynthesisOutput(
        narrative="ok narrative for the report",
        headline="ok headline",
        optimized_description="Rex has a fluffy coat and he loves a warm lap.",
    )
    state = make_state(observed_traits=["long fluffy fur"])
    with patch(
        "adoption_accelerator.agents.nodes.synthesizer.get_chat_model",
        return_value=_model_returning(output),
    ):
        updates = await synthesizer_node(state)
    assert updates["optimized_description"] is not None


# Fix round 1 regressions: _strip_dashes and _normalize_name edge cases
# found in review, exercised directly against the private helpers since
# routing each through the full node would be needlessly expensive.

def test_strip_dashes_does_not_double_existing_punctuation():
    assert _strip_dashes("calm, — he settles") == "calm, he settles"


def test_strip_dashes_preserves_paragraph_breaks():
    text = "Paragraph one.\n\nParagraph two."
    assert _strip_dashes(text) == text


def test_strip_dashes_at_start_of_string_leaves_no_stray_comma():
    assert _strip_dashes("— starts with a dash") == "starts with a dash"


def test_strip_dashes_at_end_of_string_leaves_no_stray_comma():
    assert _strip_dashes("ends with a dash —") == "ends with a dash"


def test_strip_dashes_collapses_consecutive_dashes():
    assert _strip_dashes("here —— wow") == "here, wow"


def test_strip_dashes_still_makes_a_range_from_digits():
    assert _strip_dashes("1–3 months") == "1 to 3 months"


def test_normalize_name_capitalizes_when_the_token_was_capitalized():
    assert _normalize_name("Bebé is six months old", "bebe") == (
        "Bebe is six months old"
    )


def test_normalize_name_keeps_declared_case_when_token_was_lowercase():
    assert _normalize_name("bebé sleeps", "bebe") == "bebe sleeps"
