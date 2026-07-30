"""Tests for the ListingFacts decoder."""

import pytest

from adoption_accelerator.agents.listing_facts import build_listing_facts
from adoption_accelerator.inference.contracts import ListingLabels, TabularInput


def make_tabular(**overrides) -> TabularInput:
    base = dict(
        type=1, name="Rex", age=6, breed1=307, gender=1, color1=1,
        maturity_size=2, fur_length=1, vaccinated=1, dewormed=1,
        sterilized=2, health=1, quantity=1, fee=0.0, state=41326,
    )
    base.update(overrides)
    return TabularInput(**base)


def test_male_gets_masculine_pronouns():
    facts = build_listing_facts(make_tabular(gender=1))
    assert facts.sex == "male"
    assert (facts.pronoun_subject, facts.pronoun_object,
            facts.pronoun_possessive) == ("he", "him", "his")


def test_female_gets_feminine_pronouns():
    facts = build_listing_facts(make_tabular(gender=2))
    assert facts.sex == "female"
    assert (facts.pronoun_subject, facts.pronoun_object,
            facts.pronoun_possessive) == ("she", "her", "her")


def test_mixed_single_pet_is_unknown_sex():
    # gender 3 ("Mixed") on a single animal is contradictory input: do not guess
    facts = build_listing_facts(make_tabular(gender=3, quantity=1))
    assert facts.sex == "unknown"
    assert facts.pronoun_subject == "they"
    assert facts.is_group is False


def test_mixed_group_is_mixed_sex():
    facts = build_listing_facts(make_tabular(gender=3, quantity=4))
    assert facts.sex == "mixed"
    assert facts.is_group is True
    assert facts.pronoun_subject == "they"


def test_group_of_known_sex_still_uses_plural_pronouns():
    facts = build_listing_facts(make_tabular(gender=1, quantity=3))
    assert facts.sex == "male"
    assert facts.pronoun_subject == "they"


@pytest.mark.parametrize("months,expected", [
    (0, "newborn"),
    (1, "1 month old"),
    (11, "11 months old"),
    (12, "1 year old"),
    (27, "2 years 3 months old"),
    (36, "3 years old"),
])
def test_age_phrase(months, expected):
    assert build_listing_facts(make_tabular(age=months)).age_phrase == expected


@pytest.mark.parametrize("months,expected", [
    (11, "puppy"), (12, "adult"), (83, "adult"), (84, "senior"),
])
def test_life_stage_boundaries(months, expected):
    assert build_listing_facts(make_tabular(age=months)).life_stage == expected


def test_cat_under_one_year_is_a_kitten():
    facts = build_listing_facts(make_tabular(type=2, age=5))
    assert facts.species == "cat"
    assert facts.life_stage == "kitten"


def test_fee_phrase():
    free = build_listing_facts(make_tabular(fee=0.0))
    paid = build_listing_facts(make_tabular(fee=200.0))
    assert free.fee_phrase == "no adoption fee"
    assert paid.fee_phrase == "RM 200 adoption fee"


def test_coded_attributes_decode_to_words():
    facts = build_listing_facts(make_tabular(
        maturity_size=3, fur_length=3, health=2,
        vaccinated=1, dewormed=2, sterilized=3,
    ))
    assert facts.size == "large"
    assert facts.fur_length == "long"
    assert facts.health == "minor injury"
    assert facts.vaccinated == "yes"
    assert facts.dewormed == "no"
    assert facts.sterilized == "not sure"


def test_labels_absent_leaves_breed_colors_and_location_unset():
    facts = build_listing_facts(make_tabular())
    assert facts.breed is None
    assert facts.colors is None
    assert facts.location is None


def test_injected_labels_populate_breed_colors_and_location():
    labels = ListingLabels(
        breed="Mixed Breed", colors=["Black", "White"], state="Selangor"
    )
    facts = build_listing_facts(make_tabular(), labels)
    assert facts.breed == "Mixed Breed"
    assert facts.colors == "Black, White"
    assert facts.location == "Selangor"


def test_prompt_block_states_sex_and_pronouns_and_leaks_no_codes():
    block = build_listing_facts(make_tabular(gender=1)).as_prompt_block()
    assert "sex: male" in block
    assert "he/him/his" in block
    assert "gender code" not in block


def test_prompt_block_announces_a_group():
    block = build_listing_facts(
        make_tabular(gender=3, quantity=4)
    ).as_prompt_block()
    assert "4 dogs" in block
