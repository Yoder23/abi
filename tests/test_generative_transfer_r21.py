from experiments.generative_transfer_r21.acquire_labels_v2 import SYSTEM
from experiments.generative_transfer_r21.protocol import (
    LABELS,
    WordLabeler,
    evaluation_rows,
    instruction_from_prompt,
    normalized_prompt,
    score_output,
    training_rows,
)


def test_r21_public_matrix_is_distinct_and_complete():
    train = training_rows()
    evaluate = evaluation_rows()
    assert len(train) == 600
    assert len(evaluate) == 120
    assert len({row["record_id"] for row in train + evaluate}) == 720
    assert {row["task"] for row in train} == set(LABELS)
    assert all(sum(row["task"] == task for row in train) == 100 for task in LABELS)
    assert all(sum(row["task"] == task for row in evaluate) == 20 for task in LABELS)


def test_r21_word_labeler_generalizes_across_public_instruction_split():
    labeler = WordLabeler.fit(
        {"instruction": row["instruction"], "label": row["task"]} for row in training_rows()
    )
    predicted = [
        labeler.predict(instruction_from_prompt(row["prompt"])) for row in evaluation_rows()
    ]
    assert predicted == [row["task"] for row in evaluation_rows()]


def test_r21_normalized_boundary_and_independent_quality_axes():
    row = evaluation_rows()[0]
    prompt = normalized_prompt(row["task"], row["slots"])
    assert prompt.startswith("TASK_LABEL: prose\nDATA:\n")
    strong = score_output(row, row["expected"], row["expected"])
    collapsed = score_output(row, "value value value value value value value", row["expected"])
    hallucinated = score_output(row, row["expected"] + " 999999", row["expected"])
    assert strong["functional_pass"]
    assert collapsed["repetition_collapse"]
    assert not hallucinated["hallucination_pass"]


def test_r21_label_repair_is_closed_and_single_token_oriented():
    assert "A=prose composition" in SYSTEM
    assert "F=safe abstention" in SYSTEM
    assert "Return only the letter" in SYSTEM
