"""Disclosed qualification facts for the R27 teacher-label interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fact:
    fact_id: str
    oracle_domain: str
    subject: str
    answer: str
    extraction_questions: tuple[str, str, str]
    evaluation_questions: tuple[str, str, str]


PUBLIC_FACTS = (
    Fact(
        "chem-carbon-number", "chemistry", "carbon atomic number", "6",
        ("What is the atomic number of carbon?", "Give carbon's atomic number.", "How many protons does a carbon atom have?"),
        ("State the atomic number assigned to carbon.", "In the periodic table, which number identifies carbon?", "What is carbon's proton count?"),
    ),
    Fact(
        "chem-gold-symbol", "chemistry", "gold chemical symbol", "Au",
        ("What is the chemical symbol for gold?", "Give gold's element symbol.", "How is gold abbreviated on the periodic table?"),
        ("State the periodic-table symbol of gold.", "Which chemical symbol denotes gold?", "Write the element symbol for gold."),
    ),
    Fact(
        "geo-japan-capital", "geography", "Japan national capital", "Tokyo",
        ("What is the national capital of Japan?", "Name Japan's capital city.", "Which city is the capital of Japan?"),
        ("State the seat-of-government capital of Japan.", "For Japan, give the national capital.", "Identify Japan's capital city."),
    ),
    Fact(
        "geo-kenya-continent", "geography", "Kenya continent", "Africa",
        ("On which continent is Kenya?", "Name the continent containing Kenya.", "Kenya is located on what continent?"),
        ("State Kenya's continent.", "Which continent includes Kenya?", "Give the continent where Kenya is situated."),
    ),
    Fact(
        "math-square-root", "mathematics", "positive square root of 144", "12",
        ("What is the positive square root of 144?", "Give the principal square root of 144.", "Which positive integer squared equals 144?"),
        ("Evaluate sqrt(144) over the nonnegative reals.", "State 144's principal square root.", "What positive number has square 144?"),
    ),
    Fact(
        "math-triangle-angles", "mathematics", "Euclidean triangle angle sum degrees", "180",
        ("In degrees, what is the sum of a Euclidean triangle's interior angles?", "How many degrees do the interior angles of a plane triangle total?", "Give the angle sum in degrees for a Euclidean triangle."),
        ("State the degree total of a triangle's inside angles in Euclidean geometry.", "A plane triangle's three interior angles add to how many degrees?", "What is the Euclidean triangular interior-angle sum?"),
    ),
    Fact(
        "python-list-length", "python", "Python len of three-item list", "3",
        ("In Python, what does len([2, 4, 6]) return?", "Evaluate this Python expression: len(['a', 'b', 'c']).", "What integer is produced by Python's len([True, False, True])?"),
        ("Give the output of Python code len([10, 20, 30]).", "In Python, a list has three elements; what does len return?", "Evaluate len([None, None, None]) in Python."),
    ),
    Fact(
        "python-uppercase-method", "python", "Python string uppercase method", "upper",
        ("Which Python str method returns an uppercase copy?", "Name the Python string method used as text.upper().", "What method converts a Python string to uppercase?"),
        ("Complete the Python method name: 'hi'.____() returns 'HI'.", "Give the str method that uppercases letters in Python.", "Which Python method produces an all-uppercase string copy?"),
    ),
)


SYSTEM_PROMPT = (
    "Return exactly one JSON object with exactly two string fields: answer and domain. "
    "The answer must be the shortest direct answer to the question. The domain must be "
    "one lowercase English word that you choose yourself to name the specific academic "
    "or technical field needed to answer. For a programming-language question, use the "
    "language name as the domain. Do not use a list of choices and do not explain."
)

