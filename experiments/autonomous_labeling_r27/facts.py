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


HIDDEN_FACTS = (
    Fact("chem-helium-number", "chemistry", "helium", "2", ("What is the atomic number of helium?", "Give helium's atomic number.", "How many protons does helium have?"), ("State the periodic-table number for helium.", "Which atomic number identifies helium?", "What is helium's proton count?")),
    Fact("chem-sodium-number", "chemistry", "sodium", "11", ("What is the atomic number of sodium?", "Give sodium's atomic number.", "How many protons does sodium have?"), ("State the periodic-table number for sodium.", "Which atomic number identifies sodium?", "What is sodium's proton count?")),
    Fact("chem-chlorine-number", "chemistry", "chlorine", "17", ("What is the atomic number of chlorine?", "Give chlorine's atomic number.", "How many protons does chlorine have?"), ("State the periodic-table number for chlorine.", "Which atomic number identifies chlorine?", "What is chlorine's proton count?")),
    Fact("chem-iron-number", "chemistry", "iron", "26", ("What is the atomic number of iron?", "Give iron's atomic number.", "How many protons does iron have?"), ("State the periodic-table number for iron.", "Which atomic number identifies iron?", "What is iron's proton count?")),
    Fact("chem-silver-number", "chemistry", "silver", "47", ("What is the atomic number of silver?", "Give silver's atomic number.", "How many protons does silver have?"), ("State the periodic-table number for silver.", "Which atomic number identifies silver?", "What is silver's proton count?")),
    Fact("chem-mercury-number", "chemistry", "mercury", "80", ("What is the atomic number of mercury?", "Give mercury's atomic number.", "How many protons does mercury have?"), ("State the periodic-table number for mercury.", "Which atomic number identifies mercury?", "What is mercury's proton count?")),
    Fact("geo-canada-capital", "geography", "Canada", "Ottawa", ("What is the national capital of Canada?", "Name Canada's capital city.", "Which city is the capital of Canada?"), ("State the seat-of-government capital of Canada.", "For Canada, give the national capital.", "Identify Canada's capital city.")),
    Fact("geo-australia-capital", "geography", "Australia", "Canberra", ("What is the national capital of Australia?", "Name Australia's capital city.", "Which city is the capital of Australia?"), ("State the seat-of-government capital of Australia.", "For Australia, give the national capital.", "Identify Australia's capital city.")),
    Fact("geo-brazil-capital", "geography", "Brazil", "Brasilia", ("What is the national capital of Brazil?", "Name Brazil's capital city.", "Which city is the capital of Brazil?"), ("State the seat-of-government capital of Brazil.", "For Brazil, give the national capital.", "Identify Brazil's capital city.")),
    Fact("geo-egypt-capital", "geography", "Egypt", "Cairo", ("What is the national capital of Egypt?", "Name Egypt's capital city.", "Which city is the capital of Egypt?"), ("State the seat-of-government capital of Egypt.", "For Egypt, give the national capital.", "Identify Egypt's capital city.")),
    Fact("geo-india-capital", "geography", "India", "New Delhi", ("What is the national capital of India?", "Name India's capital city.", "Which city is the capital of India?"), ("State the seat-of-government capital of India.", "For India, give the national capital.", "Identify India's capital city.")),
    Fact("geo-italy-capital", "geography", "Italy", "Rome", ("What is the national capital of Italy?", "Name Italy's capital city.", "Which city is the capital of Italy?"), ("State the seat-of-government capital of Italy.", "For Italy, give the national capital.", "Identify Italy's capital city.")),
    Fact("math-square-15", "mathematics", "15 squared", "225", ("What is 15 squared?", "Calculate 15 squared.", "What is the square of 15?"), ("Evaluate 15 squared.", "Multiply 15 by itself.", "Which number equals 15 squared?")),
    Fact("math-root-169", "mathematics", "square root of 169", "13", ("What is the positive square root of 169?", "Calculate the principal square root of 169.", "Which positive integer squared equals 169?"), ("Evaluate the square root of 169.", "State 169's principal square root.", "What positive number has square 169?")),
    Fact("math-product-7-8", "mathematics", "7 times 8", "56", ("What is 7 times 8?", "Calculate the product of 7 and 8.", "Multiply 7 by 8."), ("Evaluate 7 times 8.", "Which number is 7 multiplied by 8?", "Give the product 7 times 8.")),
    Fact("math-gcd", "mathematics", "gcd of 48 and 18", "6", ("What is the greatest common divisor of 48 and 18?", "Calculate gcd(48, 18).", "Give the greatest positive integer dividing both 48 and 18."), ("Evaluate the gcd of 48 and 18.", "What greatest common factor do 48 and 18 share?", "State gcd(48, 18).")),
    Fact("math-mod", "mathematics", "29 modulo 6", "5", ("What is 29 modulo 6?", "Calculate the remainder when 29 is divided by 6.", "Evaluate 29 mod 6."), ("State 29 modulo 6.", "Which remainder results from 29 divided by 6?", "Give 29 mod 6.")),
    Fact("math-power", "mathematics", "2 to the eighth power", "256", ("What is 2 to the eighth power?", "Calculate 2^8.", "Raise 2 to power 8."), ("Evaluate 2 to the eighth power.", "Which number equals 2^8?", "Give the eighth power of 2.")),
    Fact("python-len-four", "python", "len([1, 2, 3, 4])", "4", ("In Python, what does len([1, 2, 3, 4]) return?", "Evaluate the Python expression len([1, 2, 3, 4]).", "What integer is produced by Python's len([1, 2, 3, 4])?"), ("Give the output of Python code len([1, 2, 3, 4]).", "Under Python, evaluate len([1, 2, 3, 4]).", "State the result of len([1, 2, 3, 4]) in Python.")),
    Fact("python-empty-bool", "python", "bool([])", "False", ("In Python, what does bool([]) return?", "Evaluate the Python expression bool([]).", "What Boolean value is produced by Python's bool([])?"), ("Give the output of Python code bool([]).", "Under Python, evaluate bool([]).", "State the result of bool([]) in Python.")),
    Fact("python-range-stop", "python", "list(range(5))", "[0, 1, 2, 3, 4]", ("In Python, what does list(range(5)) return?", "Evaluate the Python expression list(range(5)).", "What list is produced by Python's list(range(5))?"), ("Give the output of Python code list(range(5)).", "Under Python, evaluate list(range(5)).", "State the result of list(range(5)) in Python.")),
    Fact("python-append-return", "python", "list.append return value", "None", ("In Python, what value does list.append return?", "What is the return value of Python's list.append method?", "A Python list append succeeds; which value is returned?"), ("State the return value from list.append in Python.", "Which value does Python's list.append produce as its return?", "Give the Python list.append return value.")),
    Fact("python-int-type", "python", "type(7).__name__", "int", ("In Python, what does type(7).__name__ return?", "Evaluate the Python expression type(7).__name__.", "What string is produced by Python's type(7).__name__?"), ("Give the output of Python code type(7).__name__.", "Under Python, evaluate type(7).__name__.", "State the result of type(7).__name__ in Python.")),
    Fact("python-floor-division", "python", "17 // 5", "3", ("In Python, what does 17 // 5 return?", "Evaluate the Python expression 17 // 5.", "What integer is produced by Python's 17 // 5?"), ("Give the output of Python code 17 // 5.", "Under Python, evaluate 17 // 5.", "State the result of 17 // 5 in Python.")),
)


SYSTEM_PROMPT = (
    "Return exactly one JSON object with exactly two string fields: answer and domain. "
    "The answer must be the shortest direct answer to the question. The domain must be "
    "one lowercase English word that you choose yourself to name the specific academic "
    "or technical field needed to answer. For a programming-language question, use the "
    "language name as the domain. Do not use a list of choices and do not explain."
)
