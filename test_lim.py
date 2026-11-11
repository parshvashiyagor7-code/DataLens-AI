import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from google.genai import errors
from google import genai
from google.genai import types

from app import load_user_dataset, profile_data, TOOL_REGISTRY


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "gemini-3.8-flash"


# ============================================================
# 1. LOAD USER DATASET
# ============================================================

path = input("\nEnter CSV/Excel file or folder path: ").strip()

tables = load_user_dataset(path)

print("\n================ DATASET LOADED ================\n")
print("Tables:", list(tables.keys()))


# ============================================================
# 2. PROFILE ALL TABLES
# ============================================================

profiles = {}

for table_name, table in tables.items():
    profiles[table_name] = profile_data(table)

    print(f"\n================ {table_name} PROFILE ================\n")
    print(profiles[table_name])


# ============================================================
# 3. USER QUESTION
# ============================================================

question = input("\nAsk your data question: ").strip()


# ============================================================
# 4. SEMANTIC INTENT EXTRACTION
# ============================================================

def extract_semantic_intent(question, profiles):
    prompt = f"""
You are the semantic intent parser for a data analyst system.

Your ONLY job is to convert the ORIGINAL USER QUESTION into the
semantic fields below.

The ORIGINAL USER QUESTION is the ONLY source of truth.
Dataset examples and profile values are NOT requirements.
Never copy a value from an example unless it is actually present
in the ORIGINAL USER QUESTION.

Do NOT calculate anything.
Do NOT execute anything.
Do NOT create a plan.
Do NOT output STEP or TOOL.
Do NOT invent filters, limits, rankings, entities, metrics, or values.

DATASET PROFILES:
{profiles}

============================================================
ORIGINAL USER QUESTION
============================================================
{question}

============================================================
OUTPUT FORMAT
============================================================

ENTITY=<entity or none>
ENTITY_ROLE=<list/group/metric/none>
FILTER=<raw-record condition>
FILTER=<raw-record condition>
METRIC=<actual column or none>
AGGREGATION=<sum/mean/min/max/count/none>
RANK=<highest/lowest/none>
GROUP_BY=<entity/category or none>
HAVING=<aggregated condition or none>
LIMIT=<integer or none>
RESULT_TYPE=<list/single/grouped/ranked/grouped_filtered>

Zero or more HAVING lines.
Use one HAVING line for each independent aggregate condition.
Exactly one LIMIT line.
If no HAVING: HAVING=none
If no LIMIT: LIMIT=none

============================================================
DECISION RULES
============================================================

1. ENTITY
Use the entity explicitly asked about.
client/customer -> Client
region -> Region
category -> Category
product -> Product

2. LIST VS GROUP
Use RESULT_TYPE=list when the user wants matching raw records/entities
and does not request aggregation, comparison, ranking, top N, or bottom N.
For LIST queries:
ENTITY_ROLE=list
GROUP_BY=none
METRIC=none
AGGREGATION=none
RANK=none
HAVING=none
LIMIT=none
RESULT_TYPE=list

Use GROUP queries when the user asks for totals, averages, counts,
minimums, maximums, comparisons, rankings, top N, or bottom N.

3. FILTER
FILTER applies to individual/raw records before aggregation.
Examples:
Gold clients -> FILTER=Client_Type=Gold
from North -> FILTER=Region=North
individual purchases above 3000 -> FILTER=Purchase_Value>3000

4. HAVING
HAVING applies after aggregation.
Examples:
total purchase value above 3000 -> AGGREGATION=sum + HAVING=Purchase_Value>3000
average purchase value below 3000 -> AGGREGATION=mean + HAVING=Purchase_Value<3000
more than 5000 in total revenue -> AGGREGATION=sum + HAVING=<revenue metric>5000
above 5000 in total sales -> AGGREGATION=sum + HAVING=<sales metric>5000

Phrases such as "more than X in total revenue", "above X total sales",
"over X in total amount", and "more than X total revenue" describe an
AGGREGATED threshold, so they belong in HAVING, not FILTER.

If a threshold applies to an INDIVIDUAL purchase/order/transaction,
it is FILTER, not HAVING.

5. METRIC / AGGREGATION
average -> mean

total -> sum

highest individual value -> max
lowest individual value -> min

most/fewest purchases, orders, transactions -> count

6. RANK
highest/best/top -> highest
lowest/worst/bottom -> lowest
otherwise -> none

7. GROUP_BY
Use the entity/category being compared or aggregated.
"Which client..." with aggregation/ranking -> GROUP_BY=Client
"Which region..." with aggregation/ranking -> GROUP_BY=Region
"Which category..." with aggregation/ranking -> GROUP_BY=Category

Pure raw-record LIST questions use GROUP_BY=none.

8. TOP/BOTTOM N
Only use LIMIT when the ORIGINAL USER QUESTION explicitly says
top N or bottom N.
"top 3" -> RANK=highest, LIMIT=3
"bottom 2" -> RANK=lowest, LIMIT=2
Otherwise LIMIT=none.

9. IMPORTANT VALUE DISTINCTIONS
"highest purchase value" -> max
"highest total purchase value" -> sum + highest
"highest average purchase value" -> mean + highest
"most purchases" -> count + highest

10. NEVER INVENT
Do not copy Gold, North, 3000, 5000, 2, 3, or anything else
from any example. Only use information present in the ORIGINAL USER QUESTION.

============================================================
MINIMAL EXAMPLES
============================================================

Example A
Question: Which Gold clients have individual purchases above 3000?
ENTITY=Client
ENTITY_ROLE=list
FILTER=Client_Type=Gold
FILTER=Purchase_Value>3000
METRIC=none
AGGREGATION=none
RANK=none
GROUP_BY=none
HAVING=none
LIMIT=none
RESULT_TYPE=list

Example B
Question: Which clients have a total purchase value above 3000?
ENTITY=Client
ENTITY_ROLE=group
FILTER=none
METRIC=Purchase_Value
AGGREGATION=sum
RANK=none
GROUP_BY=Client
HAVING=Purchase_Value>3000
LIMIT=none
RESULT_TYPE=grouped_filtered

Example C
Question: Show the bottom 2 clients by total purchase value.
ENTITY=Client
ENTITY_ROLE=group
FILTER=none
METRIC=Purchase_Value
AGGREGATION=sum
RANK=lowest
GROUP_BY=Client
HAVING=none
LIMIT=2
RESULT_TYPE=ranked

============================================================
FINAL CHECK
============================================================

- Every explicit raw-record condition -> FILTER.
- Every aggregated threshold -> HAVING.
- No unrelated filters.
- No invented LIMIT.
- No invented RANK.
- No unnecessary GROUP_BY.
- No unnecessary aggregation.
- If LIST query: GROUP_BY=none and AGGREGATION=none.

Return ONLY the semantic fields.
"""
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. "
            "Configure it in Streamlit Secrets."
        )

    client = genai.Client(api_key=api_key)

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=300,
                ),
            )

            if not response.text:
                raise RuntimeError(
                    "Gemini returned an empty semantic response."
                )

            return response.text.strip()

        except errors.ServerError as e:
            if e.code == 503 and attempt < 2:
                print(f"Gemini overloaded. Retrying... ({attempt + 1}/3)")
                time.sleep(5 * (attempt + 1))
            else:
                raise

semantic_intent = extract_semantic_intent(question, profiles)

print("\n================ SEMANTIC INTENT ================\n")
print(semantic_intent)


# ============================================================
# 5. VALIDATE SEMANTIC INTENT AGAINST ORIGINAL QUESTION
# ============================================================

def validate_semantic_intent_against_question(semantic_intent, question):
    """Validate LLM semantics against the original question.

    The model is an interpreter, not the final authority. This validator:
    - removes invented filters
    - keeps raw-record conditions in FILTER
    - moves aggregate thresholds to HAVING
    - supports multiple independent HAVING conditions
    - supports COUNT + another aggregate in the same grouped query
    - distinguishes grouped rankings from raw-record top/bottom queries
    - deterministically derives RANK/LIMIT from the original question
    - derives GROUP_BY from ENTITY for aggregate/ranking questions
    """
    question_lower = question.lower().strip()
    lines = [line.strip() for line in semantic_intent.splitlines() if line.strip()]

    scalar = {
        "ENTITY": "none",
        "ENTITY_ROLE": "none",
        "METRIC": "none",
        "AGGREGATION": "none",
        "GROUP_BY": "none",
        "RESULT_TYPE": "grouped",
    }

    for line in lines:
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().upper()
        value = value.strip()
        if key in scalar:
            scalar[key] = value

    # ------------------------------------------------------------
    # Deterministically resolve the TARGET entity from the original
    # question. This prevents the LLM from confusing a filter entity
    # (e.g. Premium customers) with the entity being ranked/grouped
    # (e.g. category).
    # ------------------------------------------------------------
    def detect_target_entity(question_text):
        entity_patterns = [
            (r"\b(?:which|what)\s+(?:\w+\s+){0,4}(?:category|categories)\b", "Category"),
            (r"\b(?:which|what)\s+(?:\w+\s+){0,4}products?\b", "Product"),
            (r"\b(?:which|what)\s+(?:\w+\s+){0,4}customers?\b", "Customer"),
            (r"\b(?:which|what)\s+(?:\w+\s+){0,4}clients?\b", "Client"),
            (r"\b(?:which|what)\s+(?:\w+\s+){0,4}regions?\b", "Region"),
            (r"\b(?:which|what)\s+(?:\w+\s+){0,4}(?:city|cities)\b", "City"),
            (r"\b(?:by|per|group(?:ed)?\s+by)\s+(?:category|categories)\b", "Category"),
            (r"\b(?:by|per|group(?:ed)?\s+by)\s+products?\b", "Product"),
            (r"\b(?:by|per|group(?:ed)?\s+by)\s+(?:each|every\s+)?customers?\b", "Customer"),
            (r"\b(?:by|per|group(?:ed)?\s+by)\s+clients?\b", "Client"),
            (r"\b(?:by|per|group(?:ed)?\s+by)\s+regions?\b", "Region"),
            (r"\b(?:by|per|group(?:ed)?\s+by)\s+(?:city|cities)\b", "City"),
        ]

        for pattern, entity_name in entity_patterns:
            if re.search(pattern, question_text):
                return entity_name

        return None

    target_entity = detect_target_entity(question_lower)
    if target_entity:
        scalar["ENTITY"] = target_entity
        scalar["GROUP_BY"] = target_entity

    metric = None if scalar["METRIC"].lower() == "none" else scalar["METRIC"]
    llm_aggregation = scalar["AGGREGATION"].lower()
    aggregation = None if llm_aggregation == "none" else llm_aggregation

    def normalize_value(value):
        return str(value).strip().strip("'\"")

    def value_explicitly_present(value):
        value = normalize_value(value)
        if value.lower() in {"none", "null", ""}:
            return False
        if re.fullmatch(r"\d+(?:\.\d+)?", value):
            return re.search(
                rf"(?<!\d){re.escape(value)}(?!\d)",
                question_lower,
            ) is not None
        return value.lower() in question_lower

    def parse_condition(condition):
        condition = condition.strip()

        null_match = re.match(
            r"^(.+?)\s+(is\s+not\s+null|is\s+null|not\s+null)$",
            condition,
            re.IGNORECASE,
        )
        if null_match:
            column = null_match.group(1).strip()
            phrase = re.sub(r"\s+", " ", null_match.group(2).strip().lower())
            if phrase in {"is null"}:
                return column, "is null", None
            return column, "is not null", None

        # Normalize LLM-style NULL comparisons such as:
        #   City=None / City=null     -> IS NULL
        #   City!=None / City!=null  -> IS NOT NULL
        null_compare = re.match(
            r"^(.+?)\s*(==|!=|=)\s*(none|null|nan|n/a)$",
            condition,
            re.IGNORECASE,
        )
        if null_compare:
            column = null_compare.group(1).strip()
            operator = null_compare.group(2).strip()
            return column, ("is null" if operator in {"=", "=="} else "is not null"), None

        match = re.match(
            r"^(.+?)\s*(>=|<=|==|!=|>|<|=)\s*(.+)$",
            condition,
        )
        if not match:
            return None

        column = match.group(1).strip()
        operator = match.group(2).strip()
        value = normalize_value(match.group(3))

        if operator == "=":
            operator = "=="

        return column, operator, value

    explicit_havings = []
    raw_filters = []

    # ------------------------------------------------------------
    # Deterministic missing / NULL condition detection.
    # Examples:
    #   "customers with missing city values"
    #   "rows where city is null"
    #   "customers whose city is not null"
    # ------------------------------------------------------------
    null_conditions = []

    null_patterns = [
        (r"\b(?:missing|null|blank|empty)\s+(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b", "is null"),
        (r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:is|are)\s+(?:missing|null|blank|empty)\b", "is null"),
        (r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+is\s+not\s+null\b", "is not null"),
        (r"\b(?:non[- ]?missing|non[- ]?null|not[- ]?missing|not[- ]?null)\s+(?:the\s+)?([a-zA-Z_][a-zA-Z0-9_]*)\b", "is not null"),
        (r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s+(?:is|are)\s+(?:non[- ]?missing|non[- ]?null|not[- ]?missing|not[- ]?null)\b", "is not null"),
    ]

    for pattern, operator in null_patterns:
        for null_match in re.finditer(pattern, question_lower):
            semantic_column = null_match.group(1).strip()
            if semantic_column not in {"values", "value", "records", "rows", "data"}:
                null_conditions.append((semantic_column, operator, None))

    # Normalize NULL semantics from the ORIGINAL QUESTION.
    # Negative forms such as "non-missing city", "not missing city",
    # and "city is not null" must never coexist with a positive IS NULL
    # condition for the same column. Likewise, an explicit missing/null
    # request must not retain an IS NOT NULL condition produced by the LLM.
    null_positive_cols = set()
    null_negative_cols = set()

    for col, op, _ in null_conditions:
        col_key = str(col).strip().lower()
        if op == "is null":
            null_positive_cols.add(col_key)
        elif op == "is not null":
            null_negative_cols.add(col_key)

    # Remove contradictory null conditions already collected.
    if null_negative_cols:
        null_conditions = [
            item for item in null_conditions
            if not (
                str(item[0]).strip().lower() in null_negative_cols
                and item[1] == "is null"
            )
        ]
    if null_positive_cols:
        null_conditions = [
            item for item in null_conditions
            if not (
                str(item[0]).strip().lower() in null_positive_cols
                and item[1] == "is not null"
            )
        ]

    # De-duplicate NULL conditions while preserving order.
    null_conditions = list(dict.fromkeys(null_conditions))

    # Rebuild only canonical NULL filters from the normalized conditions.
    raw_filters = [
        item for item in raw_filters
        if item[1] not in {"is null", "is not null"}
    ]
    raw_filters.extend(null_conditions)

    # ------------------------------------------------------------
    # Deterministic amount/value threshold detection.
    # Handles natural phrasing such as:
    # - orders worth more than 2500
    # - purchases valued over 3000
    # - transactions above 5000
    # These are raw-record amount filters unless an aggregate keyword
    # makes them explicitly aggregated.
    # ------------------------------------------------------------
    amount_threshold = re.search(
        r"\b(?:orders?|purchases?|transactions?|sales?|items?)\s+"
        r"(?:that\s+)?(?:are\s+)?"
        r"(?:worth|valued\s+at|valued)\s+"
        r"(more\s+than|greater\s+than|above|over|at\s+least|at\s+most|"
        r"less\s+than|under|below)\s+(\d+(?:\.\d+)?)\b",
        question_lower,
    )

    if amount_threshold:
        phrase = amount_threshold.group(1).lower()
        number = amount_threshold.group(2)

        if phrase in {"more than", "greater than", "above", "over"}:
            amount_operator = ">"
        elif phrase == "at least":
            amount_operator = ">="
        elif phrase == "at most":
            amount_operator = "<="
        else:
            amount_operator = "<"

        raw_filters.append(("__AMOUNT__", amount_operator, number))

    # Read LLM HAVING/FILTER candidates.
    for line in lines:
        upper = line.upper()

        if upper.startswith("HAVING="):
            condition = line.split("=", 1)[1].strip()
            if condition.lower() != "none":
                parsed = parse_condition(condition)
                if parsed and value_explicitly_present(parsed[2]):
                    explicit_havings.append(parsed)

        elif upper.startswith("FILTER="):
            condition = line.split("=", 1)[1].strip()
            if condition.lower() == "none":
                continue

            parsed = parse_condition(condition)
            if not parsed:
                continue

            column, operator, value = parsed
            if operator not in {"is null", "is not null"} and not value_explicitly_present(value):
                continue

            raw_filters.append((column, operator, value))

    # ------------------------------------------------------------
    # Deterministic raw order/purchase amount threshold detection.
    # Phrases such as:
    #   "customers have an order amount greater than 3000"
    #   "orders with amount above 5000"
    # describe INDIVIDUAL records unless an explicit aggregate
    # modifier (total/average/mean/sum) is present.
    # ------------------------------------------------------------
    raw_value_threshold_match = re.search(
        r"\b(?:an?\s+)?(?:order|orders|purchase|purchases|transaction|transactions)\s+"
        r"(?:amount|value)\s*"
        r"(?:is\s+)?"
        r"(at least|at most|more than|greater than|above|over|less than|below|under|exactly)\s+"
        r"(\d+(?:\.\d+)?)\b",
        question_lower,
    )

    if raw_value_threshold_match and not re.search(
        r"\b(total|average|avg|mean|sum|spent|spend)\b",
        question_lower,
    ):
        phrase = raw_value_threshold_match.group(1).lower()
        number = raw_value_threshold_match.group(2)
        if phrase in {"more than", "greater than", "above", "over"}:
            raw_amount_operator = ">"
        elif phrase == "at least":
            raw_amount_operator = ">="
        elif phrase == "at most":
            raw_amount_operator = "<="
        elif phrase == "exactly":
            raw_amount_operator = "=="
        else:
            raw_amount_operator = "<"

        raw_filters.append(("__AMOUNT__", raw_amount_operator, number))

        # An LLM may misclassify the same raw-record threshold as HAVING.
        explicit_havings = [
            having
            for having in explicit_havings
            if str(having[0]).strip().lower()
            not in {"order_amount", "amount", "value", "purchase_value"}
        ]

    # ------------------------------------------------------------
    # Deterministic ranking and LIMIT from ORIGINAL QUESTION.
    # ------------------------------------------------------------
    limit_match = re.search(r"\b(top|bottom)\s+(\d+)\b", question_lower)
    limit = int(limit_match.group(2)) if limit_match else None

    if re.search(r"\bbottom\s+\d+\b", question_lower):
        rank = "lowest"
    elif re.search(r"\btop\s+\d+\b", question_lower):
        rank = "highest"
    elif re.search(r"\b(highest|best)\b", question_lower):
        rank = "highest"
    elif re.search(r"\b(lowest|worst)\b", question_lower):
        rank = "lowest"
    elif re.search(r"\bmost\b", question_lower) and not re.search(
        r"\bat most\b", question_lower
    ):
        rank = "highest"
    elif (
        re.search(r"\bfewest\b", question_lower)
        or (
            re.search(r"\bleast\b", question_lower)
            and not re.search(r"\bat least\b", question_lower)
        )
    ):
        rank = "lowest"
    else:
        rank = "none"

    # ------------------------------------------------------------
    # RAW top/bottom detection.
    # ------------------------------------------------------------
    raw_top_bottom = bool(
        re.search(
            r"\b(?:top|bottom)\s+\d+\s+"
            r"(?:gold\s+|silver\s+|standard\s+|premium\s+|north\s+|south\s+|east\s+|west\s+)*"
            r"(?:purchases?|orders?|transactions?|records?|items?|rows?)\b",
            question_lower,
        )
    )

    explicit_raw_records = bool(
        re.search(
            r"\b(individual|each|raw)\s+"
            r"(purchase|purchases|order|orders|transaction|transactions|record|records)\b",
            question_lower,
        )
    )

    # Count-threshold wording, e.g. "more than 2 orders".
    count_threshold = re.search(
        r"\b(at least|at most|more than|greater than|above|over|"
        r"fewer than|less than|below|under|exactly)\s+(\d+)\s+"
        r"(?:times\b|(?:purchases?|orders?|transactions?|records?|events?|items?)\b)",
        question_lower,
    )

    # Do not let a COUNT phrase become a fake numeric filter such as
    # Quantity > 2 unless the question explicitly talks about quantity.
    if count_threshold and not re.search(
        r"\b(quantity|quantities|unit|units)\b",
        question_lower,
    ):
        raw_filters = [
            item
            for item in raw_filters
            if not re.search(
                r"\b(quantity|qty|units?)\b",
                item[0],
                re.IGNORECASE,
            )
        ]

    # ------------------------------------------------------------
    # Reject hallucinated ID filters in COUNT-threshold questions.
    #
    # Example:
    #   "Which customers placed at least 2 orders?"
    #
    # The LLM may invent:
    #   FILTER=Customer_ID>2
    #
    # The "2" is the COUNT threshold, not a Customer_ID filter.
    # Keep an ID filter only when the user explicitly refers to an
    # identifier/code/ID/number for that entity.
    # ------------------------------------------------------------
    if count_threshold:
        explicit_identifier_reference = bool(
            re.search(
                r"\b(?:customer|client|product|order|purchase|"
                r"transaction|record|sale)\s*[_ -]?"
                r"(?:id|identifier|code|number)\b",
                question_lower,
            )
        ) or bool(
            re.search(
                r"\b(?:id|identifier|code)\b",
                question_lower,
            )
        )

        if not explicit_identifier_reference:
            raw_filters = [
                item
                for item in raw_filters
                if not str(item[0]).strip().lower().endswith("_id")
                and str(item[0]).strip().lower() not in {
                    "id",
                    "identifier",
                    "code",
                }
            ]

    count_language = bool(
        re.search(
            r"\b(most|fewest|how many|number of)\s+"
            r"(purchases?|orders?|transactions?|records?|events?|items?)\b",
            question_lower,
        )
    )

    aggregate_language = bool(
        re.search(
            r"\b(total|average|mean|highest|lowest|maximum|minimum|"
            r"most|fewest|count|compare|comparison|spent|spend)\b",
            question_lower,
        )
    )

    # ------------------------------------------------------------
    # Deterministic aggregate-threshold detection.
    # Handles natural-language phrases such as:
    #   "more than 5000 in total revenue"
    #   "above 5000 in total sales"
    #   "over 5000 in total amount"
    # These are post-GROUP conditions, so they must become HAVING.
    # ------------------------------------------------------------
    aggregate_threshold = re.search(
        r"\b(at least|at most|more than|greater than|above|over|"
        r"less than|below|under|exactly)\s+(\d+(?:\.\d+)?)\s+"
        r"(?:in\s+)?(?:the\s+)?(?:total\s+)?"
        r"(revenue|sales|amount|order\s+amount|purchase\s+value|value)\b",
        question_lower,
    )

    reverse_aggregate_threshold = re.search(
        r"\b(?:total\s+)?(revenue|sales|amount|order\s+amount|"
        r"purchase\s+value|value)\s+"
        r"(at least|at most|more than|greater than|above|over|"
        r"less than|below|under|exactly)\s+(\d+(?:\.\d+)?)\b",
        question_lower,
    )

    def threshold_operator(phrase):
        phrase = phrase.lower()
        if phrase == "at least":
            return ">="
        if phrase == "at most":
            return "<="
        if phrase in {"more than", "greater than", "above", "over"}:
            return ">"
        if phrase in {"less than", "below", "under"}:
            return "<"
        return "=="

    # Also support: "average order amount above 3000" and similar
    # modifier-first aggregate threshold phrasing.
    modifier_first_threshold = re.search(
        r"\b(average|avg|mean|total|sum)\s+"
        r"(?:(?:order|purchase)\s+)?"
        r"(amount|value|revenue|sales)\s+"
        r"(at least|at most|more than|greater than|above|over|less than|below|under|exactly)\s+"
        r"(\d+(?:\.\d+)?)\b",
        question_lower,
    )

    aggregate_threshold_condition = None

    if aggregate_threshold:
        aggregate_threshold_condition = (
            aggregate_threshold.group(1),
            aggregate_threshold.group(2),
            aggregate_threshold.group(3),
        )
    elif reverse_aggregate_threshold:
        aggregate_threshold_condition = (
            reverse_aggregate_threshold.group(2),
            reverse_aggregate_threshold.group(3),
            reverse_aggregate_threshold.group(1),
        )
    elif modifier_first_threshold:
        aggregate_threshold_condition = (
            modifier_first_threshold.group(3),
            modifier_first_threshold.group(4),
            modifier_first_threshold.group(2),
        )


    is_list_query = (
        raw_top_bottom
        or explicit_raw_records
        or (
            raw_value_threshold_match is not None
            and not re.search(
                r"\b(total|average|avg|mean|sum|spent|spend|count|most|fewest|highest|lowest|maximum|minimum|compare|comparison)\b",
                question_lower,
            )
            and count_threshold is None
            and rank == "none"
            and limit is None
        )
        or (
            not aggregate_language
            and limit is None
            and rank == "none"
            and count_threshold is None
        )
    )

    if count_language and not raw_top_bottom:
        is_list_query = False

    if count_threshold is not None and not raw_top_bottom:
        is_list_query = False

    if aggregate_threshold_condition and not raw_top_bottom:
        is_list_query = False

    # ------------------------------------------------------------
    # Correct aggregation from the question.
    # COUNT thresholds can coexist with SUM/MEAN/etc.
    # ------------------------------------------------------------
    aggregation_out = aggregation

    # ------------------------------------------------------------
    # Deterministic aggregate intent recovery from ORIGINAL QUESTION.
    # The LLM can leave METRIC=none / choose sum incorrectly. Resolve
    # phrases such as:
    #   "average order amount above 3000"
    #   "mean sales below 5000"
    #   "total revenue above 10000"
    # before final semantic normalization.
    # ------------------------------------------------------------
    aggregate_metric_patterns = [
        (r"\b(?:average|avg|mean)\s+(?:order\s+|purchase\s+)?(amount|value|revenue|sales)\b", "mean"),
        (r"\b(?:total|sum)\s+(?:order\s+|purchase\s+)?(amount|value|revenue|sales)\b", "sum"),
        (r"\baverage\s+order\s+amount\b", "mean"),
        (r"\bmean\s+order\s+amount\b", "mean"),
        (r"\btotal\s+order\s+amount\b", "sum"),
    ]

    question_metric_phrase = None
    question_metric_aggregation = None
    for pattern, agg_name in aggregate_metric_patterns:
        match = re.search(pattern, question_lower)
        if match:
            question_metric_phrase = match.group(0)
            question_metric_aggregation = agg_name
            break

    if question_metric_aggregation:
        aggregation_out = question_metric_aggregation

    if (not metric or str(metric).lower() == "none") and question_metric_phrase:
        # Keep this semantic; schema resolution later maps it to the
        # actual dataset column.
        if re.search(r"\b(order\s+amount|amount)\b", question_metric_phrase):
            metric = "Order_Amount"
        elif re.search(r"\b(revenue|sales)\b", question_metric_phrase):
            metric = "Revenue"
        elif re.search(r"\b(purchase\s+value|value)\b", question_metric_phrase):
            metric = "Purchase_Value"

    if metric and str(metric).lower() != "none":
        scalar["METRIC"] = metric

    if count_threshold:
        # A phrase such as "exactly 2 orders" is a COUNT requirement.
        # Only keep another aggregate when the ORIGINAL QUESTION explicitly
        # asks for that aggregate as well.
        explicit_non_count_aggregate = (
            "total" in question_lower
            or re.search(
                r"\b(spent|spend|revenue|sales)\b",
                question_lower,
            )
            or "average" in question_lower
            or re.search(r"\bmean\b", question_lower)
            or "maximum" in question_lower
            or re.search(r"\bhighest value\b", question_lower)
            or "minimum" in question_lower
            or re.search(r"\blowest value\b", question_lower)
        )

        if explicit_non_count_aggregate:
            if aggregation in {"sum", "mean", "min", "max"}:
                aggregation_out = aggregation
            elif "total" in question_lower or re.search(
                r"\b(spent|spend|revenue|sales)\b",
                question_lower,
            ):
                aggregation_out = "sum"
            elif (
                "average" in question_lower
                or re.search(r"\bmean\b", question_lower)
            ):
                aggregation_out = "mean"
            elif (
                "maximum" in question_lower
                or re.search(r"\bhighest value\b", question_lower)
            ):
                aggregation_out = "max"
            elif (
                "minimum" in question_lower
                or re.search(r"\blowest value\b", question_lower)
            ):
                aggregation_out = "min"
            else:
                aggregation_out = "count"
        else:
            aggregation_out = "count"

    elif count_language:
        aggregation_out = "count"
    elif "total" in question_lower:
        aggregation_out = "sum"
    elif "average" in question_lower or re.search(r"\bmean\b", question_lower):
        aggregation_out = "mean"
    elif "maximum" in question_lower or re.search(
        r"\bhighest value\b",
        question_lower,
    ):
        aggregation_out = "max"
    elif "minimum" in question_lower or re.search(
        r"\blowest value\b",
        question_lower,
    ):
        aggregation_out = "min"


    # ------------------------------------------------------------
    # Count threshold becomes a grouped HAVING condition.
    # ------------------------------------------------------------
    if count_threshold:
        phrase = count_threshold.group(1).lower()
        number = count_threshold.group(2)

        if phrase == "at least":
            op = ">="
        elif phrase == "at most":
            op = "<="
        elif phrase in {"more than", "greater than", "above", "over"}:
            op = ">"
        elif phrase in {"fewer than", "less than", "below", "under"}:
            op = "<"
        else:
            op = "=="

        count_having = ("__COUNT__", op, number)

        if count_having not in explicit_havings:
            explicit_havings.append(count_having)

        # Pure COUNT-threshold questions must not inherit unrelated LLM
        # HAVING conditions such as Quantity==2. Preserve other HAVING
        # conditions only when the ORIGINAL QUESTION explicitly contains
        # another aggregate/value threshold.
        has_non_count_threshold = bool(
            re.search(
                r"\b(total|average|mean|spent|spend|revenue|sales|"
                r"amount|value|highest|lowest|maximum|minimum)\b",
                question_lower,
            )
        )

        if not has_non_count_threshold:
            explicit_havings = [
                having
                for having in explicit_havings
                if having[0] == "__COUNT__"
            ]

    # ------------------------------------------------------------
    # FINAL QUESTION-AUTHORITY OVERRIDE
    # ------------------------------------------------------------
    # The COUNT-threshold block above may preserve the LLM's aggregation
    # (for example SUM) even when the original question explicitly says
    # "average". The original question must win. Re-apply the detected
    # aggregate intent here, after COUNT handling.
    if question_metric_aggregation:
        aggregation_out = question_metric_aggregation

    if question_metric_phrase:
        if re.search(r"\b(order\s+amount|amount)\b", question_metric_phrase):
            metric = "Order_Amount"
        elif re.search(r"\b(revenue|sales)\b", question_metric_phrase):
            metric = "Revenue"
        elif re.search(r"\b(purchase\s+value|value)\b", question_metric_phrase):
            metric = "Purchase_Value"

        scalar["METRIC"] = metric

    # ------------------------------------------------------------
    # Final semantic normalization.
    # ------------------------------------------------------------
    if raw_value_threshold_match is not None and not re.search(
        r"\b(total|average|avg|mean|sum|spent|spend|count|most|fewest|highest|lowest|maximum|minimum|compare|comparison)\b",
        question_lower,
    ) and count_threshold is None and rank == "none" and limit is None:
        is_list_query = True

    if is_list_query:
        entity_role = "list"
        group_by = "none"
        final_rank = rank if raw_top_bottom else "none"
        final_limit = limit if raw_top_bottom else None
        metric_out = metric if raw_top_bottom and metric else "none"
        aggregation_out = None
        final_havings = []

        # A HAVING condition on a raw list is really a raw FILTER.
        for having in explicit_havings:
            raw_filters.append(having)

        result_type = "list"

    else:
        entity = scalar["ENTITY"].strip()
        entity_role = scalar["ENTITY_ROLE"].lower()
        metric_out = metric or scalar["METRIC"]
        group_by = scalar["GROUP_BY"].strip()
        final_rank = rank
        final_limit = limit
        final_havings = explicit_havings.copy()

        if count_threshold:
            if aggregation_out == "count":
                # COUNT threshold is the primary operation for pure
                # "N orders/purchases" questions.
                metric_out = "__COUNT__"

            if group_by.lower() == "none" and entity.lower() != "none":
                group_by = entity

            entity_role = "group"
            result_type = "grouped_filtered"

        # Aggregate/ranked entity queries must have a grouping key.
        if group_by.lower() == "none" and entity.lower() != "none" and (
            aggregation_out is not None
            or final_rank in {"highest", "lowest"}
        ):
            group_by = entity

        # Deterministically create HAVING for aggregate-threshold wording.
        if aggregate_threshold_condition and metric_out and metric_out.lower() != "none":
            phrase, threshold_value, _metric_phrase = aggregate_threshold_condition
            condition = (
                metric_out,
                threshold_operator(phrase),
                threshold_value,
            )
            if condition not in final_havings:
                final_havings.append(condition)

        # Convert an LLM FILTER on the aggregate metric into HAVING.
        # This is useful when the model emits Order_Amount > 9000 as FILTER
        # instead of HAVING for wording such as "spent over 9000".
        aggregate_words = re.search(
            r"\b(total|average|mean|count|maximum|minimum|highest|lowest|"
            r"spent|spend|revenue|sales)\b",
            question_lower,
        )

        if (
            metric_out
            and metric_out.lower() != "none"
            and aggregation_out in {"sum", "mean", "min", "max"}
            and aggregate_words
        ):
            remaining_filters = []

            for candidate in raw_filters:
                column, operator, value = candidate

                normalized_filter_column = str(column).strip().lower()
                aggregate_metric_alias = normalized_filter_column in {
                    "revenue", "sales", "amount", "order_amount",
                    "purchase_value", "value", "__amount__"
                }

                if (
                    normalized_filter_column == metric_out.lower()
                    or aggregate_metric_alias
                ):
                    if candidate not in final_havings:
                        final_havings.append(candidate)
                    continue

                remaining_filters.append(candidate)

            raw_filters = remaining_filters

        # Keep COUNT threshold + aggregate threshold together.
        if final_havings:
            result_type = "grouped_filtered"
        elif final_rank in {"highest", "lowest"}:
            result_type = "ranked"
        elif aggregation_out is not None:
            result_type = "grouped"
        else:
            result_type = "grouped"

    output = [
        f"ENTITY={scalar.get('ENTITY', 'none')}",
        f"ENTITY_ROLE={entity_role}",
    ]

    if raw_filters:
        for column, operator, value in raw_filters:
            if operator in {"is null", "is not null"}:
                output.append(f"FILTER={column} {operator}")
            else:
                output.append(f"FILTER={column}{operator}{value}")
    else:
        output.append("FILTER=none")

    output.extend(
        [
            f"METRIC={metric_out or 'none'}",
            f"AGGREGATION={aggregation_out or 'none'}",
            f"RANK={final_rank}",
            f"GROUP_BY={group_by}",
        ]
    )

    if final_havings:
        for column, operator, value in final_havings:
            output.append(f"HAVING={column}{operator}{value}")
    else:
        output.append("HAVING=none")

    output.append(
        f"LIMIT={final_limit if final_limit is not None else 'none'}"
    )
    output.append(f"RESULT_TYPE={result_type}")

    return "\n".join(output)


validated_semantic_intent = validate_semantic_intent_against_question(
    semantic_intent,
    question,
)

print("\n================ VALIDATED SEMANTIC INTENT ================\n")
print(validated_semantic_intent)


# ============================================================
# 6. MAP SEMANTIC INTENT TO REQUIREMENTS
# ============================================================

def resolve_schema_column(name, tables):
    """Resolve a semantic field to an actual existing schema column."""
    if not name or str(name).lower() == "none":
        return None

    name_lower = str(name).strip().lower()

    # Exact match first.
    for df in tables.values():
        for column in df.columns:
            if column.lower() == name_lower:
                return column

    # Entity -> entity ID, e.g. Client -> Client_ID.
    candidate = f"{name_lower}_id"
    for df in tables.values():
        for column in df.columns:
            if column.lower() == candidate:
                return column

    return None


def tables_for_column(column, tables):
    if not column:
        return []

    return [
        table_name
        for table_name, df in tables.items()
        if column in df.columns
    ]


def map_semantic_intent_to_requirements(
    semantic_intent,
    tables,
    question,
):
    requirements = {
        "tables": [],
        "filters": [],
        "having": [],
        "group_column": None,
        "value_column": None,
        "aggregation": None,
        "order": None,
        "limit": None,
        "entity": None,
        "entity_column": None,
        "result_type": None,
        "sort_column": None,
        "additional_aggregations": [],
        "source_column": None,
    }

    lines = semantic_intent.splitlines()

    # ENTITY
    for line in lines:
        if line.upper().startswith("ENTITY="):
            value = line.split("=", 1)[1].strip()
            if value.lower() != "none":
                requirements["entity"] = value

    # RESULT_TYPE
    for line in lines:
        if line.upper().startswith("RESULT_TYPE="):
            value = line.split("=", 1)[1].strip().lower()
            if value in {
                "list",
                "single",
                "grouped",
                "ranked",
                "grouped_filtered",
            }:
                requirements["result_type"] = value

    # GROUP_BY
    for line in lines:
        if not line.upper().startswith("GROUP_BY="):
            continue

        value = line.split("=", 1)[1].strip()

        if value.lower() == "none":
            requirements["group_column"] = None
            continue

        requirements["group_column"] = resolve_schema_column(
            value,
            tables,
        )

    def choose_amount_column_for_question(tables, question):
        preferred_tokens = [
            "order_amount", "purchase_value", "amount", "revenue",
            "sales", "value", "price", "total", "cost", "spend"
        ]

        candidates = []
        for table_name, df in tables.items():
            for column in df.columns:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    continue

                name = str(column).lower()
                score = 0
                for index, token in enumerate(preferred_tokens):
                    if token in name:
                        score = max(score, len(preferred_tokens) - index)

                if score > 0:
                    candidates.append((score, table_name, column))

        if candidates:
            candidates.sort(key=lambda item: (-item[0], item[1]))
            return candidates[0][2]

        return None

    # FILTERS
    for line in lines:
        if not line.upper().startswith("FILTER="):
            continue

        value = line.split("=", 1)[1].strip()
        if value.lower() == "none":
            continue

        null_match = re.match(
            r"^(.+?)\s+(is\s+not\s+null|is\s+null)$",
            value,
            re.IGNORECASE,
        )

        if null_match:
            column_name = null_match.group(1).strip()
            operator = re.sub(r"\s+", " ", null_match.group(2).strip().lower())
            filter_value = None
        else:
            match = re.match(
                r"^(.+?)\s*(>=|<=|==|!=|>|<|=)\s*(.+)$",
                value,
            )

            if not match:
                continue

            column_name = match.group(1).strip()
            operator = match.group(2).strip()
            filter_value = match.group(3).strip().strip("'\"")

            if operator == "=":
                operator = "=="

        if column_name == "__AMOUNT__":
            actual_column = choose_amount_column_for_question(
                tables,
                question,
            )
        else:
            actual_column = resolve_schema_column(
                column_name,
                tables,
            )

        if not actual_column:
            # Keep it so validation reports the issue instead of silently hiding it.
            actual_column = column_name

        requirements["filters"].append({
            "column": actual_column,
            "operator": operator,
            "value": filter_value,
        })

    # HAVING
    for line in lines:
        if not line.upper().startswith("HAVING="):
            continue

        value = line.split("=", 1)[1].strip()
        if value.lower() == "none":
            continue

        match = re.match(
            r"^(.+?)\s*(>=|<=|==|!=|>|<|=)\s*(.+)$",
            value,
        )

        if not match:
            continue

        column_name = match.group(1).strip()
        operator = match.group(2).strip()
        having_value = match.group(3).strip().strip("'\"")

        if operator == "=":
            operator = "=="

        actual_column = resolve_schema_column(
            column_name,
            tables,
        )

        # Map semantic aggregate aliases (e.g. Revenue) to the actual
        # numeric metric column used by this question when the alias is not
        # a literal schema column.
        if not actual_column:
            alias = str(column_name).strip().lower()
            if alias in {
                "revenue", "sales", "amount", "order_amount",
                "purchase_value", "value",
            }:
                # Prefer the question-matching numeric business metric.
                metric_candidates = []
                for table_name, df in tables.items():
                    for candidate in df.columns:
                        if not pd.api.types.is_numeric_dtype(df[candidate]):
                            continue

                        normalized = re.sub(r"[_\-]+", " ", str(candidate)).lower()
                        score = 0
                        if normalized and normalized in question.lower():
                            score += 100

                        candidate_tokens = {
                            token for token in re.findall(r"[a-z0-9]+", normalized)
                            if token != "id"
                        }
                        score += sum(
                            10 for token in candidate_tokens
                            if re.search(rf"\b{re.escape(token)}\b", question.lower())
                        )

                        if alias in {"revenue", "sales", "amount", "value"} and any(
                            token in normalized for token in {"amount", "revenue", "sales", "value"}
                        ):
                            score += 20

                        if score > 0:
                            metric_candidates.append((score, table_name, candidate))

                if metric_candidates:
                    metric_candidates.sort(key=lambda item: (-item[0], item[1]))
                    actual_column = metric_candidates[0][2]

        actual_column = actual_column or column_name

        requirements["having"].append({
            "column": actual_column,
            "operator": operator,
            "value": having_value,
        })

    # METRIC
    metric_name = None

    for line in lines:
        if line.upper().startswith("METRIC="):
            value = line.split("=", 1)[1].strip()
            if value.lower() != "none":
                metric_name = value
            break

    if metric_name:

        metric_clean = metric_name.strip()

        # Remove common aggregation wrappers such as:
        # Average(Purchase_Value)
        # Mean(Purchase_Value)
        # Sum(Purchase_Value)
        # Count(Purchase_ID)
        match = re.match(
            r"^(?:average|avg|mean|sum|total|count|min|minimum|max|maximum)\s*\(\s*(.+?)\s*\)$",
            metric_clean,
            re.IGNORECASE,
        )

        if match:
            metric_clean = match.group(1).strip()

        resolved_metric = resolve_schema_column(
            metric_clean,
            tables,
        )

        # Semantic aliases such as Revenue/Sales may not exist as literal
        # schema columns. When they do not resolve, fall back to the real
        # numeric metric inferred from the ORIGINAL QUESTION.
        if not resolved_metric:
            resolved_metric = next((
                column
                for table_name, df in tables.items()
                for column in df.columns
                if pd.api.types.is_numeric_dtype(df[column])
                and str(column).lower() in {
                    "order_amount", "purchase_value", "amount",
                    "revenue", "sales", "value"
                }
            ), None)

        requirements["value_column"] = resolved_metric or metric_clean
    # AGGREGATION
    for line in lines:
        if line.upper().startswith("AGGREGATION="):
            value = line.split("=", 1)[1].strip().lower()
            if value != "none":
                requirements["aggregation"] = value
            break

    # The ORIGINAL QUESTION is authoritative for the aggregate operation.
    # This corrects common LLM mistakes such as:
    #   "average order amount" -> AGGREGATION=sum
    #   "total revenue"         -> AGGREGATION=mean
    #   "more than 2 orders and average amount above 3000"
    #                             -> mean + COUNT threshold
    question_lower_for_agg = question.lower()
    count_language = bool(
        re.search(
            r"\b(most|fewest|how many|number of)\s+"
            r"(purchases?|orders?|transactions?|records?|events?|items?)\b",
            question_lower_for_agg,
        )
    )

    if "average" in question_lower_for_agg or re.search(
        r"\bmean\b|\bavg\b", question_lower_for_agg
    ):
        requirements["aggregation"] = "mean"
    elif "total" in question_lower_for_agg or re.search(
        r"\b(spent|spend|revenue|sales)\b", question_lower_for_agg
    ):
        requirements["aggregation"] = "sum"
    elif "maximum" in question_lower_for_agg or re.search(
        r"\bhighest value\b", question_lower_for_agg
    ):
        requirements["aggregation"] = "max"
    elif "minimum" in question_lower_for_agg or re.search(
        r"\blowest value\b", question_lower_for_agg
    ):
        requirements["aggregation"] = "min"
    elif count_language:
        requirements["aggregation"] = "count"

    def choose_metric_column_from_question(tables, question):
        """Infer a numeric metric column from the original question.

        Used when the LLM leaves METRIC empty for an aggregate query.
        Matches natural-language forms such as:
        "order amount", "total revenue", "sales amount", etc.
        """
        question_lower = question.lower()

        # Build normalized token forms for every column.
        candidates = []

        for table_name, df in tables.items():
            for column in df.columns:
                if not pd.api.types.is_numeric_dtype(df[column]):
                    continue

                raw = str(column).strip()
                normalized = re.sub(r"[_\-]+", " ", raw).lower()
                normalized = re.sub(r"\s+", " ", normalized).strip()

                score = 0

                # Exact normalized phrase match.
                if normalized and normalized in question_lower:
                    score += 100

                # Match individual tokens from the column name.
                tokens = [
                    token
                    for token in re.findall(r"[a-z0-9]+", normalized)
                    if token not in {"id"}
                ]

                matched_tokens = sum(
                    1
                    for token in tokens
                    if re.search(rf"\b{re.escape(token)}\b", question_lower)
                )

                score += matched_tokens * 10

                # Common business-language aliases.
                alias_groups = {
                    "order amount": {"amount", "value", "sales", "revenue"},
                    "purchase value": {"value", "amount", "sales", "revenue"},
                    "revenue": {"revenue", "sales", "amount", "value"},
                    "sales": {"sales", "revenue", "amount", "value"},
                    "value": {"value", "amount", "revenue", "sales"},
                    "amount": {"amount", "value", "revenue", "sales"},
                }

                for phrase, aliases in alias_groups.items():
                    if phrase in question_lower:
                        if any(alias in normalized for alias in aliases):
                            score += 6

                if score > 0:
                    candidates.append(
                        (score, table_name, raw)
                    )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (item[0], item[1] == "Orders"),
            reverse=True,
        )

        return candidates[0][2]

    # Resolve a semantic metric alias such as Revenue/Sales to the actual
    # numeric schema column when the LLM emitted a name that is not present.
    current_value_column = requirements.get("value_column")
    if current_value_column and not any(
        current_value_column in df.columns for df in tables.values()
    ):
        inferred_metric = choose_metric_column_from_question(
            tables,
            question,
        )
        if inferred_metric:
            requirements["value_column"] = inferred_metric

    def choose_count_column_for_question(tables, question):
        all_columns = [
            column
            for df in tables.values()
            for column in df.columns
        ]

        question_lower_for_count = question.lower()

        context_preferences = []

        if any(word in question_lower_for_count for word in [
            "purchase", "purchases", "bought", "buy"
        ]):
            context_preferences.extend([
                "Purchase_ID", "PurchaseId", "PurchaseID"
            ])

        if any(word in question_lower_for_count for word in [
            "order", "orders"
        ]):
            context_preferences.extend([
                "Order_ID", "OrderId", "OrderID"
            ])

        if any(word in question_lower_for_count for word in [
            "transaction", "transactions"
        ]):
            context_preferences.extend([
                "Transaction_ID", "TransactionId", "TransactionID"
            ])

        if any(word in question_lower_for_count for word in [
            "sale", "sales"
        ]):
            context_preferences.extend([
                "Sale_ID", "SaleId", "SaleID"
            ])

        context_preferences.extend([
            "Record_ID", "RecordId", "RecordID", "ID"
        ])

        for candidate in context_preferences:
            for actual_column in all_columns:
                if actual_column.lower() == candidate.lower():
                    return actual_column

        for actual_column in all_columns:
            if str(actual_column).lower().endswith("_id"):
                return actual_column

        return all_columns[0] if all_columns else None

    # Schema-ground the LLM metric after the metric-inference helper is
    # available. This repairs business-language metrics such as
    # "Revenue" when the actual dataset column is Order_Amount.
    if metric_name and requirements.get("value_column"):
        metric_candidate = requirements["value_column"]
        if not any(
            metric_candidate in df.columns
            for df in tables.values()
        ):
            inferred_metric = choose_metric_column_from_question(
                tables,
                question,
            )
            if inferred_metric:
                requirements["value_column"] = inferred_metric

    # COUNT MUST use a real record identifier.
    # Do not allow a numeric measure such as Order_Amount to remain as the
    # value column for "most/fewest/how many" queries.
    if requirements["aggregation"] == "count":
        requirements["value_column"] = choose_count_column_for_question(
            tables,
            question,
        )

    # If an aggregate/ranking query has no metric, infer a numeric metric
    # deterministically from the original question instead of failing later.
    if (
        requirements.get("aggregation") in {"sum", "mean", "min", "max"}
        and not requirements.get("value_column")
    ):
        inferred_metric = choose_metric_column_from_question(
            tables,
            question,
        )

        if inferred_metric:
            requirements["value_column"] = inferred_metric

    # RANK / ORDER
    for line in lines:
        if line.upper().startswith("RANK="):
            rank = line.split("=", 1)[1].strip().lower()
            if rank == "highest":
                requirements["order"] = "desc"
            elif rank == "lowest":
                requirements["order"] = "asc"

    # Deterministic top/bottom correction directly from original question.
    question_lower = question.lower()

    if re.search(r"\bbottom\s+\d+\b", question_lower):
        requirements["order"] = "asc"
    elif re.search(r"\btop\s+\d+\b", question_lower):
        requirements["order"] = "desc"

    # LIMIT
    for line in lines:
        if line.upper().startswith("LIMIT="):
            value = line.split("=", 1)[1].strip().lower()
            if value not in {"none", "null", "", "-"} and value.isdigit():
                requirements["limit"] = int(value)

    # Original question is the final authority for explicit top/bottom N.
    limit_match = re.search(
        r"\b(?:top|bottom)\s+(\d+)\b",
        question_lower,
    )
    requirements["limit"] = (
        int(limit_match.group(1)) if limit_match else None
    )

    # Original question is authoritative for raw top/bottom queries.
    result_type = requirements.get("result_type")
    question_lower_for_mode = question.lower()

    raw_top_bottom = bool(
        re.search(
            r"\b(?:top|bottom)\s+\d+\s+(?:gold\s+|silver\s+|standard\s+|premium\s+|north\s+|south\s+|east\s+|west\s+)*"
            r"(?:purchases?|orders?|transactions?|records?|items?|rows?)\b",
            question_lower_for_mode,
        )
    )

    list_query = result_type == "list" or raw_top_bottom

    if list_query:
        requirements["result_type"] = "list"
        requirements["group_column"] = None
        requirements["aggregation"] = None

        # Raw top/bottom queries still need a sort direction and limit.
        if raw_top_bottom:
            if re.search(r"\bbottom\s+\d+\b", question_lower_for_mode):
                requirements["order"] = "asc"
            else:
                requirements["order"] = "desc"
        else:
            requirements["order"] = None
            requirements["limit"] = None

    elif requirements["result_type"] is None:
        if requirements["having"]:
            requirements["result_type"] = "grouped_filtered"
        elif requirements["order"] in {"asc", "desc"}:
            requirements["result_type"] = "ranked"
        else:
            requirements["result_type"] = "grouped"

    # ------------------------------------------------------------
    # RAW TOP/BOTTOM SORT COLUMN
    # ------------------------------------------------------------
    requirements["sort_column"] = None

    if requirements.get("result_type") == "list" and requirements.get("limit") is not None:
        if requirements.get("value_column"):
            requirements["sort_column"] = requirements["value_column"]
        else:
            metric_keywords = {
                "value": ["Purchase_Value", "Amount", "Revenue", "Value"],
                "amount": ["Amount", "Purchase_Value", "Revenue"],
                "revenue": ["Revenue", "Amount", "Purchase_Value"],
                "price": ["Price", "Amount", "Purchase_Value"],
                "score": ["Score", "Performance_Score"],
            }

            numeric_columns = [
                column
                for df in tables.values()
                for column in df.columns
                if pd.api.types.is_numeric_dtype(df[column])
            ]

            for keyword, candidates in metric_keywords.items():
                if re.search(rf"\b{re.escape(keyword)}\b", question_lower_for_mode):
                    for candidate in candidates:
                        for actual in numeric_columns:
                            if actual.lower() == candidate.lower():
                                requirements["sort_column"] = actual
                                break
                        if requirements["sort_column"]:
                            break
                if requirements["sort_column"]:
                    break

            if requirements["sort_column"] is None and numeric_columns:
                # Avoid choosing units/count-like columns when a purchase value exists.
                requirements["sort_column"] = (
                    next((c for c in numeric_columns if c.lower() == "purchase_value"), numeric_columns[0])
                )

            if requirements.get("source_column") is None:
                requirements["source_column"] = requirements.get("sort_column")

    # ------------------------------------------------------------
    # SCHEMA-GROUNDED NORMALIZATION: "Gold purchases" means
    # purchases made by Gold clients, not Category=Gold, when the
    # dataset contains Client_Type.
    # ------------------------------------------------------------
    if requirements.get("entity", "").strip().lower() == "purchase":
        all_columns = [column for df in tables.values() for column in df.columns]
        client_type_column = next(
            (c for c in all_columns if c.lower() == "client_type"),
            None,
        )
        category_column = next(
            (c for c in all_columns if c.lower() == "category"),
            None,
        )

        if client_type_column and re.search(r"\bgold\s+(?:purchases?|orders?|transactions?)\b", question_lower_for_mode):
            normalized_filters = []
            replaced = False
            for item in requirements["filters"]:
                if (
                    category_column
                    and item["column"] == category_column
                    and item["value"].strip().lower() == "gold"
                ):
                    if not replaced:
                        normalized_filters.append({
                            "column": client_type_column,
                            "operator": "==",
                            "value": "Gold",
                        })
                        replaced = True
                else:
                    normalized_filters.append(item)

            if not replaced:
                normalized_filters.append({
                    "column": client_type_column,
                    "operator": "==",
                    "value": "Gold",
                })

            requirements["filters"] = normalized_filters

    # ------------------------------------------------------------
    # DETERMINISTIC AGGREGATE-THRESHOLD RECOVERY
    # ------------------------------------------------------------
    # Recover aggregate thresholds directly from the ORIGINAL QUESTION.
    # This protects against LLM omissions for phrases such as:
    #   "total order amount above 5000"
    #   "total revenue more than 5000"
    #   "average order amount below 3000"
    #
    # These conditions belong to HAVING, never FILTER.
    aggregate_threshold_match = re.search(
        r"\b(at least|at most|more than|greater than|above|over|"
        r"less than|below|under|exactly)\s+(\d+(?:\.\d+)?)\s+"
        r"(?:in\s+)?(?:the\s+)?(?:total\s+|average\s+|avg\s+|mean\s+)?"
        r"(revenue|sales|amount|order\s+amount|purchase\s+value|value)\b",
        question_lower_for_mode,
    )

    reverse_aggregate_threshold_match = re.search(
        r"\b(?:total\s+|average\s+|avg\s+|mean\s+)?"
        r"(revenue|sales|amount|order\s+amount|purchase\s+value|value)\s+"
        r"(at least|at most|more than|greater than|above|over|"
        r"less than|below|under|exactly)\s+(\d+(?:\.\d+)?)\b",
        question_lower_for_mode,
    )

    def aggregate_threshold_operator(phrase):
        phrase = phrase.lower()

        if phrase == "at least":
            return ">="
        if phrase == "at most":
            return "<="
        if phrase in {"more than", "greater than", "above", "over"}:
            return ">"
        if phrase in {"less than", "below", "under"}:
            return "<"

        return "=="

    detected_aggregate_threshold = None

    if aggregate_threshold_match:
        detected_aggregate_threshold = (
            aggregate_threshold_match.group(1),
            aggregate_threshold_match.group(2),
            aggregate_threshold_match.group(3),
        )
    elif reverse_aggregate_threshold_match:
        detected_aggregate_threshold = (
            reverse_aggregate_threshold_match.group(2),
            reverse_aggregate_threshold_match.group(3),
            reverse_aggregate_threshold_match.group(1),
        )

    if (
        detected_aggregate_threshold
        and requirements.get("aggregation") in {
            "sum", "mean", "min", "max"
        }
    ):
        phrase, threshold_value, _ = detected_aggregate_threshold

        metric_column = requirements.get("value_column")

        if metric_column:
            aggregate_having = {
                "column": metric_column,
                "operator": aggregate_threshold_operator(phrase),
                "value": threshold_value,
            }

            already_present = any(
                item.get("column") == aggregate_having["column"]
                and item.get("operator") == aggregate_having["operator"]
                and str(item.get("value")) == str(aggregate_having["value"])
                for item in requirements.get("having", [])
            )

            if not already_present:
                requirements["having"].append(aggregate_having)

            requirements["result_type"] = "grouped_filtered"

    # COUNT threshold marker -> use the GROUP-generated Count column.
    # When another aggregate (SUM/MEAN/etc.) is primary, compute COUNT as an
    # additional aggregate and merge it into the grouped result.
    for having in requirements.get("having", []):
        if having.get("column") == "__COUNT__":
            if requirements.get("aggregation") != "count":
                count_column = choose_count_column_for_question(
                    tables,
                    question,
                )

                if not any(
                    item.get("aggregation") == "count"
                    for item in requirements.get("additional_aggregations", [])
                ):
                    requirements["additional_aggregations"].append(
                        {
                            "aggregation": "count",
                            "value_column": count_column,
                        }
                    )

            having["column"] = "Count"



    # Entity column is only necessary for entity extraction.
    if requirements["group_column"]:
        requirements["entity_column"] = requirements["group_column"]
    elif requirements["entity"]:
        requirements["entity_column"] = resolve_schema_column(
            requirements["entity"],
            tables,
        )

    # ------------------------------------------------------------
    # RAW/LIST SOURCE COLUMN INFERENCE
    # ------------------------------------------------------------
    # A raw visualization query may have no GROUP_BY / METRIC in the
    # semantic result, but the original question can still identify the
    # numeric source column. Example:
    #   "Show the distribution of order amounts."
    # The query should therefore start from the table containing
    # Order_Amount (e.g. Orders), not simply the first loaded table.
    if requirements.get("result_type") == "list" and not requirements.get("value_column"):
        inferred_source_column = choose_metric_column_from_question(
            tables,
            question,
        )
        if inferred_source_column:
            requirements["source_column"] = inferred_source_column

    # ------------------------------------------------------------
    # DETERMINE ONLY THE TABLES ACTUALLY NEEDED BY THIS QUESTION
    # ------------------------------------------------------------
    required_tables = []

    def add_tables_for_column(column):
        for table_name in tables_for_column(column, tables):
            if table_name not in required_tables:
                required_tables.append(table_name)

    add_tables_for_column(requirements.get("group_column"))
    add_tables_for_column(requirements.get("value_column"))
    add_tables_for_column(requirements.get("entity_column"))
    add_tables_for_column(requirements.get("source_column"))

    for item in requirements.get("filters", []):
        add_tables_for_column(item.get("column"))

    for item in requirements.get("having", []):
        having_column = item.get("column")
        if having_column != "Count":
            add_tables_for_column(having_column)

    for item in requirements.get("additional_aggregations", []):
        add_tables_for_column(item.get("value_column"))

    # Safe fallback for generic raw-table questions that do not mention
    # a schema column explicitly.
    if not required_tables and tables:
        required_tables.append(next(iter(tables)))

    requirements["tables"] = required_tables

    # Final generic metric fallback. Prefer a numeric column from the
    # question before using any legacy compatibility fallback.
    if (
        requirements["value_column"] is None
        and requirements["aggregation"] in {"sum", "mean", "min", "max"}
    ):
        inferred_metric = choose_metric_column_from_question(
            tables,
            question,
        )

        if inferred_metric:
            requirements["value_column"] = inferred_metric

    if (
        requirements["value_column"] is None
        and requirements["aggregation"] in {"sum", "mean", "min", "max"}
    ):
        for table_name in requirements.get("tables", []):
            df = tables.get(table_name)

            if df is None:
                continue

            numeric_columns = [
                column
                for column in df.columns
                if pd.api.types.is_numeric_dtype(df[column])
                and not str(column).lower().endswith("_id")
            ]

            if numeric_columns:
                requirements["value_column"] = numeric_columns[0]
                break

    # ------------------------------------------------------------
    # MOVE AGGREGATE-METRIC FILTERS INTO HAVING
    # ------------------------------------------------------------
    # LLMs sometimes emit a row-level filter for an aggregate phrase, e.g.
    #   "customers ... average order amount above 3000"
    # as FILTER=Order_Amount>3000.
    # Once the question clearly requests an aggregate threshold, that
    # condition must be evaluated after GROUP, not before GROUP.
    aggregate_threshold_language = bool(
        re.search(
            r"\b(?:above|over|more than|greater than|at least|at most|"
            r"below|under|less than|exactly)\s+\d+(?:\.\d+)?\b",
            question_lower_for_mode,
        )
    )
    explicit_aggregate_words = bool(
        re.search(
            r"\b(?:total|average|avg|mean|spent|spend|revenue|sales|"
            r"maximum|minimum|highest|lowest)\b",
            question_lower_for_mode,
        )
    )

    if (
        requirements.get("aggregation") in {"sum", "mean", "min", "max"}
        and requirements.get("value_column")
        and aggregate_threshold_language
        and explicit_aggregate_words
    ):
        metric_column = requirements["value_column"]
        moved_filters = []
        remaining_filters = []

        for item in requirements.get("filters", []):
            if str(item.get("column")).strip().lower() == str(metric_column).strip().lower():
                moved_filters.append(item)
            else:
                remaining_filters.append(item)

        requirements["filters"] = remaining_filters

        for item in moved_filters:
            if not any(
                having.get("column") == item.get("column")
                and having.get("operator") == item.get("operator")
                and str(having.get("value")) == str(item.get("value"))
                for having in requirements.get("having", [])
            ):
                requirements["having"].append(item)

        if moved_filters or requirements.get("having"):
            requirements["result_type"] = "grouped_filtered"


    # Re-run aggregate-threshold recovery after final metric inference.
    # This is intentionally deterministic and uses the original question.
    final_aggregate_threshold = re.search(
        r"\b(at least|at most|more than|greater than|above|over|"
        r"less than|below|under|exactly)\s+(\d+(?:\.\d+)?)\s+"
        r"(?:in\s+)?(?:the\s+)?(?:total\s+|average\s+|avg\s+|mean\s+)?"
        r"(revenue|sales|amount|order\s+amount|purchase\s+value|value)\b",
        question.lower(),
    )

    final_reverse_aggregate_threshold = re.search(
        r"\b(?:total\s+|average\s+|avg\s+|mean\s+)?"
        r"(revenue|sales|amount|order\s+amount|purchase\s+value|value)\s+"
        r"(at least|at most|more than|greater than|above|over|"
        r"less than|below|under|exactly)\s+(\d+(?:\.\d+)?)\b",
        question.lower(),
    )

    if (
        requirements.get("aggregation") in {"sum", "mean", "min", "max"}
        and requirements.get("value_column")
    ):
        if final_aggregate_threshold:
            threshold_phrase = final_aggregate_threshold.group(1)
            threshold_value = final_aggregate_threshold.group(2)
        elif final_reverse_aggregate_threshold:
            threshold_phrase = final_reverse_aggregate_threshold.group(2)
            threshold_value = final_reverse_aggregate_threshold.group(3)
        else:
            threshold_phrase = None
            threshold_value = None

        if threshold_phrase is not None:
            threshold_ops = {
                "at least": ">=",
                "at most": "<=",
                "more than": ">",
                "greater than": ">",
                "above": ">",
                "over": ">",
                "less than": "<",
                "below": "<",
                "under": "<",
                "exactly": "==",
            }

            aggregate_having = {
                "column": requirements["value_column"],
                "operator": threshold_ops[threshold_phrase.lower()],
                "value": threshold_value,
            }

            if not any(
                item.get("column") == aggregate_having["column"]
                and item.get("operator") == aggregate_having["operator"]
                and str(item.get("value")) == str(aggregate_having["value"])
                for item in requirements.get("having", [])
            ):
                requirements["having"].append(aggregate_having)

            requirements["result_type"] = "grouped_filtered"

    return requirements


requirements = map_semantic_intent_to_requirements(
    validated_semantic_intent,
    tables,
    question,
)

print("\n================ REQUIREMENTS ================\n")
print(requirements)


# ============================================================
# 7. REQUIREMENT VALIDATION
# ============================================================

def validate_requirements(requirements, tables):
    errors = []

    required_tables = requirements.get("tables", [])

    # Tables
    for table_name in required_tables:
        if table_name not in tables:
            errors.append(f"Unknown table: {table_name}")

    def column_exists(column):
        if not column:
            return False
        return any(
            column in tables[table_name].columns
            for table_name in required_tables
            if table_name in tables
        )

    # Filters
    for item in requirements.get("filters", []):
        if not column_exists(item.get("column")):
            errors.append(
                f"Filter column '{item.get('column')}' not found."
            )

        if item.get("operator") not in {
            ">", "<", ">=", "<=", "==", "!=", "is null", "is not null"
        }:
            errors.append(
                f"Invalid filter operator: {item.get('operator')}"
            )

    # Having
    for item in requirements.get("having", []):
        having_column = item.get("column")

        # "Count" is a generated column produced by GROUP for COUNT
        # queries. It does not exist in the original source tables.
        generated_count_column = (
            str(having_column).lower() in {"count", "counts", "n"}
            and (
                requirements.get("aggregation") == "count"
                or any(
                    extra.get("aggregation") == "count"
                    for extra in requirements.get(
                        "additional_aggregations", []
                    )
                )
            )
        )

        if not column_exists(having_column) and not generated_count_column:
            errors.append(
                f"HAVING column '{having_column}' not found."
            )

        if item.get("operator") not in {
            ">", "<", ">=", "<=", "==", "!="
        }:
            errors.append(
                f"Invalid HAVING operator: {item.get('operator')}"
            )

    # Group column
    if requirements.get("group_column") and not column_exists(
        requirements["group_column"]
    ):
        errors.append(
            f"Group column '{requirements['group_column']}' not found."
        )

    # Value column
    if requirements.get("value_column") and not column_exists(
        requirements["value_column"]
    ):
        errors.append(
            f"Value column '{requirements['value_column']}' not found."
        )

    # Aggregation
    aggregation = requirements.get("aggregation")
    if aggregation and aggregation not in {
        "sum", "mean", "min", "max", "count"
    }:
        errors.append(f"Invalid aggregation: {aggregation}")

    # Order
    if requirements.get("order") and requirements["order"] not in {
        "asc", "desc"
    }:
        errors.append(f"Invalid order: {requirements['order']}")

    # Sort column (used by raw top/bottom list queries)
    sort_column = requirements.get("sort_column")
    if sort_column and not column_exists(sort_column):
        errors.append(f"Sort column '{sort_column}' not found.")

    # Limit
    limit = requirements.get("limit")
    if limit is not None and (
        not isinstance(limit, int) or limit <= 0
    ):
        errors.append(f"Invalid LIMIT: {limit}")

    return errors


validation_errors = validate_requirements(requirements, tables)

if validation_errors:
    print("\n================ REQUIREMENT ERRORS ================\n")
    for error in validation_errors:
        print("[ERROR]", str(error))
    raise ValueError("Invalid requirements.")

print("\n================ REQUIREMENTS VALID ================\n")


# ============================================================
# 8. BUILD EXECUTION PLAN
# ============================================================

def build_execution_plan(requirements, tables):
    plan_steps = []
    step_number = 1

    required_tables = requirements.get("tables", [])

    if not required_tables:
        raise ValueError("No tables required for this question.")

    # ------------------------------------------------------------
    # Select a sensible starting table. Prefer the table containing
    # the primary measure/count column.
    # ------------------------------------------------------------
    start_table = required_tables[0]

    preferred_columns = []
    if requirements.get("value_column"):
        preferred_columns.append(requirements["value_column"])

    for extra in requirements.get("additional_aggregations", []):
        if extra.get("value_column"):
            preferred_columns.append(extra["value_column"])

    for candidate_table in required_tables:
        df = tables[candidate_table]
        if any(column in df.columns for column in preferred_columns):
            start_table = candidate_table
            break

    current_df = tables[start_table]
    current_source = start_table
    remaining_tables = [
        table_name
        for table_name in required_tables
        if table_name != start_table
    ]

    # ------------------------------------------------------------
    # Build a generic JOIN chain.
    # ------------------------------------------------------------
    connected_columns = set(current_df.columns)

    while remaining_tables:
        selected_table = None
        selected_join_column = None

        for table_name in remaining_tables:
            right_df = tables[table_name]
            common_columns = [
                column
                for column in right_df.columns
                if column in connected_columns
            ]

            if not common_columns:
                continue

            id_columns = [
                column
                for column in common_columns
                if str(column).lower().endswith("_id")
                or str(column).lower() == "id"
            ]

            selected_join_column = (
                id_columns[0] if id_columns else common_columns[0]
            )
            selected_table = table_name
            break

        if selected_table is None:
            raise ValueError(
                "Could not build a JOIN path for required tables: "
                + ", ".join(required_tables)
            )

        left_table = current_source if step_number == 1 else "__CURRENT__"

        plan_steps.append({
            "step": step_number,
            "tool": "JOIN",
            "table": None,
            "left_table": left_table,
            "right_table": selected_table,
            "left_column": selected_join_column,
            "right_column": selected_join_column,
            "column": None,
            "operator": None,
            "value": None,
            "group_column": None,
            "value_column": None,
            "aggregation": None,
            "order": None,
            "limit": None,
            "having": [],
            "additional_aggregations": [],
        })
        step_number += 1

        # Simulate the join so the next table is chosen against the
        # accumulated schema.
        current_df = TOOL_REGISTRY["JOIN"](
            current_df,
            tables[selected_table],
            selected_join_column,
            selected_join_column,
        )
        connected_columns.update(tables[selected_table].columns)
        current_source = f"{current_source}+{selected_table}"
        remaining_tables.remove(selected_table)

    # ------------------------------------------------------------
    # FILTERS
    # ------------------------------------------------------------
    for filter_requirement in requirements["filters"]:
        plan_steps.append({
            "step": step_number,
            "tool": "FILTER",
            "table": start_table,
            "left_table": None,
            "right_table": None,
            "left_column": None,
            "right_column": None,
            "column": filter_requirement["column"],
            "operator": filter_requirement["operator"],
            "value": filter_requirement["value"],
            "group_column": None,
            "value_column": None,
            "aggregation": None,
            "order": None,
            "limit": None,
            "having": [],
            "additional_aggregations": [],
        })
        step_number += 1

    # ------------------------------------------------------------
    # GROUP
    # ------------------------------------------------------------
    if requirements["group_column"]:
        plan_steps.append({
            "step": step_number,
            "tool": "GROUP",
            "table": start_table,
            "left_table": None,
            "right_table": None,
            "left_column": None,
            "right_column": None,
            "column": None,
            "operator": None,
            "value": None,
            "group_column": requirements["group_column"],
            "value_column": requirements["value_column"],
            "aggregation": requirements["aggregation"],
            "additional_aggregations": requirements.get(
                "additional_aggregations",
                [],
            ),
            "order": requirements["order"],
            "limit": requirements.get("limit"),
            "having": requirements.get("having", []),
        })

    # Raw list queries without JOIN/FILTER/GROUP still need an explicit
    # source step so the executor has a dataframe to work with.
    if not plan_steps:
        plan_steps.append({
            "step": step_number,
            "tool": "SOURCE",
            "table": start_table,
            "left_table": None,
            "right_table": None,
            "left_column": None,
            "right_column": None,
            "column": None,
            "operator": None,
            "value": None,
            "group_column": None,
            "value_column": None,
            "aggregation": None,
            "order": None,
            "limit": None,
            "having": [],
            "additional_aggregations": [],
        })

    return plan_steps


plan_steps = build_execution_plan(requirements, tables)

print("\n================ GENERATED PLAN ================\n")
for step in plan_steps:
    print(
        f"STEP={step['step']} TOOL={step['tool']}"
    )


# ============================================================
# 9. PLAN VALIDATION
# ============================================================

def validate_plan(plan_steps, tables):
    errors = []

    if not plan_steps:
        errors.append("No execution steps were generated.")
        return errors

    expected_step = 1

    for step in plan_steps:
        if step["step"] != expected_step:
            errors.append(
                f"Invalid step order: expected {expected_step}, got {step['step']}"
            )
        expected_step += 1

        tool = step["tool"]

        if tool == "SOURCE":
            source_table = step.get("table")
            if source_table not in tables:
                errors.append(f"Unknown SOURCE table: {source_table}")

        elif tool == "JOIN":
            right_table = step.get("right_table")
            left_table = step.get("left_table")

            if right_table not in tables:
                errors.append(f"Unknown JOIN table: {right_table}")

            if left_table != "__CURRENT__" and left_table not in tables:
                errors.append(f"Unknown JOIN table: {left_table}")

            if (
                right_table in tables
                and step.get("right_column") not in tables[right_table].columns
            ):
                errors.append(
                    f"JOIN column {step.get('right_column')} "
                    f"not found in {right_table}"
                )

            if (
                left_table != "__CURRENT__"
                and left_table in tables
                and step.get("left_column") not in tables[left_table].columns
            ):
                errors.append(
                    f"JOIN column {step.get('left_column')} "
                    f"not found in {left_table}"
                )

        elif tool == "FILTER":
            if not step.get("column"):
                errors.append("FILTER is missing COLUMN.")

            if step.get("operator") not in {
                ">", "<", ">=", "<=", "==", "!=", "is null", "is not null"
            }:
                errors.append(
                    f"Invalid FILTER operator: {step.get('operator')}"
                )

            if step.get("value") is None and step.get("operator") not in {"is null", "is not null"}:
                errors.append("FILTER is missing VALUE.")

        elif tool == "GROUP":
            if not step.get("group_column"):
                errors.append("GROUP is missing GROUP_COLUMN.")

            if not step.get("value_column"):
                errors.append("GROUP is missing VALUE_COLUMN.")

            if step.get("aggregation") not in {
                "sum", "mean", "min", "max", "count"
            }:
                errors.append(
                    f"Invalid aggregation: {step.get('aggregation')}"
                )

            if step.get("order") not in {None, "asc", "desc"}:
                errors.append(
                    f"Invalid order: {step.get('order')}"
                )

            if step.get("limit") is not None:
                if not isinstance(step["limit"], int) or step["limit"] <= 0:
                    errors.append(
                        f"Invalid LIMIT: {step.get('limit')}"
                    )

        else:
            errors.append(f"Unknown TOOL: {tool}")

    return errors


plan_errors = validate_plan(plan_steps, tables)

if plan_errors:
    print("\n================ PLAN ERRORS ================\n")
    for error in plan_errors:
        print("[ERROR]", str(error))
    raise ValueError("Invalid execution plan.")

print("\n================ PLAN VALID ================\n")


# ============================================================
# 10. EXECUTE PLAN
# ============================================================

def execute_plan(plan_steps, tables):
    current_df = None

    for step in plan_steps:
        tool = step["tool"]

        print(
            f"\n================ STEP {step['step']} : {tool} ================\n"
        )

        if tool == "SOURCE":
            source_table = step["table"]
            current_df = tables[source_table].copy()

        elif tool == "JOIN":
            right_df = tables[step["right_table"]]

            if step.get("left_table") == "__CURRENT__":
                if current_df is None:
                    raise ValueError(
                        "Chained JOIN requires an existing dataframe."
                    )
                left_df = current_df
            else:
                left_df = tables[step["left_table"]]

            current_df = TOOL_REGISTRY["JOIN"](
                left_df,
                right_df,
                step["left_column"],
                step["right_column"],
            )

        elif tool == "FILTER":
            if current_df is None:
                table_name = step.get("table")
                if table_name is None:
                    raise ValueError("FILTER has no source table.")
                current_df = tables[table_name]

            column = step["column"]
            operator = step["operator"]
            value = step["value"]

            if column not in current_df.columns:
                raise ValueError(
                    f"Filter column '{column}' not found in current data."
                )

            series = current_df[column]

            if operator in {"is null", "is not null"}:
                missing_mask = series.isna()
                if pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series):
                    missing_mask = missing_mask | series.astype(str).str.strip().eq("")
                mask = missing_mask if operator == "is null" else ~missing_mask
                current_df = current_df.loc[mask].copy()
                continue

            # Numeric comparison when possible.
            try:
                numeric_value = float(value)
                is_numeric_filter = True
            except (ValueError, TypeError):
                numeric_value = None
                is_numeric_filter = False

            if is_numeric_filter:
                numeric_series = pd.to_numeric(
                    series,
                    errors="coerce",
                )

                if operator == ">":
                    mask = numeric_series > numeric_value
                elif operator == "<":
                    mask = numeric_series < numeric_value
                elif operator == ">=":
                    mask = numeric_series >= numeric_value
                elif operator == "<=":
                    mask = numeric_series <= numeric_value
                elif operator == "==":
                    mask = numeric_series == numeric_value
                elif operator == "!=":
                    mask = numeric_series != numeric_value
                else:
                    raise ValueError(
                        f"Unsupported FILTER operator: {operator}"
                    )

            else:
                text_series = series.astype(str).str.strip()
                text_value = str(value).strip()

                if operator == "==":
                    mask = text_series.str.lower() == text_value.lower()
                elif operator == "!=":
                    mask = text_series.str.lower() != text_value.lower()
                else:
                    raise ValueError(
                        f"Operator '{operator}' is not valid for text filtering."
                    )

            current_df = current_df.loc[mask].copy()

        elif tool == "GROUP":
            if current_df is None:
                table_name = step.get("table")
                if table_name is None:
                    raise ValueError("GROUP has no source table.")
                current_df = tables[table_name]

            group_column = step["group_column"]
            value_column = step["value_column"]
            aggregation = (step["aggregation"] or "").lower()
            order = step.get("order")
            limit = step.get("limit")
            having_conditions = step.get("having", [])

            if group_column not in current_df.columns:
                raise ValueError(
                    f"Group column '{group_column}' not found."
                )

            # COUNT should always operate on a real record identifier.
            if aggregation == "count" and value_column not in current_df.columns:
                candidate_count_columns = [
                    column
                    for column in current_df.columns
                    if str(column).lower().endswith("_id")
                ]

                if candidate_count_columns:
                    value_column = candidate_count_columns[0]
                else:
                    raise ValueError(
                        f"Value column '{value_column}' not found for COUNT."
                    )

            elif value_column not in current_df.columns:
                raise ValueError(
                    f"Value column '{value_column}' not found."
                )

            # Keep source data before GROUP so descriptive columns such as
            # Client_Name can be restored after aggregation.
            source_df = current_df.copy()

            grouped_df = TOOL_REGISTRY["GROUP"](
                current_df,
                group_column,
                value_column,
                aggregation,
            )

            # Preserve the canonical name column for ID-based entity groups.
            descriptive_columns = []
            if str(group_column).lower().endswith("_id"):
                prefix = str(group_column)[:-3]
                preferred_name = prefix + "_Name"
                for column in source_df.columns:
                    if column.lower() == preferred_name.lower():
                        descriptive_columns.append(column)
                        break

            if descriptive_columns:
                lookup = (
                    source_df[[group_column] + descriptive_columns]
                    .drop_duplicates(subset=[group_column])
                )
                current_df = grouped_df.merge(
                    lookup,
                    on=group_column,
                    how="left",
                )
            else:
                current_df = grouped_df

            # Compute additional aggregates requested by the semantic layer,
            # e.g. SUM(Order_Amount) + COUNT(Order_ID) for:
            # "customers with more than 2 orders and spent over 9000".
            additional_aggregations = step.get(
                "additional_aggregations",
                [],
            )

            for extra in additional_aggregations:
                extra_aggregation = (extra.get("aggregation") or "").lower()
                extra_value_column = extra.get("value_column")

                if not extra_value_column:
                    raise ValueError(
                        "Additional aggregation is missing VALUE_COLUMN."
                    )

                if extra_value_column not in source_df.columns:
                    raise ValueError(
                        f"Additional value column "
                        f"'{extra_value_column}' not found."
                    )

                extra_grouped = TOOL_REGISTRY["GROUP"](
                    source_df,
                    group_column,
                    extra_value_column,
                    extra_aggregation,
                )

                # Normalize COUNT output to a stable "Count" column.
                if extra_aggregation == "count":
                    if "Count" not in extra_grouped.columns:
                        extra_metric_columns = [
                            c for c in extra_grouped.columns
                            if c != group_column
                        ]
                        if extra_metric_columns:
                            extra_grouped = extra_grouped.rename(
                                columns={
                                    extra_metric_columns[0]: "Count"
                                }
                            )

                metric_columns = [
                    c for c in extra_grouped.columns
                    if c != group_column
                ]

                if not metric_columns:
                    raise ValueError(
                        "Additional aggregation produced no metric column."
                    )

                for metric_column in metric_columns:
                    if metric_column in current_df.columns:
                        raise ValueError(
                            f"Additional aggregation column "
                            f"'{metric_column}' already exists."
                        )

                current_df = current_df.merge(
                    extra_grouped,
                    on=group_column,
                    how="left",
                )

            # Apply HAVING after all aggregate columns are available.
            for having in having_conditions:
                column = having["column"]
                operator = having["operator"]
                raw_value = having["value"]

                if column not in current_df.columns:
                    if aggregation == "count":
                        count_columns = [
                            c for c in current_df.columns
                            if str(c).lower() in {"count", "counts", "n"}
                        ]

                        if count_columns:
                            column = count_columns[0]
                        else:
                            raise ValueError(
                                f"HAVING column '{column}' not found after GROUP."
                            )
                    else:
                        raise ValueError(
                            f"HAVING column '{column}' not found after GROUP."
                        )

                try:
                    numeric_value = float(raw_value)
                    series = pd.to_numeric(
                        current_df[column],
                        errors="coerce",
                    )
                    numeric_mode = True
                except (ValueError, TypeError):
                    numeric_value = None
                    series = current_df[column].astype(str).str.strip()
                    numeric_mode = False

                if numeric_mode:
                    if operator == ">":
                        mask = series > numeric_value
                    elif operator == "<":
                        mask = series < numeric_value
                    elif operator == ">=":
                        mask = series >= numeric_value
                    elif operator == "<=":
                        mask = series <= numeric_value
                    elif operator == "==":
                        mask = series == numeric_value
                    elif operator == "!=":
                        mask = series != numeric_value
                    else:
                        raise ValueError(
                            f"Unsupported HAVING operator: {operator}"
                        )
                else:
                    text_value = str(raw_value).strip()
                    if operator == "==":
                        mask = series.str.lower() == text_value.lower()
                    elif operator == "!=":
                        mask = series.str.lower() != text_value.lower()
                    else:
                        raise ValueError(
                            f"Operator '{operator}' is not valid for text HAVING."
                        )

                current_df = current_df.loc[mask].copy()
            # ------------------------------------------------------------
            # SORT AGGREGATED RESULT
            # ------------------------------------------------------------
            if order in {"asc", "desc"}:
                sort_column = value_column

                # COUNT does not retain the original value column.
                # The GROUP tool produces a count column instead.
                if aggregation == "count":
                    count_candidates = [
                        column
                        for column in current_df.columns
                        if str(column).lower() in {
                            "count",
                            "counts",
                            "count_" + str(value_column).lower(),
                            "n",
                        }
                    ]

                    if count_candidates:
                        sort_column = count_candidates[0]
                    else:
                        # Generic fallback: find a numeric column that is
                        # not the GROUP column.
                        numeric_candidates = [
                            column
                            for column in current_df.columns
                            if column != group_column
                            and pd.api.types.is_numeric_dtype(
                                current_df[column]
                            )
                        ]

                        if numeric_candidates:
                            sort_column = numeric_candidates[0]

                if sort_column not in current_df.columns:
                    raise ValueError(
                        f"Sort column '{sort_column}' not found after GROUP."
                    )

                current_df = current_df.sort_values(
                    by=sort_column,
                    ascending=(order == "asc"),
                    kind="stable",
                ).reset_index(drop=True)
            # Apply TOP/BOTTOM N after sorting.
            if limit is not None:
                current_df = current_df.head(limit).reset_index(drop=True)

        else:
            raise ValueError(f"Unknown TOOL: {tool}")

        print(current_df)

    return current_df


final_result = execute_plan(
    plan_steps,
    tables,
)


# ------------------------------------------------------------
# SCALAR AGGREGATION: overall COUNT without GROUP BY
# Example: "How many orders are there?"
# The planner intentionally has no group_column for this question,
# so SOURCE returns the source rows. Convert them into one scalar row.
# ------------------------------------------------------------
if (
    requirements.get("aggregation") == "count"
    and not requirements.get("group_column")
    and requirements.get("result_type") != "list"
):
    count_column = requirements.get("value_column")

    if count_column and count_column in final_result.columns:
        # Count distinct record IDs so duplicate rows do not inflate
        # the number of orders when an order identifier is available.
        total_count = int(final_result[count_column].nunique(dropna=True))
        result_label = (
            "Total_Orders" if "order" in question.lower()
            else "Total_Count"
        )
    else:
        # Fallback for datasets without a suitable record identifier.
        total_count = int(len(final_result))
        result_label = (
            "Total_Orders" if "order" in question.lower()
            else "Total_Count"
        )

    final_result = pd.DataFrame([{result_label: total_count}])
    print("\nSCALAR COUNT RESULT:", total_count)


# ------------------------------------------------------------
# SCALAR AGGREGATION: overall SUM without GROUP BY
# Example: "What is the total sales amount?"
# ------------------------------------------------------------
if (
    requirements.get("aggregation") == "sum"
    and not requirements.get("group_column")
    and requirements.get("result_type") != "list"
):
    sum_column = requirements.get("value_column")

    if not sum_column or sum_column not in final_result.columns:
        raise ValueError(
            "SUM was requested, but the selected numeric value column "
            f"was not found: {sum_column!r}. Check the detected value_column."
        )

    numeric_values = pd.to_numeric(
        final_result[sum_column], errors="coerce"
    )
    total_sum = numeric_values.sum(min_count=1)

    if pd.isna(total_sum):
        raise ValueError(
            f"Column '{sum_column}' contains no valid numeric values to sum."
        )

    question_lower = question.lower()
    if "sale" in question_lower or "revenue" in question_lower:
        result_label = "Total_Sales"
    else:
        result_label = f"Total_{sum_column}"

    final_result = pd.DataFrame([{result_label: float(total_sum)}])
    print("\nSCALAR SUM RESULT:", total_sum, "FROM COLUMN:", sum_column)


# ------------------------------------------------------------
# TARGETED TOP-N CUSTOMER SALES RECOVERY
# Handles questions such as:
#   "Which 5 customers generated the most sales?"
# This is intentionally isolated after the existing planner executes.
# It does not change the LLM, planner, JOIN/FILTER/GROUP tools, or other
# query paths. It uses only matching columns present in the loaded tables.
# ------------------------------------------------------------
_top_customer_match = re.search(
    r"\b(?:top\s+)?(\d+)\s+customers?\b.*\b(?:sales|revenue)\b|"
    r"\bwhich\s+(\d+)\s+customers?\b.*\b(?:sales|revenue)\b",
    question,
    flags=re.IGNORECASE,
)
if _top_customer_match:
    _top_n = int(_top_customer_match.group(1) or _top_customer_match.group(2))
    _customer_id_col = None
    _sales_col = None
    _orders_df = None
    _customer_lookup = None

    # Locate the transaction table by schema, not by filename.
    for _table_df in tables.values():
        _lower_to_actual = {str(c).strip().lower(): c for c in _table_df.columns}
        _candidate_customer_id = next(
            (_lower_to_actual[k] for k in ("customer_id", "client_id") if k in _lower_to_actual),
            None,
        )
        _candidate_sales = next(
            (_lower_to_actual[k] for k in ("sales_amount", "sales", "revenue", "total_sales") if k in _lower_to_actual),
            None,
        )
        if _candidate_customer_id and _candidate_sales:
            _orders_df = _table_df.copy()
            _customer_id_col = _candidate_customer_id
            _sales_col = _candidate_sales
            break

    if _orders_df is not None:
        _sales_values = pd.to_numeric(_orders_df[_sales_col], errors="coerce")
        _customer_work = _orders_df[[_customer_id_col]].copy()
        _customer_work["__sales_value__"] = _sales_values
        _top_customers = (
            _customer_work.dropna(subset=[_customer_id_col, "__sales_value__"])
            .groupby(_customer_id_col, as_index=False)["__sales_value__"]
            .sum()
            .rename(columns={"__sales_value__": _sales_col})
            .sort_values(_sales_col, ascending=False, kind="stable")
            .head(_top_n)
            .reset_index(drop=True)
        )

        # Add customer names only when a matching lookup table is available.
        for _table_df in tables.values():
            _lower_to_actual = {str(c).strip().lower(): c for c in _table_df.columns}
            _lookup_id = _lower_to_actual.get(str(_customer_id_col).lower())
            _lookup_name = next(
                (_lower_to_actual[k] for k in ("customer_name", "client_name") if k in _lower_to_actual),
                None,
            )
            if _lookup_id and _lookup_name:
                _customer_lookup = _table_df[[_lookup_id, _lookup_name]].drop_duplicates(subset=[_lookup_id])
                if _lookup_id != _customer_id_col:
                    _customer_lookup = _customer_lookup.rename(columns={_lookup_id: _customer_id_col})
                _top_customers = _top_customers.merge(
                    _customer_lookup,
                    on=_customer_id_col,
                    how="left",
                )
                break

        final_result = _top_customers

        # Preserve the explicit Top-N request so the later unlimited-ranking
        # tie handler does not collapse this result back to a single row.
        requirements["result_type"] = "ranked"
        requirements["group_column"] = _customer_id_col
        requirements["value_column"] = _sales_col
        requirements["aggregation"] = "sum"
        requirements["order"] = "desc"
        requirements["limit"] = _top_n

        print(f"\nTOP {_top_n} CUSTOMER SALES RESULT:")
        print(final_result)


# ------------------------------------------------------------
# TARGETED TWO-DIMENSIONAL RANKING: REGION + PRODUCT CATEGORY
# Example: "Which region and product category combination generated
# the highest sales?"
# Isolated post-processing: does not alter the existing planner, LLM,
# generic JOIN/GROUP tools, or unrelated query paths.
# ------------------------------------------------------------
_region_category_requested = bool(
    re.search(r"\bregions?\b", question, flags=re.IGNORECASE)
    and re.search(r"\bproduct\s+categor(?:y|ies)\b|\bcategor(?:y|ies)\b", question, flags=re.IGNORECASE)
    and re.search(r"\b(highest|most|top|greatest)\b", question, flags=re.IGNORECASE)
    and re.search(r"\b(sales|revenue)\b", question, flags=re.IGNORECASE)
)

if _region_category_requested:
    _sales_source = None
    _customer_source = None
    _product_source = None

    # Find tables by their actual schemas rather than filenames.
    for _table_name, _table_df in tables.items():
        _cols = {str(c).strip().lower(): c for c in _table_df.columns}

        _sales_key = next(
            (_cols[k] for k in ("sales_amount", "sales", "revenue", "total_sales") if k in _cols),
            None,
        )
        _customer_key = next(
            (_cols[k] for k in ("customer_id", "client_id") if k in _cols),
            None,
        )
        _product_key = next(
            (_cols[k] for k in ("product_id", "item_id") if k in _cols),
            None,
        )
        if _sales_key and _customer_key and _product_key:
            _sales_source = (_table_name, _table_df.copy(), _sales_key, _customer_key, _product_key)
            break

    for _table_name, _table_df in tables.items():
        _cols = {str(c).strip().lower(): c for c in _table_df.columns}
        _customer_key = next(
            (_cols[k] for k in ("customer_id", "client_id") if k in _cols),
            None,
        )
        _region_key = next(
            (_cols[k] for k in ("region", "region_name", "sales_region") if k in _cols),
            None,
        )
        if _customer_key and _region_key:
            _customer_source = (_table_name, _table_df[[_customer_key, _region_key]].copy(), _customer_key, _region_key)
            break

    for _table_name, _table_df in tables.items():
        _cols = {str(c).strip().lower(): c for c in _table_df.columns}
        _product_key = next(
            (_cols[k] for k in ("product_id", "item_id") if k in _cols),
            None,
        )
        _category_key = next(
            (_cols[k] for k in ("category", "product_category") if k in _cols),
            None,
        )
        if _product_key and _category_key:
            _product_source = (_table_name, _table_df[[_product_key, _category_key]].copy(), _product_key, _category_key)
            break

    if not (_sales_source and _customer_source and _product_source):
        raise ValueError(
            "Region + product category analysis needs a sales table with "
            "Customer_ID, Product_ID and Sales_Amount, a customer table with "
            "Region, and a product table with Category."
        )

    _sales_table_name, _sales_df, _sales_col, _sales_customer_key, _sales_product_key = _sales_source
    _customer_table_name, _customer_dim, _customer_dim_key, _region_col = _customer_source
    _product_table_name, _product_dim, _product_dim_key, _category_col = _product_source

    # Normalize dimension keys before joining and prevent duplicate lookup
    # rows from multiplying transaction sales.
    _customer_dim = _customer_dim.drop_duplicates(subset=[_customer_dim_key]).rename(
        columns={_customer_dim_key: "__customer_join_key__", _region_col: "Region"}
    )
    _product_dim = _product_dim.drop_duplicates(subset=[_product_dim_key]).rename(
        columns={_product_dim_key: "__product_join_key__", _category_col: "Category"}
    )

    _combo_work = _sales_df.copy()
    _combo_work["__sales_numeric__"] = pd.to_numeric(_combo_work[_sales_col], errors="coerce")
    _combo_work["__customer_join_key__"] = _combo_work[_sales_customer_key].astype(str).str.strip()
    _combo_work["__product_join_key__"] = _combo_work[_sales_product_key].astype(str).str.strip()
    _customer_dim["__customer_join_key__"] = _customer_dim["__customer_join_key__"].astype(str).str.strip()
    _product_dim["__product_join_key__"] = _product_dim["__product_join_key__"].astype(str).str.strip()

    _combo_work = _combo_work.merge(
        _customer_dim[["__customer_join_key__", "Region"]],
        on="__customer_join_key__", how="left", validate="many_to_one"
    ).merge(
        _product_dim[["__product_join_key__", "Category"]],
        on="__product_join_key__", how="left", validate="many_to_one"
    )

    _combo_work = _combo_work.dropna(subset=["Region", "Category", "__sales_numeric__"])
    if _combo_work.empty:
        raise ValueError("No matching sales records found after joining Region and Category dimensions.")

    final_result = (
        _combo_work.groupby(["Region", "Category"], as_index=False)["__sales_numeric__"]
        .sum()
        .rename(columns={"__sales_numeric__": "Total_Sales"})
        .sort_values("Total_Sales", ascending=False, kind="stable")
        .reset_index(drop=True)
    )
    final_result.insert(
        0, "Region_Category",
        final_result["Region"].astype(str) + " | " + final_result["Category"].astype(str)
    )

    # Keep the highest value and any exact ties; retain both dimensions
    # explicitly in the returned table.
    _highest_combo_sales = final_result["Total_Sales"].max()
    final_result = final_result.loc[
        final_result["Total_Sales"] == _highest_combo_sales
    ].reset_index(drop=True)

    requirements["result_type"] = "ranked"
    requirements["aggregation"] = "sum"
    requirements["group_column"] = "Region_Category"
    requirements["value_column"] = "Total_Sales"
    requirements["order"] = "desc"
    requirements["limit"] = None
    print("\nREGION + CATEGORY SALES RESULT:")
    print(final_result)


# ------------------------------------------------------------
# TIME-SERIES AGGREGATION: monthly / yearly / daily trends
# ------------------------------------------------------------
# This is deliberately isolated after normal plan execution. It does not
# modify the existing JOIN, FILTER, GROUP, or planner behavior.
# Example: "What is the monthly sales trend?"
question_lower_for_time_series = question.lower()
monthly_trend_requested = bool(
    re.search(r"\b(monthly|by\s+month|per\s+month|each\s+month)\b", question_lower_for_time_series)
    and re.search(r"\b(trend|sales|revenue|amount|total)\b", question_lower_for_time_series)
)
yearly_trend_requested = bool(
    re.search(r"\b(yearly|annually|by\s+year|per\s+year|each\s+year)\b", question_lower_for_time_series)
    and re.search(r"\b(trend|sales|revenue|amount|total)\b", question_lower_for_time_series)
)
daily_trend_requested = bool(
    re.search(r"\b(daily|by\s+day|per\s+day|each\s+day)\b", question_lower_for_time_series)
    and re.search(r"\b(trend|sales|revenue|amount|total)\b", question_lower_for_time_series)
)

time_series_grain = (
    "month" if monthly_trend_requested else
    "year" if yearly_trend_requested else
    "day" if daily_trend_requested else None
)

if time_series_grain:
    # Find a real date/time column in the executed result.
    date_column = None
    date_name_tokens = ("date", "time", "timestamp", "datetime")
    for candidate in final_result.columns:
        candidate_lower = str(candidate).lower()
        if not any(token in candidate_lower for token in date_name_tokens):
            continue
        parsed_candidate = pd.to_datetime(final_result[candidate], errors="coerce")
        if len(parsed_candidate) and parsed_candidate.notna().mean() >= 0.8:
            date_column = candidate
            break

    if date_column is None:
        raise ValueError(
            "A time trend was requested, but no usable date column was found "
            "in the query result. Check that the selected source table contains a date column."
        )

    # Prefer the metric already selected by the semantic parser. If it is
    # absent/invalid, infer it from the original question and result schema.
    trend_value_column = requirements.get("value_column")
    if (
        not trend_value_column
        or trend_value_column not in final_result.columns
        or not pd.api.types.is_numeric_dtype(final_result[trend_value_column])
    ):
        trend_value_column = choose_metric_column_from_question(tables, question)

    if (
        not trend_value_column
        or trend_value_column not in final_result.columns
        or not pd.api.types.is_numeric_dtype(final_result[trend_value_column])
    ):
        # Conservative fallback: choose a sales/revenue/amount-like numeric
        # column; do not silently sum IDs, quantities, or unrelated measures.
        metric_candidates = [
            column for column in final_result.select_dtypes(include=["number"]).columns
            if re.search(r"(sales|revenue|amount|turnover)", str(column), re.I)
            and not str(column).lower().endswith("_id")
        ]
        trend_value_column = metric_candidates[0] if metric_candidates else None

    if not trend_value_column:
        raise ValueError(
            "A time trend was requested, but no suitable numeric sales/revenue metric "
            "could be identified."
        )

    trend_work = final_result[[date_column, trend_value_column]].copy()
    trend_work[date_column] = pd.to_datetime(trend_work[date_column], errors="coerce")
    trend_work[trend_value_column] = pd.to_numeric(
        trend_work[trend_value_column], errors="coerce"
    )
    trend_work = trend_work.dropna(subset=[date_column, trend_value_column])

    if trend_work.empty:
        raise ValueError("No valid date and numeric sales records remain for the requested trend.")

    if time_series_grain == "month":
        trend_work["__Period__"] = trend_work[date_column].dt.to_period("M").dt.to_timestamp()
        period_label = "Month"
    elif time_series_grain == "year":
        trend_work["__Period__"] = trend_work[date_column].dt.to_period("Y").dt.to_timestamp()
        period_label = "Year"
    else:
        trend_work["__Period__"] = trend_work[date_column].dt.floor("D")
        period_label = "Date"

    final_result = (
        trend_work.groupby("__Period__", as_index=False)[trend_value_column]
        .sum()
        .rename(columns={"__Period__": period_label, trend_value_column: "Total_Sales"})
        .sort_values(period_label)
        .reset_index(drop=True)
    )
    requirements["value_column"] = "Total_Sales"
    print(
        f"\nTIME SERIES RESULT ({time_series_grain}): "
        f"{len(final_result)} periods using SUM({trend_value_column}) grouped by {date_column}"
    )


# ============================================================
# 11. ENTITY EXTRACTION FOR LISTED ENTITY QUERIES
# ============================================================

print(
    "\nENTITY CHECK:",
    requirements.get("entity"),
    requirements.get("entity_column"),
    requirements.get("group_column"),
    requirements.get("aggregation"),
)

if (
    requirements.get("entity_column")
    and requirements.get("group_column") is None
    and requirements.get("aggregation") is None
):
    entity_column = requirements["entity_column"]

    if entity_column in final_result.columns:
        final_result = final_result.drop_duplicates(
            subset=[entity_column]
        ).copy()


print(
    "\nRANK CHECK:",
    requirements.get("group_column"),
    requirements.get("aggregation"),
    requirements.get("order"),
    requirements.get("value_column"),
    "LIMIT=",
    requirements.get("limit"),
)

# ------------------------------------------------------------
# RAW TOP/BOTTOM LIST POST-PROCESSING
# ------------------------------------------------------------
if (
    requirements.get("result_type") == "list"
    and requirements.get("limit") is not None
    and requirements.get("sort_column")
    and requirements.get("order") in {"asc", "desc"}
    and not final_result.empty
):
    sort_column = requirements["sort_column"]
    if sort_column in final_result.columns:
        numeric_series = pd.to_numeric(final_result[sort_column], errors="coerce")
        final_result = (
            final_result.assign(__sort_value=numeric_series)
            .sort_values(
                by="__sort_value",
                ascending=(requirements["order"] == "asc"),
                kind="stable",
            )
            .drop(columns=["__sort_value"])
            .head(requirements["limit"])
            .reset_index(drop=True)
        )

# ------------------------------------------------------------
# UNLIMITED RANKING: retain all ties for the best aggregated value.
# COUNT uses the GROUP-generated Count column, not Purchase_ID.
# ------------------------------------------------------------
if (
    requirements.get("result_type") == "ranked"
    and requirements.get("group_column")
    and requirements.get("aggregation") in {
        "sum", "mean", "min", "max", "count"
    }
    and requirements.get("order") in {"asc", "desc"}
    and requirements.get("limit") is None
    and not final_result.empty
):
    if requirements.get("aggregation") == "count":
        count_column = next(
            (
                c for c in final_result.columns
                if str(c).lower() in {"count", "counts", "n"}
            ),
            None,
        )
        ranking_column = count_column
    else:
        ranking_column = requirements.get("value_column")

    if ranking_column and ranking_column in final_result.columns:
        best_value = final_result[ranking_column].iloc[0]
        final_result = final_result.loc[
            final_result[ranking_column] == best_value
        ].copy()

print(
    "\n================ RESULT AFTER ENTITY EXTRACTION ================\n"
)
print(final_result)


# ============================================================
# 12. DETERMINISTIC FINAL FORMATTING
# ============================================================

def format_value(value):
    if pd.isna(value):
        return "None"

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.2f}".rstrip("0").rstrip(".")

    return str(value)


def format_rows(result):
    lines = []

    for _, row in result.iterrows():
        parts = []
        for column in result.columns:
            parts.append(
                f"{column}: {format_value(row[column])}"
            )
        lines.append("- " + ", ".join(parts))

    return lines


def format_final_answer(question, result, requirements):
    if result is None or result.empty:
        return "No matching records found."

    result_type = requirements.get("result_type")
    aggregation = requirements.get("aggregation")
    order = requirements.get("order")
    limit = requirements.get("limit")
    group_column = requirements.get("group_column")
    value_column = requirements.get("value_column")

    rows = format_rows(result)

    if result_type == "list":
        return "The matching records are:\n" + "\n".join(rows)

    if result_type == "grouped_filtered":
        return "The matching groups are:\n" + "\n".join(rows)

    if result_type == "ranked":
        direction = "highest" if order == "desc" else "lowest"
        display_metric = value_column

        if aggregation == "count":
            count_column = next(
                (c for c in result.columns if str(c).lower() in {"count", "counts", "n"}),
                value_column,
            )
            display_metric = count_column

        if limit is not None:
            return (
                f"The {direction} {limit} result(s) by "
                f"{display_metric} are:\n" + "\n".join(rows)
            )

        if len(result) == 1:
            row = result.iloc[0]
            entity_value = (
                row[group_column]
                if group_column in result.columns
                else "the result"
            )
            metric_value = (
                row[display_metric]
                if display_metric in result.columns
                else None
            )
            if metric_value is not None:
                return (
                    f"The {direction} {group_column} is "
                    f"{entity_value}, with {display_metric} of "
                    f"{format_value(metric_value)}."
                )

        return (
            f"The {direction} results are:\n" +
            "\n".join(rows)
        )

    if result_type == "grouped":
        if aggregation == "sum":
            description = "total"
        elif aggregation == "mean":
            description = "average"
        elif aggregation == "count":
            description = "count"
        elif aggregation == "min":
            description = "minimum"
        elif aggregation == "max":
            description = "maximum"
        else:
            description = "calculated"

        return (
            f"The {description} results by {group_column} are:\n" +
            "\n".join(rows)
        )

    # Scalar / fallback
    return "The result is:\n" + "\n".join(rows)


final_answer = format_final_answer(
    question,
    final_result,
    requirements,
)

if _region_category_requested and not final_result.empty:
    _combo_lines = [
        f"Region: {row['Region']}, Category: {row['Category']}, Total Sales: {format_value(row['Total_Sales'])}"
        for _, row in final_result.iterrows()
    ]
    if len(_combo_lines) == 1:
        final_answer = "The highest-sales region and product category combination is:\n" + _combo_lines[0]
    else:
        final_answer = "The highest-sales region and product category combinations (tied) are:\n" + "\n".join(_combo_lines)

print("\n================ FINAL ANSWER ================\n")
print(final_answer)

print(
    "__STREAMLIT_RESULT__" +
    json.dumps({
        "answer": str(final_answer),
        "result": final_result.to_dict(orient="records"),
        "columns": final_result.columns.tolist()
    }, default=str)
)

print(final_result.columns)
print(final_result.dtypes)
print(final_result.shape)

def select_chart_type(result, question, requirements):
    categorical_cols = result.select_dtypes(
        include=["object", "category"]
    ).columns
    numeric_cols = result.select_dtypes(
        include=["number"]
    ).columns
    date_cols = []

    date_name_words = (
        "date",
        "time",
        "timestamp",
        "datetime",
    )

    for col in result.columns:
        col_lower = str(col).lower()

        if any(word in col_lower for word in date_name_words):
            parsed = pd.to_datetime(
                result[col],
                errors="coerce"
            )

            if parsed.notna().mean() >= 0.8:
                date_cols.append(col)

    question_lower = question.lower()

    result_type = requirements.get("result_type")

    # Distribution/frequency/spread queries are histogram requests.
    # Check this BEFORE LINE so a raw table containing both dates and
    # numeric values does not get misclassified as a time-series chart.
    distribution_words = re.search(
        r"\b(distribution|frequency|spread|histogram)\b",
        question_lower,
    )

    if distribution_words and len(numeric_cols) >= 1:
        return "HISTOGRAM"

    # Share/percentage/proportion queries are suitable for a small
    # categorical composition chart. "Distribution" is intentionally
    # excluded here because it has already been defined as HISTOGRAM.
    share_words = re.search(
        r"\b(share|percentage|proportion)\b",
        question_lower,
    )

    if (
        share_words
        and len(categorical_cols) >= 1
        and len(numeric_cols) >= 1
        and len(result) <= 8
        and len(date_cols) == 0
    ):
        return "PIE"

    temporal_words = re.search(
        r"\b(date|time|daily|monthly|yearly|trend|timeline|"
        r"over time|by date|by month|by year)\b",
        question_lower,
    )

    mentioned_date_column = False
    normalized_question = re.sub(r"[_\-]+", " ", question_lower)
    normalized_question = re.sub(r"\s+", " ", normalized_question).strip()

    for column in date_cols:
        normalized_column = re.sub(r"[_\-]+", " ", str(column)).lower()
        normalized_column = re.sub(r"\s+", " ", normalized_column).strip()
        if normalized_column and normalized_column in normalized_question:
            mentioned_date_column = True
            break

    if (
        (temporal_words or mentioned_date_column)
        and len(date_cols) >= 1
        and len(numeric_cols) >= 1
    ):
        return "LINE"

    # Raw/list lookups normally should not create charts, even when joins
    # leave unrelated numeric or date columns in the result.
    if result_type in {"list", "single"}:
        return "NONE"

    if len(categorical_cols) >= 1 and len(numeric_cols) >= 1:
        return "BAR"

    return "NONE"


chart_type = select_chart_type(final_result, question, requirements)
if time_series_grain and len(final_result) > 0:
    chart_type = "LINE"

print("CHART TYPE:", chart_type)

if chart_type == "NONE":
    print("NO CHART REQUIRED.")

def choose_chart_numeric_column(result, question):
    numeric_cols = list(
        result.select_dtypes(include=["number"]).columns
    )

    if not numeric_cols:
        return None

    question_lower = question.lower()

    # Prefer a numeric column whose normalized schema name occurs directly
    # in the question. Example: "order amounts" -> Order_Amount.
    for column in numeric_cols:
        normalized = re.sub(r"[_\-]+", " ", str(column)).lower()
        normalized = re.sub(r"\s+", " ", normalized).strip()
        if normalized and normalized in question_lower:
            return column

    # Generic business-language aliases.
    alias_groups = {
        "order amount": ["Order_Amount", "Amount", "Revenue", "Sales", "Value"],
        "purchase value": ["Purchase_Value", "Value", "Amount", "Revenue", "Sales"],
        "revenue": ["Revenue", "Order_Amount", "Amount", "Sales", "Value"],
        "sales": ["Sales", "Order_Amount", "Amount", "Revenue", "Value"],
        "amount": ["Order_Amount", "Amount", "Revenue", "Value", "Sales"],
        "value": ["Value", "Purchase_Value", "Order_Amount", "Amount", "Revenue", "Sales"],
    }

    for phrase, candidates in alias_groups.items():
        if phrase in question_lower:
            for candidate in candidates:
                for actual in numeric_cols:
                    if str(actual).lower() == candidate.lower():
                        return actual

    # Final deterministic fallback.
    return numeric_cols[0]


def plot_bar_chart(result):
    categorical_cols = result.select_dtypes(
        include=["object", "category"]
    ).columns

    numeric_cols = result.select_dtypes(
        include=["number"]
    ).columns

    if len(categorical_cols) == 0 or len(numeric_cols) == 0:
        return

    name_columns = [
        col for col in categorical_cols
        if "name" in col.lower()
    ]

    if name_columns:
        x_column = name_columns[0]
    else:
        x_column = categorical_cols[0]

    y_column = choose_chart_numeric_column(result, question)
    if y_column is None:
        return

    result = result.sort_values(
    by=y_column,
    ascending=False
).reset_index(drop=True)

    label_lengths = result[x_column].astype(str).str.len()

    if len(result) > 10 or label_lengths.max() > 15:
        plt.figure(figsize=(9, max(5, len(result) * 0.4)))
        plt.barh(
            result[x_column].astype(str),
            result[y_column]
        )

        for i, value in enumerate(result[y_column]):
            plt.text(
                value,
                i,
                format(value, ","),
                ha="left",
                va="center",
            )

        plt.xlabel(y_column)
        plt.ylabel(x_column)
        plt.title(f"{y_column} by {x_column}")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
    else:
        plt.figure(figsize=(9, 5))
        plt.bar(result[x_column].astype(str), result[y_column])

        for i, value in enumerate(result[y_column]):
            plt.text(
                i,
                value,
                format(value, ","),
                ha="center",
                va="bottom",
            )

        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.title(f"{y_column} by {x_column}")
        if len(result) > 5:
            plt.xticks(rotation=45)
        else:
            plt.xticks(rotation=0)
        plt.tight_layout()
        plt.show()


def plot_line_chart(result):
    date_column = next(
        (
            col for col in result.columns
            if pd.api.types.is_datetime64_any_dtype(result[col])
        ),
        None,
    )

    if date_column is None:
        date_name_words = (
            "date",
            "time",
            "timestamp",
            "datetime",
        )

        for col in result.columns:
            col_lower = str(col).lower()
            if not any(word in col_lower for word in date_name_words):
                continue

            parsed = pd.to_datetime(result[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                date_column = col
                result = result.copy()
                result[date_column] = parsed
                break

    if date_column is None:
        return

    numeric_cols = result.select_dtypes(include=["number"]).columns
    if len(numeric_cols) == 0:
        return

    y_column = choose_chart_numeric_column(result, question)
    if y_column is None:
        return

    result = result.sort_values(date_column)

    plt.figure(figsize=(9, 5))
    plt.plot(
        result[date_column],
        result[y_column],
        marker="o",
    )

    for x, y in zip(result[date_column], result[y_column]):
        plt.annotate(
            format(y, ","),
            (x, y),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center"
        )

    plt.xlabel(date_column)
    plt.ylabel(y_column)
    plt.title(f"{y_column} by {date_column}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_pie_chart(result):
    categorical_cols = result.select_dtypes(
        include=["object", "category"]
    ).columns

    numeric_cols = result.select_dtypes(
        include=["number"]
    ).columns

    if len(categorical_cols) == 0 or len(numeric_cols) == 0:
        return

    name_columns = [
        col for col in categorical_cols
        if "name" in col.lower()
    ]

    if name_columns:
        label_column = name_columns[0]
    else:
        label_column = categorical_cols[0]

    value_column = choose_chart_numeric_column(result, question)
    if value_column is None:
        return

    plt.figure(figsize=(7, 7))

    plt.pie(
        result[value_column],
        labels=result[label_column],
        autopct="%1.1f%%",
    )

    plt.title(f"{value_column} Share by {label_column}")
    plt.tight_layout()
    plt.show()


def plot_histogram(result):
    value_column = choose_chart_numeric_column(result, question)
    if value_column is None:
        return

    plt.figure(figsize=(9, 5))

    plt.hist(
        pd.to_numeric(result[value_column], errors="coerce").dropna(),
        bins=10,
        edgecolor="black",
    )

    plt.xlabel(value_column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {value_column}")
    plt.tight_layout()
    plt.show()


if chart_type == "BAR":
    plot_bar_chart(final_result)
elif chart_type == "LINE":
    plot_line_chart(final_result)
elif chart_type == "PIE":
    plot_pie_chart(final_result)
elif chart_type == "HISTOGRAM":
    plot_histogram(final_result)


elif chart_type == "NONE":
    print("No chart to display for this query.")