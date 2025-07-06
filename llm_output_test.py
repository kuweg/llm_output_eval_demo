import re
from argparse import ArgumentParser
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSerializable
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

load_dotenv()


def normalize_card_number(card_number: str) -> str:
    """
    Normalize card number by removing spaces and dashes
    to be sure that comparison is correct.

    Args:
        card_number (str): Card number to normalize

    Returns:
        str: Normalized card number
    """
    return card_number.replace(" ", "").replace("-", "")


def luhn_check(card_number: str) -> bool:
    """
    Check if card number is valid using Luhn algorithm.

    Args:
        card_number (str): Card number to check

    Returns:
        bool: True if card number is valid, False otherwise
    """
    digits = [int(d) for d in card_number if d.isdigit()]
    checksum = 0
    parity = len(digits) % 2
    for i, digit in enumerate(digits):
        if i % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


class TransactionInfo(BaseModel):
    """Pydantic model with comprehensive validation"""

    card_number: str | None = Field(None, description="Credit card number")
    date: str | None = Field(None, description="Transaction date")
    full_name: str | None = Field(None, description="Full name of cardholder")
    transaction_amount: str | None = Field(None, description="Transaction amount")
    merchant: str | None = Field(None, description="Merchant name")

    @field_validator("card_number")
    def validate_card_number(cls, v):
        """Validate card number format and length"""
        if v is None:
            return None

        if not isinstance(v, str) or not v.strip():
            raise ValueError("Card number must be a non-empty string")

        normalized = normalize_card_number(v)

        if not normalized.isdigit():
            raise ValueError("Card number must contain only digits, spaces, or dashes")

        if len(normalized) < 11 or len(normalized) > 19:
            raise ValueError("Card number must be between 11-19 digits")

        return normalized

    @field_validator("date")
    def validate_date(cls, v):
        """Validate date format"""
        if v is None:
            return None

        if not v or not v.strip():
            raise ValueError("Date cannot be empty")

        date_formats = [
            "%Y-%m-%d",
            "%m/%d/%Y",
            "%d/%m/%Y",
            "%Y/%m/%d",
            "%m-%d-%Y",
            "%d-%m-%Y",
        ]

        for fmt in date_formats:
            try:
                datetime.strptime(v, fmt)
                return v
            except ValueError:
                continue

        raise ValueError(f"Invalid date format: {v}")

    @field_validator("full_name")
    def validate_name(cls, v):
        """Validate name format"""
        if v is None:
            return None

        if not v or len(v.strip()) < 2:
            raise ValueError("Name must be at least 2 characters")

        if not re.match(r"^[a-zA-Z\s\.\-\']+$", v):
            raise ValueError("Name contains invalid characters")

        return v.strip()

    @field_validator("transaction_amount")
    def validate_amount(cls, v):
        """Validate transaction amount format"""
        if v is None:
            return None

        if not v or not v.strip():
            raise ValueError("Transaction amount cannot be empty")

        cleaned = re.sub(r"[^\d\.\-]", "", v)

        try:
            amount = float(cleaned)
            if amount < 0:
                raise ValueError("Transaction amount cannot be negative")
            return v
        except ValueError:
            raise ValueError(f"Invalid transaction amount format: {v}")

    @field_validator("merchant")
    def validate_merchant(cls, v):
        """Validate merchant name"""
        if v is None:
            return None

        if not v or len(v.strip()) < 2:
            raise ValueError("Merchant name must be at least 2 characters")

        return v.strip()


def evaluate_llm_output(
    chain: RunnableSerializable,
    df: pd.DataFrame,
    num_samples: int = 10,
    full_report: bool = False,
) -> tuple[dict[str, int], dict[int, str]]:
    """
    Evaluate LLM output quality with comprehensive validation

    Args:
        chain: The LLM chain to evaluate
        df: The dataframe to test
        num_samples: The number of samples to test

    Returns:
        Dict with detailed metrics and analysis
    """

    if num_samples > len(df):
        raise ValueError(
            "num_samples must be less than the number of rows in the dataframe"
        )

    test_df = df.sample(num_samples)

    # Metrics tracking
    stats = {
        "total_processed": 0,
        "parsing_errors": 0,
        "schema_errors": 0,
        "luhn_errors": 0,
        "mismatch_errors": 0,
        "missing_card_errors": 0,
        "successful_validations": 0,
    }

    # Here will be stored the error types by id of the sample
    error_types = {}

    for i, (text, card_number) in enumerate(
        tqdm(
            zip(test_df["text"], test_df["card_number"]),
            total=num_samples,
            desc="Evaluating LLM Output",
        )
    ):
        stats["total_processed"] += 1

        try:
            llm_output = chain.invoke({"text": text})

            if llm_output.card_number is None:
                stats["missing_card_errors"] += 1
                error_types[i] = "missing_card_error"
                continue

            if not luhn_check(llm_output.card_number):
                stats["luhn_errors"] += 1
                error_types[i] = "luhn_error"
                continue

            normalized_llm = normalize_card_number(llm_output.card_number)
            normalized_expected = normalize_card_number(str(card_number))

            if normalized_llm != normalized_expected:
                stats["mismatch_errors"] += 1
                error_types[i] = "mismatch_error"
                continue

            stats["successful_validations"] += 1
            error_types[i] = None

        except OutputParserException as e:
            stats["parsing_errors"] += 1
            error_types[i] = "parsing_error"
            continue
        except Exception as e:
            stats["schema_errors"] += 1
            error_types[i] = "schema_error"
            continue

    # Calculate metrics
    total = stats["total_processed"]
    successful = stats["successful_validations"]

    if full_report:
        metrics = {
            "total_processed": total,
            "accuracy": successful / total if total > 0 else 0,
            "error_breakdown": {
                "parsing_errors": stats["parsing_errors"] / total if total > 0 else 0,
                "schema_errors": stats["schema_errors"] / total if total > 0 else 0,
                "luhn_errors": stats["luhn_errors"] / total if total > 0 else 0,
                "mismatch_errors": stats["mismatch_errors"] / total if total > 0 else 0,
                "missing_card_errors": (
                    stats["missing_card_errors"] / total if total > 0 else 0
                ),
            },
            "raw_counts": stats,
        }

        # Print summary
        print("\n" + "=" * 50)
        print("LLM OUTPUT EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Total Samples: {total}")
        print(f"Accuracy: {metrics['accuracy']:.2%}")
        print("\nError Breakdown:")
        for error_type, rate in metrics["error_breakdown"].items():
            print(f"  {error_type.replace('_', ' ').title()}: {rate:.2%}")
    else:
        print(f"Accuracy: {successful / total if total > 0 else 0:.2%}")

    return stats, error_types


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="data/dummy_data.csv")
    parser.add_argument("--full_report", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.data_path)

    extractor_prompt = """
[role]
You are a helpful assistant that extracts entities from the text.
[task]
Locate the entities in the text.
Entities are:
- Card number (if present)
- Date (if present)
- Full name (if present)
- Transaction amount (if present)
- Merchant (if present)
[input]
{text}

[output]
return the entities in the following json format. Only include fields that are present in the text:
{{
    "card_number": "1234567890",
    "date": "2025-01-01",
    "full_name": "John Doe",
    "transaction_amount": "100.00",
    "merchant": "Walmart"
}}
"""

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    output_parser = PydanticOutputParser(pydantic_object=TransactionInfo)
    prompt = PromptTemplate.from_template(extractor_prompt)
    card_extraction_chain = prompt | llm | output_parser

    results, error_types = evaluate_llm_output(
        chain=card_extraction_chain,
        df=df,
        num_samples=args.num_samples,
        full_report=args.full_report,
    )

    print(f"\nRaw stats: {results}")
    print(f"\nError types: {error_types}")
