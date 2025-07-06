# Validating LLM Output Experiment

A comprehensive system for evaluating LLM (Large Language Model) output quality when extracting credit card transaction information from raw text. This project demonstrates enterprise-level validation techniques using multiple layers of verification.

## 🎯 The Challenge

When LLMs extract structured data from unstructured text, we need systematic validation because:
- LLMs can produce **malformed JSON**
- Extracted data might have **incorrect formats**
- Business rules might be **violated** (invalid card numbers)
- Data might be **partially missing** or **completely wrong**

## 🏗️ Multi-Layer Validation Strategy

### The Solution: 3-Layer Validation Pipeline

```
Raw Text → LLM → JSON → Pydantic → Business Logic → Ground Truth
   ↓         ↓      ↓        ↓           ↓              ↓
Input    Extract Parse   Validate   Domain Check   Accuracy
```

## 🔧 Layer 1: Pydantic Schema Validation

### Field Validation Strategy

The system validates that each field has the **right format** and **meaningful content**:

**Card Number Validation:**
Normalizes format (removes spaces/dashes), validates only digits, ensures 11-19 digit length.

**Date Format Validation:**
Handles multiple common date formats and validates the string can be parsed as a valid date.

**Name Validation:**
Ensures minimum length (2+ characters) and validates character set (letters, spaces, punctuation).

**Transaction Amount Validation:**
Handles currency formats, validates numeric value, ensures no negative amounts.

**Merchant Name Validation:**
Basic format validation with minimum meaningful length requirements.

## ⚙️ Layer 2: Business Logic Validation

### Domain-Specific Rules

**Luhn Algorithm for Credit Cards:**
The Luhn algorithm is a **domain-specific validation** that goes beyond format checking. While Pydantic ensures a card number has the right format (correct length, only digits), the Luhn algorithm validates that it's a **mathematically valid credit card number**.

This is crucial because:
- Credit card numbers follow a specific mathematical pattern
- Random digit sequences that look like card numbers will fail Luhn validation
- It catches transcription errors and fake numbers
- It's an industry standard used by all major card networks

## 📊 Layer 3: Ground Truth Comparison

### Accuracy Measurement
After ensuring the extracted data is well-formatted and valid, the system compares it against known correct answers. Both extracted and expected values are normalized to ensure fair comparison (removing formatting differences like spaces or dashes).

## 🚨 Comprehensive Error Classification

The system categorizes every possible failure mode:

| Validation Layer | Error Type | Description | Software Impact |
|------------------|------------|-------------|-----------------|
| **JSON Parsing** | `parsing_error` | Malformed LLM output | Cannot proceed with validation |
| **Pydantic Schema** | `schema_error` | Type/format validation failed | Data structure is invalid |
| **Business Logic** | `luhn_error` | Card number fails Luhn check | Extracted data is invalid |
| **Business Logic** | `missing_card_error` | No card number extracted | Incomplete extraction |
| **Ground Truth** | `mismatch_error` | Extracted ≠ Expected | Incorrect extraction |

## 🔄 Validation Processing Flow

The evaluation process follows these steps:

1. **LLM Extraction + JSON Parsing** - Convert raw text to structured data
2. **Check for Missing Critical Data** - Identify incomplete extractions
3. **Business Logic Validation** - Apply domain-specific rules (Luhn algorithm)
4. **Ground Truth Comparison** - Compare against known correct answers
5. **Success Recording** - Track successful validations

Each step can fail independently, providing specific error categorization for debugging and improvement.

## 🚀 Setup

### Prerequisites
Install required dependencies:
```bash
pip install -r requirements.txt
```

### Environment Configuration
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Data Requirements
Ensure your CSV dataset has the following columns:
- `text`: Raw text containing transaction information
- `card_number`: Ground truth card number for validation

## 📋 Running Parameters

### Command Line Options

**Basic Usage:**
```bash
python llm_output_test.py
```

**Available Parameters:**

- `--num_samples` (int, default=10): Number of samples to evaluate from the dataset
- `--data_path` (str, default="data/dummy_data.csv"): Path to your CSV dataset
- `--full_report` (flag): Enable detailed error breakdown and comprehensive metrics

### Usage Examples

**Quick Validation:**
```bash
python llm_output_test.py --num_samples 20
```
Output: Simple accuracy percentage

**Detailed Analysis:**
```bash
python llm_output_test.py --num_samples 50 --full_report
```
Output: Comprehensive breakdown with error categorization

**Custom Dataset:**
```bash
python llm_output_test.py --data_path "data/custom_data.csv" --full_report
```

## 📈 Output Analysis

### Simple Mode Output
```
Accuracy: 87.50%
```

### Full Report Output
```
==================================================
LLM OUTPUT EVALUATION SUMMARY
==================================================
Total Samples: 50
Accuracy: 87.50%

Error Breakdown:
  Parsing Errors: 2.00%      # JSON malformed
  Schema Errors: 4.00%       # Pydantic validation failed
  Luhn Errors: 6.00%         # Invalid card numbers
  Mismatch Errors: 0.50%     # Wrong extraction
  Missing Card Errors: 0.00% # No card found
```
