from numbers import Real
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

NUTRITION_REFERENCE_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "nutrition"
    / "nutrition_reference.csv"
)


REQUIRED_COLUMNS = [
    "foodseg103_class_id",
    "class_name",
    "calories_kcal",
    "protein_g",
    "carb_g",
    "fat_total_g",
    "fiber_g",
]


def load_nutrition_reference(
    reference_path: Path = NUTRITION_REFERENCE_PATH,
) -> pd.DataFrame:
    """Load and validate the processed nutrition reference."""

    reference_path = Path(reference_path)

    if not reference_path.exists():
        raise FileNotFoundError(
            f"Nutrition reference not found: {reference_path}"
        )

    reference = pd.read_csv(reference_path)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in reference.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Nutrition reference is missing columns: {missing_columns}"
        )

    if reference["foodseg103_class_id"].isna().any():
        raise ValueError(
            "Nutrition reference contains missing FoodSeg103 class IDs."
        )

    if reference["foodseg103_class_id"].duplicated().any():
        raise ValueError(
            "Nutrition reference contains duplicate FoodSeg103 class IDs."
        )

    return reference


def lookup_food(
    foodseg103_class_id: int,
    reference: pd.DataFrame,
) -> pd.Series:
    """Return the nutrition record for one FoodSeg103 class."""

    matches = reference[
        reference["foodseg103_class_id"] == foodseg103_class_id
    ]

    if matches.empty:
        raise KeyError(
            f"FoodSeg103 class ID not found in nutrition reference: "
            f"{foodseg103_class_id}"
        )

    return matches.iloc[0]


def calculate_nutrition(
    foodseg103_class_id: int,
    grams: float,
    reference: pd.DataFrame,
) -> dict:
    """Calculate nutrition for a food quantity given in grams."""

    if not isinstance(grams, Real):
        raise TypeError("Grams must be a numeric value.")

    if grams <= 0:
        raise ValueError("Grams must be greater than 0.")

    food = lookup_food(foodseg103_class_id, reference)

    scale = grams / 100.0

    result = {
        "foodseg103_class_id": int(food["foodseg103_class_id"]),
        "class_name": food["class_name"],
        "grams": float(grams),
        "calories_kcal": (
            food["calories_kcal"] * scale
            if pd.notna(food["calories_kcal"])
            else None
        ),
        "protein_g": (
            food["protein_g"] * scale
            if pd.notna(food["protein_g"])
            else None
        ),
        "carb_g": (
            food["carb_g"] * scale
            if pd.notna(food["carb_g"])
            else None
        ),
        "fat_total_g": (
            food["fat_total_g"] * scale
            if pd.notna(food["fat_total_g"])
            else None
        ),
        "fiber_g": (
            food["fiber_g"] * scale
            if pd.notna(food["fiber_g"])
            else None
        ),
    }

    return result


def calculate_meal_nutrition(
    foods: list[dict],
    reference: pd.DataFrame,
) -> dict:
    """Calculate nutrition totals for multiple foods in a meal."""

    if not foods:
        raise ValueError("Meal must contain at least one food.")

    food_results = []

    for food in foods:
        if "foodseg103_class_id" not in food:
            raise ValueError(
                "Each food must contain 'foodseg103_class_id'."
            )

        if "grams" not in food:
            raise ValueError(
                "Each food must contain 'grams'."
            )

        result = calculate_nutrition(
            foodseg103_class_id=food["foodseg103_class_id"],
            grams=food["grams"],
            reference=reference,
        )

        food_results.append(result)

    totals = {
        "calories_kcal": 0.0,
        "protein_g": 0.0,
        "carb_g": 0.0,
        "fat_total_g": 0.0,
        "fiber_g": 0.0,
    }

    for result in food_results:
        for nutrient in totals:
            value = result[nutrient]

            if value is not None:
                totals[nutrient] += float(value)

    return {
        "foods": food_results,
        "totals": totals,
    }