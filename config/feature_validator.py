from model.predict import artifacts

feature_col = artifacts['features']

EXPECTED_FEATURES = [
    "amt",
    "category_idx",
    "hour",
    "day_of_week",
    "is_weekend",
    "age",
    "distance_km",
    "city_pop",
    "is_online",
]

if not feature_col == EXPECTED_FEATURES:
     raise ValueError(f'Expected: {EXPECTED_FEATURES}\n But get: {feature_col}')
