from pydantic import BaseModel, Field, field_validator, computed_field
from datetime import date, time
from typing import Annotated
import numpy as np
from model.predict import artifacts
from enum import Enum


cat2idx = artifacts['cat2idx']
categories = artifacts['categories']

CategoryEnum = Enum(
    "CategoryEnum",
    {f"category_{i}": c for i, c in enumerate(categories)},
    type=str
)

class UserInput(BaseModel):
    amount: Annotated[float, Field(..., description='How much traction cardholder did')]
    category: str
    transaction_date: Annotated[date, Field(..., description=' Transaction date')]
    transaction_time: Annotated[time, Field(..., description=' Time when trasaction happends')]
    date_of_birth: Annotated[date, Field(..., description='date of birth of the cardholder')]
    city_population: Annotated[int, Field(..., description='city population where the transaction happend')]
    customer_lat: Annotated[float, Field(..., description='cardholder latitude')]
    customer_lon: Annotated[float, Field(..., description='cardholder longitude')]
    merchant_lat: Annotated[float, Field(..., description='merchant latitude')]
    merchant_lon: Annotated[float, Field(..., description='merchant longitude')]

    @computed_field
    @property
    def cat2id(self)->int:
        cat2id = int(cat2idx[self.category])
        return cat2id

    @computed_field
    @property
    def hour(self)->int:
         return self.transaction_time.hour

    @computed_field
    @property
    def day_of_week(self)->int:
        return self.transaction_date.weekday()

    @computed_field
    @property
    def is_weekend(self)->int:
        return int(self.day_of_week >= 5)

    @computed_field
    @property
    def age(self)->int:
        return int(((self.transaction_date - self.date_of_birth).days) // 365.25)

    @computed_field
    @property
    def distance_km(self)->float:
        lon1, lat1, lon2, lat2 = map(np.radians, [self.customer_lat, self.customer_lon, self.merchant_lat, self.merchant_lon])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return round(float(6367 * c), 2)

    @computed_field
    @property
    def is_online(self)->int:
        return int(self.category.endswith('_net'))

    @field_validator('category')
    @classmethod
    def check_category(cls, value)->str:
        if value not in categories:
            raise ValueError('Invalid category')
        return value
