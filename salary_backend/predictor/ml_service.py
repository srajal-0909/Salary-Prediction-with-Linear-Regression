import json
import logging
import joblib
import pandas as pd
from django.conf import settings

logger = logging.getLogger(__name__)

class ModelService:
    _instance = None
    _pipeline = None
    _metadata = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
            cls._instance._load_artifacts()
        return cls._instance

    def _load_artifacts(self):
        try:
            model_path = getattr(settings, 'MODEL_PATH')
            metadata_path = getattr(settings, 'METADATA_PATH')
            
            logger.info(f"Loading ML model pipeline from: {model_path}")
            self._pipeline = joblib.load(model_path)
            
            logger.info(f"Loading metadata from: {metadata_path}")
            with open(metadata_path, 'r') as f:
                self._metadata = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load ML artifacts: {e}", exc_info=True)
            raise RuntimeError(f"Could not load ML artifacts: {e}")

    @property
    def metadata(self):
        return self._metadata

    def get_experience_tier(self, years: float) -> str:
        if years < 2.0:
            return "Entry Level / Junior"
        elif years < 5.0:
            return "Mid-Level Professional"
        elif years < 9.0:
            return "Senior Professional"
        elif years < 14.0:
            return "Staff / Principal Lead"
        else:
            return "Executive / Director Level"

    def predict(self, years_experience: float) -> dict:
        df_input = pd.DataFrame({'YearsExperience': [years_experience]})
        raw_pred = float(self._pipeline.predict(df_input)[0])
        
        # Calculate 95% approximate prediction interval using test RMSE
        test_rmse = self._metadata['performance_metrics']['test']['rmse']
        margin = 1.96 * test_rmse
        lower_bound = max(0.0, raw_pred - margin)
        upper_bound = raw_pred + margin

        return {
            "years_experience": float(years_experience),
            "predicted_salary": round(raw_pred, 2),
            "formatted_salary": f"${raw_pred:,.2f}",
            "currency": "USD",
            "experience_tier": self.get_experience_tier(years_experience),
            "confidence_interval_95": {
                "lower": round(lower_bound, 2),
                "upper": round(upper_bound, 2),
                "margin_error": round(margin, 2)
            },
            "model_version": self._metadata.get('model_name', 'Linear Regression Salary Predictor v1.0')
        }

    def predict_batch(self, years_list: list) -> list:
        results = []
        for yrs in years_list:
            results.append(self.predict(yrs))
        return results
