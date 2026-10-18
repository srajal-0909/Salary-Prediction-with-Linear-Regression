from rest_framework import serializers

class SalaryPredictionInputSerializer(serializers.Serializer):
    years_experience = serializers.FloatField(
        required=True,
        min_value=0.0,
        max_value=50.0,
        help_text="Total years of professional experience (0.0 to 50.0 years)"
    )

    def validate_years_experience(self, value):
        if value < 0:
            raise serializers.ValidationError("Years of experience cannot be negative.")
        if value > 50:
            raise serializers.ValidationError("Years of experience cannot exceed 50 years.")
        return value


class BatchPredictionItemSerializer(serializers.Serializer):
    years_experience = serializers.FloatField(required=True, min_value=0.0, max_value=50.0)


class BatchPredictionInputSerializer(serializers.Serializer):
    records = serializers.ListField(
        child=BatchPredictionItemSerializer(),
        allow_empty=False,
        max_length=100,
        help_text="List of candidate experience records (up to 100)"
    )


class SalaryPredictionOutputSerializer(serializers.Serializer):
    years_experience = serializers.FloatField()
    predicted_salary = serializers.FloatField()
    formatted_salary = serializers.CharField()
    currency = serializers.CharField(default="USD")
    experience_tier = serializers.CharField()
    confidence_interval_95 = serializers.DictField()
    model_version = serializers.CharField()
