from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.shortcuts import render
from .serializers import (
    SalaryPredictionInputSerializer,
    BatchPredictionInputSerializer,
    SalaryPredictionOutputSerializer
)
from .ml_service import ModelService


class SalaryPredictView(APIView):
    """
    POST /api/predict/
    Accepts: {"years_experience": 4.5}
    Returns: Single salary prediction with 95% confidence bounds
    """
    def post(self, request):
        serializer = SalaryPredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"error": "Validation Error", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )
        
        years = serializer.validated_data['years_experience']
        try:
            service = ModelService.get_instance()
            result = service.predict(years)
            out_serializer = SalaryPredictionOutputSerializer(result)
            return Response(out_serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": "Prediction Error", "message": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class SalaryBatchPredictView(APIView):
    """
    POST /api/batch-predict/
    Accepts: {"records": [{"years_experience": 1.5}, {"years_experience": 7.0}]}
    Returns: Batch salary predictions
    """
    def post(self, request):
        serializer = BatchPredictionInputSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(
                {"error": "Validation Error", "details": serializer.errors},
                status=status.HTTP_400_BAD_REQUEST
            )

        records = serializer.validated_data['records']
        years_list = [item['years_experience'] for item in records]
        
        try:
            service = ModelService.get_instance()
            predictions = service.predict_batch(years_list)
            return Response({
                "count": len(predictions),
                "predictions": predictions
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": "Batch Prediction Error", "message": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class ModelMetricsView(APIView):
    """
    GET /api/metrics/
    Returns full model metadata, coefficients, train/test evaluation metrics, and baseline comparisons
    """
    def get(self, request):
        try:
            service = ModelService.get_instance()
            return Response(service.metadata, status=status.HTTP_200_OK)
        except Exception as e:
            return Response(
                {"error": "Failed to retrieve metrics", "message": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )


class HealthCheckView(APIView):
    """
    GET /api/health/
    Health check endpoint for monitoring
    """
    def get(self, request):
        try:
            service = ModelService.get_instance()
            is_loaded = service._pipeline is not None
            return Response({
                "status": "healthy" if is_loaded else "unhealthy",
                "model_loaded": is_loaded,
                "model_name": service.metadata.get("model_name"),
                "framework": service.metadata.get("framework")
            }, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                "status": "unhealthy",
                "message": str(e)
            }, status=status.HTTP_503_SERVICE_UNAVAILABLE)


def dashboard_view(request):
    """
    Renders interactive web dashboard
    """
    service = ModelService.get_instance()
    metadata = service.metadata
    return render(request, "predictor/dashboard.html", {
        "metadata": metadata,
        "metrics": metadata.get("performance_metrics", {}),
        "coefficients": metadata.get("coefficients", {}),
        "limitations": metadata.get("limitations", [])
    })
