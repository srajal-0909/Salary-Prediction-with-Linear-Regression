from django.urls import path
from .views import (
    SalaryPredictView,
    SalaryBatchPredictView,
    ModelMetricsView,
    HealthCheckView,
    dashboard_view
)

urlpatterns = [
    path('', dashboard_view, name='dashboard'),
    path('api/predict/', SalaryPredictView.as_view(), name='api-predict'),
    path('api/batch-predict/', SalaryBatchPredictView.as_view(), name='api-batch-predict'),
    path('api/metrics/', ModelMetricsView.as_view(), name='api-metrics'),
    path('api/health/', HealthCheckView.as_view(), name='api-health'),
]
