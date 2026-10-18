from django.test import TestCase, Client
from rest_framework import status
import json

class SalaryPredictorAPITests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_single_prediction_success(self):
        response = self.client.post(
            '/api/predict/',
            data=json.dumps({'years_experience': 5.0}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('predicted_salary', data)
        self.assertIn('confidence_interval_95', data)
        self.assertEqual(data['currency'], 'USD')
        self.assertEqual(data['experience_tier'], 'Senior Professional')
        self.assertGreater(data['predicted_salary'], 60000)
        self.assertLess(data['predicted_salary'], 90000)

    def test_single_prediction_negative_experience(self):
        response = self.client.post(
            '/api/predict/',
            data=json.dumps({'years_experience': -3.5}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        data = response.json()
        self.assertIn('error', data)

    def test_single_prediction_invalid_type(self):
        response = self.client.post(
            '/api/predict/',
            data=json.dumps({'years_experience': 'five_years'}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)

    def test_batch_prediction_success(self):
        payload = {
            'records': [
                {'years_experience': 1.2},
                {'years_experience': 4.0},
                {'years_experience': 10.5}
            ]
        }
        response = self.client.post(
            '/api/batch-predict/',
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data['count'], 3)
        self.assertEqual(len(data['predictions']), 3)

    def test_metrics_endpoint(self):
        response = self.client.get('/api/metrics/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertIn('coefficients', data)
        self.assertIn('performance_metrics', data)
        self.assertIn('limitations', data)
        self.assertIn('r2', data['performance_metrics']['test'])

    def test_health_check_endpoint(self):
        response = self.client.get('/api/health/')
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data['status'], 'healthy')
        self.assertTrue(data['model_loaded'])

    def test_dashboard_renders(self):
        from django.test import RequestFactory
        factory = RequestFactory()
        request = factory.get('/')
        from predictor.views import dashboard_view
        response = dashboard_view(request)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Salary Prediction Engine', response.content)

