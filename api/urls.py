from django.urls import path
from .views import predictAPI
urlpatterns = [
    path("predict/", predictAPI.as_view(), name="predict"),
]
