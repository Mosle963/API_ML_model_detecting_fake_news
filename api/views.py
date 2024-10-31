from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .serializers import OneentrySerializer
from .predicting import predict
from django_ratelimit.decorators import ratelimit
from django.utils.decorators import method_decorator

class predictAPI(APIView):
    @method_decorator(ratelimit(key='ip',rate='30/m'))
    def post(self, request):
        serializer = OneentrySerializer(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            prop = predict(data["text"])
            return Response(
                {"true_prob": prop}, status=status.HTTP_200_OK
            )
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)