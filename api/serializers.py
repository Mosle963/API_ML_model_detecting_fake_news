from rest_framework import serializers

class OneentrySerializer(serializers.Serializer):
    text = serializers.CharField()