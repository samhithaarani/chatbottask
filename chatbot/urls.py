from django.urls import path
from . import views  

urlpatterns = [
    path('', views.index, name='index'),
    path('chatbot/get_response/', views.chatbot_response, name='chatbot_response'),
]
