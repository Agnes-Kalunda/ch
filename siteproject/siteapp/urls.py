from django.urls import path
from . import views

urlpatterns =[
    path('',views.home,name='home'),
    path('navbar/',views.navbar,name='navbar'),
    path('footer/',views.footer,name= 'footer'),
    path('base/', views.base, name='base')
    
]