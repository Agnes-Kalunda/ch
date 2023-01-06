from django.shortcuts import render

# Create your views here.
def home(request):
    return render (request, 'home.html')

def navbar(request):
    return render (request, 'navbar.html' )

def base(request):
    return render (request, 'base.html')

def footer(request):
    return render (request, 'footer.html')