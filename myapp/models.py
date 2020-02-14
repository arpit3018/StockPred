from django.db import models

# Create your models here.
class Company(models.Model):
    comp_name = models.TextField(max_length=255)
    industry = models.TextField(max_length=255)
    symbol = models.TextField(max_length=255)