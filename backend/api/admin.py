from django.contrib import admin
from .models import UserProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ["user", "role", "level", "experience_points"]
    list_editable = ["role"]
    list_filter = ["role"]
