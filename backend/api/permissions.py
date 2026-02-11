"""
Custom permissions for API access control.
"""
from rest_framework import permissions

from .models import UserProfile


class IsExperimenter(permissions.BasePermission):
    """
    Allow access only to users with experimenter role.
    """
    message = "Access restricted to experimenters."

    def has_permission(self, request, view):
        if not request.user or not request.user.is_authenticated:
            return False
        try:
            profile = request.user.profile
            return profile.role == UserProfile.ROLE_EXPERIMENTER
        except UserProfile.DoesNotExist:
            return False
