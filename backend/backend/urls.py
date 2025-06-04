from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from api.views import CreateUserView, GenerateIntervalsView, SecureAudioView, ClearUserAudioView
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),
    path("api/user/register/", CreateUserView.as_view(), name="register"),
    path("api/token/", TokenObtainPairView.as_view(), name="get_token"),
    path("api/token/refresh/", TokenRefreshView.as_view(), name="refresh"),
    path("api/training/generate-intervals/", GenerateIntervalsView.as_view(), name="generate_intervals"),
    path("api/training/audio/<str:user_id>/<str:filename>", SecureAudioView.as_view(), name="secure_audio"),
    path("api/training/clear-audio/", ClearUserAudioView.as_view(), name="clear_audio"),
    path("api-auth/", include("rest_framework.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)