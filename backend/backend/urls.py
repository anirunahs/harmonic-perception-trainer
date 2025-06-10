from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from api.views import (CreateUserView, GenerateIntervalsView, SecureAudioView, ClearUserAudioView,
                       UserProfileView, CreateTestSessionView, SubmitAnswerView, VocalRangeSetupView,
                       TestSessionListView, TestSessionDetailView, UserAchievementsView, 
                       GenerateSingleNoteView, TestingSecureAudioView, ClearTestingAudioView)
from rest_framework_simplejwt.views import TokenObtainPairView, TokenRefreshView

urlpatterns = [
    path('admin/', admin.site.urls),

    path("api/user/register/", CreateUserView.as_view(), name="register"),
    path("api/token/", TokenObtainPairView.as_view(), name="get_token"),
    path("api/token/refresh/", TokenRefreshView.as_view(), name="refresh"),

    path("api/user/profile/", UserProfileView.as_view(), name="user_profile"),
    path("api/user/achievements/", UserAchievementsView.as_view(), name="user_achievements"),
    path("api/user/vocal-range/", VocalRangeSetupView.as_view(), name="vocal_range_setup"),

    path("api/training/generate-intervals/", GenerateIntervalsView.as_view(), name="generate_intervals"),
    path("api/training/audio/<str:user_id>/<str:filename>", SecureAudioView.as_view(), name="secure_audio"),
    path("api/training/clear-audio/", ClearUserAudioView.as_view(), name="clear_audio"),

    path("api/testing/create-session/", CreateTestSessionView.as_view(), name="create_test_session"),
    path("api/testing/submit-answer/", SubmitAnswerView.as_view(), name="submit_answer"),
    path("api/testing/sessions/", TestSessionListView.as_view(), name="test_sessions"),
    path("api/testing/sessions/<int:pk>/", TestSessionDetailView.as_view(), name="test_session_detail"),

    path("api/testing/generate-note/", GenerateSingleNoteView.as_view(), name="generate_single_note"),
    path("api/testing/audio/<str:user_id>/<str:filename>", TestingSecureAudioView.as_view(), name="testing_secure_audio"),
    path("api/testing/clear-audio/", ClearTestingAudioView.as_view(), name="clear_testing_audio"),

    path("api-auth/", include("rest_framework.urls")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)