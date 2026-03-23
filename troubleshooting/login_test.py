from huggingface_hub import HfApi, login

# 현재 설정된 토큰을 사용해 로그인 시도 (이미 HF_TOKEN가 있으면 자동 사용)
try:
    api = HfApi()
    user_info = api.whoami()          # 현재 인증된 사용자 정보 반환
    print("로그인 성공:", user_info)
except HfHubError as e:
    print("로그인 실패:", e)