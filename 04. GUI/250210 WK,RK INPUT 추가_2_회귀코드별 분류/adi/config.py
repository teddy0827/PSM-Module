# config.py
# 설정 파일: 각종 경로 및 상수를 정의합니다.

# nau 파일이 저장된 기본 폴더 경로
FOLDER_PATH = 'C:/py_data/nau/1lot'

# radius filtering에 사용할 반경 임계값
RADIUS_THRESHOLD = 150000  

# Zernike 분석의 최대 인덱스
MAX_INDEX = 64          

# Outlier 판정을 위한 상수들
OUTLIER_THRESHOLD = 3.0      # studentized residual 임계값
DMARGIN_X = 0.005            # X 방향 DMARGIN
DMARGIN_Y = 0.0025           # Y 방향 DMARGIN
OUTLIER_SPEC_RATIO = 1.5     # DMARGIN 배율

