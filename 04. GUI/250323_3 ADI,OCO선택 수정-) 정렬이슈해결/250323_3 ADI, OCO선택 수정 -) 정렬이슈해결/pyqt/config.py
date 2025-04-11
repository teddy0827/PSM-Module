import matplotlib.font_manager as fm

class Config:
    WAFER_RADIUS = 150000
    SCALE_FACTOR = 1e-7
    SCALE_BAR_LENGTH = 30000
    SCALE_BAR_POSITION = 'lower center'
    SCALE_BAR_LENGTH_PIXELS = 30000
    SCALE_BAR_SIZE_VERTICAL = 500
    TEXT_POSITION_Y = -170000  # 텍스트 추가 위치

fontprops = fm.FontProperties(size=10)
