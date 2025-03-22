# design_matrix.py
import numpy as np

# ----- KMRC 디자인 행렬 -----
def get_kmrc_design_matrices(x, y, rx, ry):
    """
    ASML 스캐너의 KMRC 보정을 위한 디자인 행렬.
    x, y는 3차항까지 모두 사용하고, rx, ry는 rx^3항은 제외.(RK20 보정불가성분분)
    """
    X_dx = np.vstack([
        np.ones(len(x)),
        x / 1e6, -y / 1e6,
        (x ** 2) / 1e12, (x * y) / 1e12, (y ** 2) / 1e12,
        (x ** 3) / 1e15, (x ** 2 * y) / 1e15, (x * y ** 2) / 1e15, (y ** 3) / 1e15,
        rx / 1e6, -ry / 1e6,
        (rx ** 2) / 1e9, (rx * ry) / 1e9, (ry ** 2) / 1e9,
        (rx ** 3) / 1e12, (rx ** 2 * ry) / 1e12, (rx * ry ** 2) / 1e12, (ry ** 3) / 1e12
    ]).T

    X_dy = np.vstack([
        np.ones(len(y)),
        y / 1e6, x / 1e6,
        (y ** 2) / 1e12, (y * x) / 1e12, (x ** 2) / 1e12,
        (y ** 3) / 1e15, (y ** 2 * x) / 1e15, (y * x ** 2) / 1e15, (x ** 3) / 1e15,
        ry / 1e6, rx / 1e6,
        (ry ** 2) / 1e9, (ry * rx) / 1e9, (rx ** 2) / 1e9,
        (ry ** 3) / 1e12, (ry ** 2 * rx) / 1e12, (ry * rx ** 2) / 1e12
    ]).T

    return X_dx, X_dy


def get_kmrc_design_matrices_18para(x, y, rx, ry):
    """
    KMRC 18파라메터 디자인 행렬:
    - RK13, RK20 2개 제거
    """
    X_dx = np.vstack([
        np.ones(len(x)),
        x / 1e6, -y / 1e6,
        (x ** 2) / 1e12, (x * y) / 1e12, (y ** 2) / 1e12,
        (x ** 3) / 1e15, (x ** 2 * y) / 1e15, (x * y ** 2) / 1e15, (y ** 3) / 1e15,
        rx / 1e6, -ry / 1e6,
        (rx ** 2) / 1e9, (rx * ry) / 1e9, (ry ** 2) / 1e9,
                         (rx ** 2 * ry) / 1e12, (rx * ry ** 2) / 1e12, (ry ** 3) / 1e12
    ]).T

    X_dy = np.vstack([
        np.ones(len(y)),
        y / 1e6, x / 1e6,
        (y ** 2) / 1e12, (y * x) / 1e12, (x ** 2) / 1e12,
        (y ** 3) / 1e15, (y ** 2 * x) / 1e15, (y * x ** 2) / 1e15, (x ** 3) / 1e15,
        ry / 1e6, rx / 1e6,
        (ry ** 2) / 1e9, (ry * rx) / 1e9, (rx ** 2) / 1e9,
        (ry ** 3) / 1e12, (ry ** 2 * rx) / 1e12, (ry * rx ** 2) / 1e12
    ]).T

    return X_dx, X_dy

# ----- PSM 디자인 행렬 -----


def get_psm_design_matrices(rx, ry):
    """
    PSM 보정을 위한 디자인 행렬.
    rx, ry 만 사용하며, 보다 고차항을 포함.
    """
    X_dx = np.vstack([
        np.ones(len(rx)),
        rx / 1e6, -ry / 1e6,
        (rx ** 2) / 1e9, (rx * ry) / 1e9, (ry ** 2) / 1e9,
        (rx ** 3) / 1e12, (rx ** 2 * ry) / 1e12, (rx * ry ** 2) / 1e12, (ry ** 3) / 1e12,
        (rx ** 4) / 1e19, (rx ** 3 * ry) / 1e19, (rx ** 2 * ry ** 2) / 1e19, (rx * ry ** 3) / 1e19, (ry ** 4) / 1e19,
        (rx ** 5) / 1e23, (rx ** 4 * ry) / 1e23, (rx ** 3 * ry ** 2) / 1e23, (rx ** 2 * ry ** 3) / 1e23, (rx * ry ** 4) / 1e23, (ry ** 5) / 1e23,
        (rx ** 6) / 1e27, (rx ** 5 * ry) / 1e27, (rx ** 4 * ry ** 2) / 1e27, (rx ** 3 * ry ** 3) / 1e27, (rx ** 2 * ry ** 4) / 1e27, (rx * ry ** 5) / 1e27, (ry ** 6) / 1e27,
        (rx ** 7) / 1e31, (rx ** 6 * ry) / 1e31, (rx ** 5 * ry ** 2) / 1e31, (rx ** 4 * ry ** 3) / 1e31, (rx ** 3 * ry ** 4) / 1e31, (rx ** 2 * ry ** 5) / 1e31, (rx * ry ** 6) / 1e31, (ry ** 7) / 1e31
    ]).T

    X_dy = np.vstack([
        np.ones(len(ry)),
        ry / 1e6, rx / 1e6,
        (ry ** 2) / 1e9, (ry * rx) / 1e9, (rx ** 2) / 1e9,
        (ry ** 3) / 1e12, (ry ** 2 * rx) / 1e12, (ry * rx ** 2) / 1e12, (rx ** 3) / 1e12,
        (ry ** 4) / 1e19, (ry ** 3 * rx) / 1e19, (ry ** 2 * rx ** 2) / 1e19, (ry * rx ** 3) / 1e19, (rx ** 4) / 1e19,
        (ry ** 5) / 1e23, (ry ** 4 * rx) / 1e23, (ry ** 3 * rx ** 2) / 1e23, (ry ** 2 * rx ** 3) / 1e23, (ry * rx ** 4) / 1e23, (rx ** 5) / 1e23,
        (ry ** 6) / 1e27, (ry ** 5 * rx) / 1e27, (ry ** 4 * rx ** 2) / 1e27, (ry ** 3 * rx ** 3) / 1e27, (ry ** 2 * rx ** 4) / 1e27, (ry * rx ** 5) / 1e27, (rx ** 6) / 1e27,
        (ry ** 7) / 1e31, (ry ** 6 * rx) / 1e31, (ry ** 5 * rx ** 2) / 1e31, (ry ** 4 * rx ** 3) / 1e31, (ry ** 3 * rx ** 4) / 1e31, (ry ** 2 * rx ** 5) / 1e31, (ry * rx ** 6) / 1e31, (rx ** 7) / 1e31
    ]).T

    return X_dx, X_dy


def get_psm_design_matrices_38para(rx, ry):
    """
    PSM 38파라메터 디자인 행렬:
    원래의 디자인 행렬에 2개의 고차항을 추가한 형태.
    """
    X_dx = np.vstack([
        np.ones(len(rx)),
        rx / 1e6,
        -ry / 1e6,
        (rx ** 2) / 1e9,
        (rx * ry) / 1e9,
        (ry ** 2) / 1e9,
        (rx ** 3) / 1e12,
        (rx ** 2 * ry) / 1e12,
        (rx * ry ** 2) / 1e12,
        (ry ** 3) / 1e12,
        (rx ** 4) / 1e19,
        (rx ** 3 * ry) / 1e19,
        (rx ** 2 * ry ** 2) / 1e19,
        (rx * ry ** 3) / 1e19,
        (ry ** 4) / 1e19,
        (rx ** 5) / 1e23,
        (rx ** 4 * ry) / 1e23,
        (rx ** 3 * ry ** 2) / 1e23,
        (rx ** 2 * ry ** 3) / 1e23,
        (rx * ry ** 4) / 1e23,
        (ry ** 5) / 1e23,
        (rx ** 6) / 1e27,
        (rx ** 5 * ry) / 1e27,
        (rx ** 4 * ry ** 2) / 1e27,
        (rx ** 3 * ry ** 3) / 1e27,
        (rx ** 2 * ry ** 4) / 1e27,
        (rx * ry ** 5) / 1e27,
        (ry ** 6) / 1e27,
        (rx ** 7) / 1e31,
        (rx ** 6 * ry) / 1e31,
        (rx ** 5 * ry ** 2) / 1e31,
        (rx ** 4 * ry ** 3) / 1e31,
        (rx ** 3 * ry ** 4) / 1e31,
        (rx ** 2 * ry ** 5) / 1e31,
        (rx * ry ** 6) / 1e31,
        (ry ** 7) / 1e31,
        (rx ** 8) / 1e35,    # 추가항 1
        (rx ** 7 * ry) / 1e35  # 추가항 2
    ]).T

    X_dy = np.vstack([
        np.ones(len(ry)),
        ry / 1e6,
        rx / 1e6,
        (ry ** 2) / 1e9,
        (ry * rx) / 1e9,
        (rx ** 2) / 1e9,
        (ry ** 3) / 1e12,
        (ry ** 2 * rx) / 1e12,
        (ry * rx ** 2) / 1e12,
        (rx ** 3) / 1e12,
        (ry ** 4) / 1e19,
        (ry ** 3 * rx) / 1e19,
        (ry ** 2 * rx ** 2) / 1e19,
        (ry * rx ** 3) / 1e19,
        (rx ** 4) / 1e19,
        (ry ** 5) / 1e23,
        (ry ** 4 * rx) / 1e23,
        (ry ** 3 * rx ** 2) / 1e23,
        (ry ** 2 * rx ** 3) / 1e23,
        (ry * rx ** 4) / 1e23,
        (rx ** 5) / 1e23,
        (ry ** 6) / 1e27,
        (ry ** 5 * rx) / 1e27,
        (ry ** 4 * rx ** 2) / 1e27,
        (ry ** 3 * rx ** 3) / 1e27,
        (ry ** 2 * rx ** 4) / 1e27,
        (ry * rx ** 5) / 1e27,
        (rx ** 6) / 1e27,
        (ry ** 7) / 1e31,
        (ry ** 6 * rx) / 1e31,
        (ry ** 5 * rx ** 2) / 1e31,
        (ry ** 4 * rx ** 3) / 1e31,
        (ry ** 3 * rx ** 4) / 1e31,
        (ry ** 2 * rx ** 5) / 1e31,
        (ry * rx ** 6) / 1e31,
        (rx ** 7) / 1e31,
        (ry ** 8) / 1e35,    # 추가항 1
        (ry ** 7 * rx) / 1e35  # 추가항 2
    ]).T

    return X_dx, X_dy

def get_psm_design_matrices_33para(rx, ry):
    """
    PSM 33파라메터 디자인 행렬:
    38파라메터 디자인 행렬에서 앞의 33개 컬럼만 사용.
    """
    X_dx_full, X_dy_full = get_psm_design_matrices_38para(rx, ry)
    X_dx = X_dx_full[:, :33]
    X_dy = X_dy_full[:, :33]
    return X_dx, X_dy

# ----- CPE 디자인 행렬 -----


def get_cpe_design_matrices(rx, ry):
    """
    CPE 보정을 위한 디자인 행렬.
    rx, ry를 입력받아 3차항까지의 행렬을 생성.
    """
    X_dx = np.vstack([
        np.ones(len(rx)),
        rx / 1e6, -ry / 1e6,
        (rx ** 2) / 1e9, (rx * ry) / 1e9, (ry ** 2) / 1e9,
        (rx ** 3) / 1e12, (rx ** 2 * ry) / 1e12, (rx * ry ** 2) / 1e12, (ry ** 3) / 1e12
    ]).T

    X_dy = np.vstack([
        np.ones(len(ry)),
        ry / 1e6, rx / 1e6,
        (ry ** 2) / 1e9, (ry * rx) / 1e9, (rx ** 2) / 1e9,
        (ry ** 3) / 1e12, (ry ** 2 * rx) / 1e12, (ry * rx ** 2) / 1e12
    ]).T

    return X_dx, X_dy


def get_cpe_design_matrices_6para(rx, ry):
    """
    CPE 6파라메터 디자인 행렬:
    간단한 6개 항 (상수, 1차, 2차 항)
    """
    X_dx = np.vstack([
        np.ones(len(rx)),
        rx / 1e6,
        -ry / 1e6,
        (rx ** 2) / 1e9,
        (rx * ry) / 1e9,
        (ry ** 2) / 1e9
    ]).T
    X_dy = np.vstack([
        np.ones(len(ry)),
        ry / 1e6,
        rx / 1e6,
        (ry ** 2) / 1e9,
        (ry * rx) / 1e9,
        (rx ** 2) / 1e9
    ]).T
    return X_dx, X_dy

def get_cpe_design_matrices_15para(rx, ry):
    """
    CPE 15파라메터 디자인 행렬:
    보다 복잡한 15개 항 (상수, 1차~4차 항 등)
    """
    X_dx = np.vstack([
        np.ones(len(rx)),
        rx / 1e6,
        -ry / 1e6,
        (rx ** 2) / 1e9,
        (rx * ry) / 1e9,
        (ry ** 2) / 1e9,
        (rx ** 3) / 1e12,
        (rx ** 2 * ry) / 1e12,
        (rx * ry ** 2) / 1e12,
        (ry ** 3) / 1e12,
        (rx ** 4) / 1e15,
        (rx ** 3 * ry) / 1e15,
        (rx ** 2 * ry ** 2) / 1e15,
        (rx * ry ** 3) / 1e15,
        (ry ** 4) / 1e15
    ]).T
    X_dy = np.vstack([
        np.ones(len(ry)),
        ry / 1e6,
        rx / 1e6,
        (ry ** 2) / 1e9,
        (ry * rx) / 1e9,
        (rx ** 2) / 1e9,
        (ry ** 3) / 1e12,
        (ry ** 2 * rx) / 1e12,
        (ry * rx ** 2) / 1e12,
        (rx ** 3) / 1e12,
        (ry ** 4) / 1e15,
        (ry ** 3 * rx) / 1e15,
        (ry ** 2 * rx ** 2) / 1e15,
        (ry * rx ** 3) / 1e15,
        (rx ** 4) / 1e15
    ]).T
    return X_dx, X_dy

# ----- 옵션 딕셔너리 구성 -----
KMRC_OPTIONS = {
    '19para': get_kmrc_design_matrices,
    '18para': get_kmrc_design_matrices_18para
}

PSM_OPTIONS = {
    '38para': get_psm_design_matrices,
    '33para': get_psm_design_matrices_33para,
    
}

CPE_OPTIONS = {
    '19para': get_cpe_design_matrices,
    '6para': get_cpe_design_matrices_6para,
    '15para': get_cpe_design_matrices_15para
}

# 최종 선택 딕셔너리
DESIGN_MATRIX_FUNCTIONS = {
    'kmrc': KMRC_OPTIONS,
    'psm': PSM_OPTIONS,
    'cpe': CPE_OPTIONS
}
