# design_matrix.py
import numpy as np

# ----- OSR(WK,RK) 디자인 행렬 -----
def osr_20para(x, y, rx, ry):
    """
    WK20para, RK20para (1023_1023)    
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
        (ry ** 3) / 1e12, (ry ** 2 * rx) / 1e12, (ry * rx ** 2) / 1e12, (rx ** 3) / 1e12
    ]).T

    return X_dx, X_dy


def osr_19para(x, y, rx, ry):
    """
    WK20para, RK19para (1023_511)
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


def osr_18para(x, y, rx, ry):
    """
    WK20para, RK18para (959_511)
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



def osr_15para(x, y, rx, ry):
    """
    WK20para, RK15para (623_255) (RK9,RK15,RK17,RK18,RK20 제외)
    """
    X_dx = np.vstack([
        np.ones(len(x)),
        x / 1e6, -y / 1e6,
        (x ** 2) / 1e12, (x * y) / 1e12, (y ** 2) / 1e12,
        (x ** 3) / 1e15, (x ** 2 * y) / 1e15, (x * y ** 2) / 1e15, (y ** 3) / 1e15,
        rx / 1e6, -ry / 1e6,
        (rx ** 2) / 1e9,                 (ry ** 2) / 1e9,
        (rx ** 3) / 1e12,                                 (ry ** 3) / 1e12
    ]).T

    X_dy = np.vstack([
        np.ones(len(y)),
        y / 1e6, x / 1e6,
        (y ** 2) / 1e12, (y * x) / 1e12, (x ** 2) / 1e12,
        (y ** 3) / 1e15, (y ** 2 * x) / 1e15, (y * x ** 2) / 1e15, (x ** 3) / 1e15,
        ry / 1e6, rx / 1e6,
        (ry ** 2) / 1e9, (ry * rx) / 1e9, (rx ** 2) / 1e9,
        (ry ** 3) / 1e12, (ry ** 2 * rx) / 1e12 
    ]).T

    return X_dx, X_dy




# ----- CPE 디자인 행렬 -----


def cpe_19para(rx, ry):
    """
    RK 19para (1023_511)
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


def cpe_15para(rx, ry):
    """
    RK 15para (623_255)
    """

    X_dx = np.vstack([
        np.ones(len(rx)),
        rx / 1e6, -ry / 1e6,
        (rx ** 2) / 1e9,                   (ry ** 2) / 1e9,
        (rx ** 3) / 1e12,                                    (ry ** 3) / 1e12
    ]).T

    X_dy = np.vstack([
        np.ones(len(ry)),
        ry / 1e6, rx / 1e6,
        (ry ** 2) / 1e9, (ry * rx) / 1e9, (rx ** 2) / 1e9,
        (ry ** 3) / 1e12, (ry ** 2 * rx) / 1e12
    ]).T

    return X_dx, X_dy


def cpe_6para(rx, ry):
    """
    RK 6para (7_7)
    """
    X_dx = np.vstack([
        np.ones(len(rx)),
        rx / 1e6, -ry / 1e6        
    ]).T

    X_dy = np.vstack([
        np.ones(len(ry)),
        ry / 1e6, rx / 1e6  
    ]).T

    return X_dx, X_dy




# ----- PSM Fitting용용 디자인 행렬 -----


def cpe_k_to_fit(rx, ry):
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




# ----- 옵션 딕셔너리 구성 -----
KMRC_OPTIONS = {
    '20para': osr_20para,
    '19para': osr_19para,
    '18para': osr_18para,
    '15para': osr_15para,
}

CPE_OPTIONS = {
    '19para': cpe_19para,
    '15para': cpe_15para,
    '6para': cpe_6para,
}

PSM_OPTIONS = {
    '38para': cpe_k_to_fit,    
}


# 최종 선택 딕셔너리
DESIGN_MATRIX_FUNCTIONS = {
    'kmrc': KMRC_OPTIONS,
    'psm': PSM_OPTIONS,
    'cpe': CPE_OPTIONS

    
}
