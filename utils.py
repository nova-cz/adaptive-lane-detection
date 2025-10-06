import cv2
import numpy as np

def gamma_correction(b, g, r, gamma):
    """Aplica corrección gamma a cada canal BGR"""
    b_corrected = np.array(255 * (b / 255) ** gamma, dtype='uint8')
    g_corrected = np.array(255 * (g / 255) ** gamma, dtype='uint8')
    r_corrected = np.array(255 * (r / 255) ** gamma, dtype='uint8')
    gamma_corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
    return gamma_corrected

def detect_shadow_mask(frame, threshold=90):
    """
    Detecta regiones de sombra usando el canal V de HSV.
    
    Parámetros:
        frame: imagen BGR
        threshold: umbral para considerar una región como sombra (0-255)
                   Valores típicos: 70-100
    
    Retorna:
        Máscara binaria donde 255 = sombra, 0 = luz
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    _, _, v = cv2.split(hsv)
    
    # Crear máscara de sombras (píxeles oscuros)
    shadow_mask = (v < threshold).astype(np.uint8) * 255
    
    # Operaciones morfológicas para limpiar la máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel)
    
    # Suavizar bordes para transición gradual
    shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)
    
    return shadow_mask

def apply_selective_clahe(frame, shadow_threshold=90, clipLimit=3.0, tileGridSize=(8, 8)):
    """
    Aplica CLAHE SOLO en regiones de sombra, preservando zonas iluminadas.
    
    Esta es la función clave para resolver tu problema.
    
    Parámetros:
        frame: imagen BGR
        shadow_threshold: umbral para detectar sombras (70-100)
        clipLimit: intensidad del realce en sombras (2.0-4.0)
        tileGridSize: tamaño de bloques CLAHE
    
    Retorna:
        Imagen con CLAHE aplicado selectivamente
    """
    # 1. Detectar dónde están las sombras
    shadow_mask = detect_shadow_mask(frame, shadow_threshold)
    
    # 2. Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # 3. Aplicar CLAHE al canal V completo
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    v_clahe = clahe.apply(v)
    
    # 4. MEZCLA SELECTIVA: usar V original en zonas claras, V+CLAHE en sombras
    # Normalizar máscara a rango 0.0-1.0 para blending suave
    shadow_weight = shadow_mask.astype(np.float32) / 255.0
    
    # Interpolación: donde shadow_weight=1 (sombra) → usar v_clahe
    #                donde shadow_weight=0 (luz) → usar v original
    v_blended = (shadow_weight * v_clahe + (1 - shadow_weight) * v).astype(np.uint8)
    
    # 5. Reconstruir imagen
    hsv_result = cv2.merge((h, s, v_blended))
    result = cv2.cvtColor(hsv_result, cv2.COLOR_HSV2BGR)
    
    return result, shadow_mask

def apply_clahe_hsv(frame, clipLimit=2.0, tileGridSize=(8, 8)):
    """
    CLAHE tradicional (aplicado a toda la imagen).
    Mantenido para comparación.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    v_clahe = clahe.apply(v)
    
    hsv_clahe = cv2.merge((h, s, v_clahe))
    result = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)
    
    return result

def enhance_yellow_lines(frame):
    """
    Realce específico para líneas amarillas usando espacio LAB.
    Complementa el procesamiento de sombras.
    """
    # Convertir a LAB (mejor para separar luminosidad y color)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE solo al canal L (luminosidad)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Realzar canal B (contiene información de amarillo)
    b_enhanced = cv2.add(b, 20)
    
    # Reconstruir
    lab_enhanced = cv2.merge((l_clahe, a, b_enhanced))
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return result

def equalize_histogram_hsv(frame, k):
    """Ecualización global del histograma en canal V"""
    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_HSV)
    
    hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    cumulative_hist = np.cumsum(hist)
    
    (M, N) = v.shape
    dx = (k - 1) / (M * N)
    y2 = np.array([np.round(cumulative_hist[i] * dx) for i in range(256)], dtype='uint8')
    
    v_equalized = y2[v]
    image_HSV = cv2.merge([h, s, v_equalized])
    result = cv2.cvtColor(image_HSV, cv2.COLOR_HSV2BGR)
    
    return result

def ajustar_contraste_hsv(frame, alow, ahigh, amin, amax, mask=None):
    """
    Auto-contraste por percentiles en canal V, con opción de aplicar SOLO
    sobre una máscara (por ejemplo, regiones de sombra).

    Parámetros:
        frame: imagen BGR
        alow: valor que representa el percentil bajo. Puede ser:
              - un "trim" en [0,1] (por ejemplo 0.05 para recortar 5% bajos)
              - o un "keep" en (0.5,1.0] donde 0.95 significa conservar 95% (recortar 5%)
        ahigh: similar a alow para el percentil alto
        amin, amax: rango de salida
        mask: máscara binaria opcional (uint8, 0/255). Si se provee, el mapeo
              se calcula usando sólo los píxeles de la máscara y se aplica
              únicamente a esos píxeles, preservando las zonas no enmascaradas.

    Retorna:
        Imagen BGR con ajuste aplicado (fuera de la máscara permanece igual si mask provista)
    """
    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_HSV)

    M, N = v.shape
    total_pixels = M * N

    # Interpretación robusta de alow/ahigh: si el usuario pasó valores cerca de 1.0
    # (por ejemplo 0.95) asumimos que representa "keep fraction" y lo convertimos a trim.
    def to_trim(x):
        if x > 0.5:
            return 1.0 - x
        return x

    low_trim = to_trim(alow)
    high_trim = to_trim(ahigh)

    # Calculamos histograma sobre la región solicitada (mask) o sobre toda la imagen
    if mask is not None:
        mask_bool = (mask > 0).astype('uint8')
        hist = cv2.calcHist([v], [0], mask_bool, [256], [0, 256])
        pixels_considered = int(np.sum(mask_bool))
        if pixels_considered == 0:
            # Si la máscara está vacía, no aplicar cambio
            return frame
    else:
        hist = cv2.calcHist([v], [0], None, [256], [0, 256])
        pixels_considered = total_pixels

    cumulative_hist = np.cumsum(hist)

    low_count = int(pixels_considered * low_trim)
    high_count = int(pixels_considered * (1.0 - high_trim))

    # Encontrar índices de intensidad correspondientes a los percentiles
    low_idx = next((i for i in range(256) if cumulative_hist[i] >= low_count), 0)
    high_idx = next((i for i in range(255, -1, -1) if cumulative_hist[i] <= high_count), 255)

    # Asegurar orden correcto
    if high_idx <= low_idx:
        # rango inválido -> no cambiar
        return frame

    dx = (amax - amin) / float(high_idx - low_idx)

    table_map = np.array([
        amin if i <= low_idx else amax if i >= high_idx else int(amin + (i - low_idx) * dx)
        for i in range(256)
    ], dtype='uint8')

    # Aplicar mapeo solo dentro de la máscara si se proporcionó
    if mask is not None:
        v_out = v.copy()
        mask_bool = (mask > 0)
        v_out[mask_bool] = table_map[v[mask_bool]]
    else:
        v_out = table_map[v]

    image_HSV = cv2.merge([h, s, v_out])
    result = cv2.cvtColor(image_HSV, cv2.COLOR_HSV2BGR)

    return result

def mascara(frame):
    """Aplica máscara ROI para enfocarse en la zona de carretera"""
    mask = np.zeros(frame.shape[:2], dtype='uint8')
    (cX, cY) = (frame.shape[1] // 2, frame.shape[0] // 2)
    
    widthl = 400
    widthr = 600
    heightp = -79
    heightb = 300
    
    cv2.rectangle(mask,
                 (cX - widthl//2, (cY) - heightp//2),
                 (cX + widthr//2, cY + heightb//2),
                 255,
                 -1)
    
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_image