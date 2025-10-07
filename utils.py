import cv2
import numpy as np

def gamma_correction(b, g, r, gamma):
    b_corrected = np.array(255 * (b / 255) ** gamma, dtype='uint8')
    g_corrected = np.array(255 * (g / 255) ** gamma, dtype='uint8')
    r_corrected = np.array(255 * (r / 255) ** gamma, dtype='uint8')
    gamma_corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
    return gamma_corrected

def equalize_histogram_hsv(frame, k):
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

def ajustar_contraste_hsv(frame, alow, ahigh, amin, amax):
    image_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_HSV)
    hist = cv2.calcHist([v], [0], None, [256], [0, 256])
    cumulative_hist = np.cumsum(hist)
    (M, N) = v.shape
    total_pixels = M * N
    multlow = int(total_pixels * alow)
    multhigh = int(total_pixels * (1 - ahigh))
    alowp = min([i for i in range(256) if cumulative_hist[i] >= multlow])
    ahighp = max([i for i in range(256) if cumulative_hist[i] <= multhigh])
    if ahighp != alowp:
        dx = (amax - amin) / (ahighp - alowp)
    else:
        dx = 1
    table_map = np.array([
        amin if i <= alowp else
        amax if i >= ahighp else
        amin + ((i - alowp) * dx)
        for i in range(256)
    ], dtype='uint8')
    v_correct = table_map[v]
    image_HSV = cv2.merge([h, s, v_correct])
    result = cv2.cvtColor(image_HSV, cv2.COLOR_HSV2BGR)
    return result

def detect_shadow_regions(frame):
    """
    Detección adaptativa de sombras basada en LAB + HSV.
    Retorna una máscara donde 255 = sombra, 0 = no sombra.
    """
    # Convertir a LAB y HSV
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    l = lab[:, :, 0].astype(np.float32)
    s = hsv[:, :, 1].astype(np.float32)
    v = hsv[:, :, 2].astype(np.float32)

    # Normalizar entre 0-1
    l_norm = l / 255.0
    s_norm = s / 255.0
    v_norm = v / 255.0

    # Sombras: baja luminosidad pero saturación normal o alta
    # (porque las sombras no desaturan el color completamente)
    shadow_score = (1.0 - l_norm) * (0.6 + 0.4 * s_norm)

    # Escalar a 0-255
    shadow_mask = (shadow_score * 255).astype(np.uint8)

    # Umbral adaptativo local
    shadow_mask = cv2.adaptiveThreshold(
        shadow_mask,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        35,   # tamaño del bloque
        -5    # valor negativo => detecta zonas oscuras
    )

    # Filtrar ruido y suavizar bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)
    shadow_mask = cv2.GaussianBlur(shadow_mask, (7, 7), 0)

    return shadow_mask



def enhance_lines_in_shadows(frame, shadow_mask):
    """
    Realza las líneas específicamente en zonas con sombras.
    Mantiene el resto de la imagen sin cambios.
    """
    # Convertir a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    # Crear versión realzada solo para sombras
    v_enhanced = v.copy().astype(np.float32)
    s_enhanced = s.copy().astype(np.float32)
    
    # Normalizar shadow_mask
    shadow_factor = shadow_mask.astype(np.float32) / 255.0
    
    # En zonas de sombra, incrementar brillo y saturación
    # Esto ayuda a recuperar el color amarillo oscurecido
    v_enhanced = v_enhanced + (shadow_factor * 820)  # Boost de brillo
    s_enhanced = s_enhanced + (shadow_factor * 420)  # Boost de saturación
    
    # Clip para mantener en rango válido
    v_enhanced = np.clip(v_enhanced, 0, 255).astype(np.uint8)
    s_enhanced = np.clip(s_enhanced, 0, 255).astype(np.uint8)
    
    # Reconstruir imagen
    hsv_enhanced = cv2.merge([h, s_enhanced, v_enhanced])
    result = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    
    return result

def adaptive_local_contrast(frame):
    """
    Aplica CLAHE local para mejorar contraste en sombras
    sin afectar demasiado las zonas claras.
    """
    # Convertir a LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Aplicar CLAHE solo al canal L (luminosidad)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    
    # Reconstruir
    lab_clahe = cv2.merge([l_clahe, a, b])
    result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    return result

def mascara(frame):
    """
    Aplica máscara ROI y mantiene solo las líneas detectadas visibles.
    MODIFICADO: Ahora detecta mejor las líneas en sombras.
    """
    # 1. Aplicar ROI geométrico (tu código original)
    mask_roi = np.zeros(frame.shape[:2], dtype='uint8')
    (cX, cY) = (frame.shape[1] // 2, frame.shape[0] // 2)
    widthl = 400
    widthr = 600
    heightp = -79
    heightb = 300
    cv2.rectangle(mask_roi,
                 (cX - widthl//2, (cY) - heightp//2),
                 (cX + widthr//2, cY + heightb//2),
                 255, -1)
    
    # 2. Si el frame es en escala de grises, convertir temporalmente a BGR
    if len(frame.shape) == 2:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    else:
        frame_bgr = frame
    
    # 3. Detectar sombras
    shadow_mask = detect_shadow_regions(frame_bgr)
    
    # 4. Realzar líneas en sombras
    frame_enhanced = enhance_lines_in_shadows(frame_bgr, shadow_mask)
    
    # 5. Aplicar contraste local adaptativo
    frame_contrast = adaptive_local_contrast(frame_enhanced)
    
    # 6. Convertir a escala de grises para umbralización
    gray = cv2.cvtColor(frame_contrast, cv2.COLOR_BGR2GRAY)
    
    # 7. Umbralización adaptativa para detectar líneas
    # Esto funciona mejor que umbral fijo, especialmente en iluminación variable
    thresh = cv2.adaptiveThreshold(
        gray, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 
        15,  # Tamaño del vecindario
        -5   # Constante de ajuste (negativa para detectar líneas claras)
    )
    
    # 8. Detectar líneas blancas adicionales con umbral alto
    _, thresh_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # 9. Combinar ambas detecciones
    combined_lines = cv2.bitwise_or(thresh, thresh_white)

    # 9.5. Detección adicional de color amarillo (en sombra y luz)
    mask_yellow = detectar_lineas_amarillas(frame_contrast)
    combined_lines = cv2.bitwise_or(combined_lines, mask_yellow)

    
    # 10. Operaciones morfológicas para limpiar ruido
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    combined_lines = cv2.morphologyEx(combined_lines, cv2.MORPH_CLOSE, kernel)
    combined_lines = cv2.morphologyEx(combined_lines, cv2.MORPH_OPEN, kernel)
    
    # 11. Aplicar ROI geométrico
    combined_lines = cv2.bitwise_and(combined_lines, combined_lines, mask=mask_roi)
    
    # 12. Aplicar la máscara de líneas al frame original en escala de grises
    masked_image = cv2.bitwise_and(frame, frame, mask=combined_lines)
    
    return masked_image


def detectar_lineas_amarillas(frame):
    """
    Detección robusta de líneas amarillas (desde tonos brillantes hasta casi negros).
    Incluye amarillos desaturados, grisáceos y bajo sombra profunda.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Amarillo brillante (zonas iluminadas)
    lower_yellow_bright = np.array([15, 80, 120])
    upper_yellow_bright = np.array([40, 255, 255])

    # Amarillo oscuro o grisáceo (zonas en sombra)
    # Aumentamos S y V máximos hasta 255 para abarcar cualquier matiz residual
    lower_yellow_dark = np.array([4, 4, 4])     # casi negro con leve tono amarillo
    upper_yellow_dark = np.array([45, 255, 255])   # rango amplio hasta el máximo posible

    # Combinar ambas máscaras
    mask_bright = cv2.inRange(hsv, lower_yellow_bright, upper_yellow_bright)
    mask_dark = cv2.inRange(hsv, lower_yellow_dark, upper_yellow_dark)
    combined_yellow = cv2.bitwise_or(mask_bright, mask_dark)

    # Suavizar y limpiar bordes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_yellow = cv2.morphologyEx(combined_yellow, cv2.MORPH_CLOSE, kernel)
    combined_yellow = cv2.morphologyEx(combined_yellow, cv2.MORPH_OPEN, kernel)

    # Realzar bordes finos (opcional, ayuda a captar líneas finas bajo sombra)
    edges = cv2.Canny(combined_yellow, 40, 150)
    combined_yellow = cv2.bitwise_or(combined_yellow, edges)

    return combined_yellow


