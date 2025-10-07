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

def mascara(frame):
    mask = np.zeros(frame.shape[:2], dtype='uint8')
    (cX, cY) = (frame.shape[1] // 2, frame.shape[0] // 2)
    widthl = 400
    widthr = 600
    heightp = -79
    heightb = 300
    cv2.rectangle(mask,
                 (cX - widthl//2, (cY) - heightp//2),
                 (cX + widthr//2, cY + heightb//2),
                 255, -1)
    masked_image = cv2.bitwise_and(frame, frame, mask=mask)
    return masked_image
