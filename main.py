import cv2
import numpy as np
import sys
from utils import gamma_correction, equalize_histogram_hsv, ajustar_contraste_hsv, mascara

def process_frame(frame, k=256, gamma=3, alow=0.95, ahigh=0.95, amin=0, amax=255):
    """
    Pipeline de procesamiento mejorado.
    Mantiene tu lógica original pero con mejor detección en sombras.
    """
    frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
    
    # Tu pipeline original
    b, g, r = cv2.split(frame)
    gamma_frame = gamma_correction(b, g, r, gamma)
    equalized_image = equalize_histogram_hsv(gamma_frame, k)
    image_contrast = ajustar_contraste_hsv(equalized_image, alow, ahigh, amin, amax)
    
    # Convertir a escala de grises
    gray_frame = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)
    
    # Aplicar máscara mejorada (ahora detecta líneas en sombras)
    # Pasamos image_contrast (BGR) en lugar de gray_frame para que
    # mascara() pueda detectar sombras y aplicar realces
    masked_frame = mascara(image_contrast)
    
    return masked_frame

def main():
    if len(sys.argv) < 2:
        print("Error: Se debe proporcionar la ruta del video de entrada como argumento.")
        print("Uso: python main.py <ruta_del_video_de_entrada> [k] [gamma] [alow] [ahigh] [amin] [amax]")
        sys.exit(1)
    
    input_video_path = sys.argv[1]
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 256
    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 3
    alow = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
    ahigh = float(sys.argv[5]) if len(sys.argv) > 5 else 0.95
    amin = float(sys.argv[6]) if len(sys.argv) > 6 else 0
    amax = float(sys.argv[7]) if len(sys.argv) > 7 else 255

    if input_video_path.endswith('.mp4'):
        output_video_path = input_video_path.replace('.mp4', '_processed.mp4')
    else:
        output_video_path = 'output_processed.mp4'

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {input_video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = int(width * 0.7)
    output_height = int(height * 0.7)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width * 2, output_height), isColor=True)

    print(f"Procesando video: {input_video_path}")
    print(f"Parámetros: k={k}, gamma={gamma}, alow={alow}, ahigh={ahigh}")
    print("Controles: ESPACIO=pausar, q=salir, r=reiniciar, +=más rápido, -=más lento")

    paused = False
    speed_multiplier = 1.0
    frame_delay = max(1, int(1000 / fps))
    frame_count = 0

    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Procesar frame con pipeline mejorado
            processed_frame = process_frame(frame, k, gamma, alow, ahigh, amin, amax)
            
            # Visualización
            frame_resized = cv2.resize(frame, (output_width, output_height))
            if processed_frame is None:
                processed_bgr = np.zeros_like(frame_resized)
            else:
                if len(processed_frame.shape) == 2:
                    processed_bgr = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)
                else:
                    processed_bgr = processed_frame
                processed_bgr = cv2.resize(processed_bgr, (output_width, output_height))

            composite = np.hstack([frame_resized, processed_bgr])
            cv2.imshow('Resultado (Original | Procesado)', composite)

            # Guardar frame compuesto
            out.write(composite)

        current_delay = max(1, int(frame_delay / speed_multiplier))
        key = cv2.waitKey(current_delay) & 0xFF

        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"\n{'Pausado' if paused else 'Reanudado'} en frame {frame_count}")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            print("\nVideo reiniciado")
        elif key == ord('+') or key == ord('='):
            speed_multiplier = min(speed_multiplier * 1.5, 5.0)
            print(f"\nVelocidad: {speed_multiplier:.1f}x")
        elif key == ord('-'):
            speed_multiplier = max(speed_multiplier / 1.5, 0.2)
            print(f"\nVelocidad: {speed_multiplier:.1f}x")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nProcesamiento finalizado. Video guardado en {output_video_path}")
    print(f"Total de frames procesados: {frame_count}")

if __name__ == "__main__":
    main()