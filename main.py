import cv2
import numpy as np
import sys
from utils import gamma_correction, equalize_histogram_hsv, ajustar_contraste_hsv, mascara

def process_frame(frame, k=256, gamma=3, alow=0.95, ahigh=0.95, amin=0, amax=255):
    frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
    b, g, r = cv2.split(frame)
    gamma_frame = gamma_correction(b, g, r, gamma)
    equalized_image = equalize_histogram_hsv(gamma_frame, k)
    image_contrast = ajustar_contraste_hsv(equalized_image, alow, ahigh, amin, amax)
    gray_frame = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)
    masked_frame = mascara(gray_frame)
    return masked_frame

def main():
    if len(sys.argv) < 2:
        print("Error: Se debe proporcionar la ruta del video de entrada como argumento.")
        print("Uso: python main.py <ruta_del_video_de_entrada> [k] [gamma] [alow] [ahigh] [amin] [amax]")
        print("\nParámetros opcionales (con valores por defecto):")
        print("  k=256      - Valor para ecualización de histograma")
        print("  gamma=3    - Valor de corrección gamma")
        print("  alow=0.95  - Porcentaje bajo para auto-contraste")
        print("  ahigh=0.95 - Porcentaje alto para auto-contraste")
        print("  amin=0     - Intensidad mínima para mapeo")
        print("  amax=255   - Intensidad máxima para mapeo")
        print("\nEjemplo: python main.py lineas.mp4 256 3 0.95 0.95 0 255")
        sys.exit(1)

    input_video_path = sys.argv[1]
    
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 256
    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 3
    alow = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
    ahigh = float(sys.argv[5]) if len(sys.argv) > 5 else 0.95
    amin = float(sys.argv[6]) if len(sys.argv) > 6 else 0
    amax = float(sys.argv[7]) if len(sys.argv) > 7 else 255
    
    print(f"Parámetros utilizados:")
    print(f"  k={k}, gamma={gamma}, alow={alow}, ahigh={ahigh}, amin={amin}, amax={amax}")
    
    if input_video_path.endswith('.mp4'):
        output_video_path = input_video_path.replace('.mp4', '_processed.mp4')
    else:
        output_video_path = 'output_processed.mp4'

    cap = cv2.VideoCapture(input_video_path)
    
    if not cap.isOpened():
        print(f"Error: No se pudo abrir el video en {input_video_path}")
        print("Verificar:")
        print("  - Ruta del archivo es correcta")
        print("  - Archivo existe y es accesible")
        print("  - Formato de video es compatible")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = int(width * 0.7)
    output_height = int(height * 0.7)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height), isColor=False)

    print(f"Procesando video: {input_video_path}")
    print("Controles:")
    print("  ESPACIO - Pausar/Reanudar")
    print("  'q' - Salir")
    print("  'r' - Reiniciar desde el inicio")
    print("  '+' - Aumentar velocidad")
    print("  '-' - Disminuir velocidad")
    
    paused = False
    speed_multiplier = 1.0
    frame_delay = max(1, int(1000 / fps))
    
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            processed_frame = process_frame(frame, k, gamma, alow, ahigh, amin, amax)
            frame_resized = cv2.resize(frame, (output_width, output_height))
            cv2.imshow('Video Original', frame_resized)
            cv2.imshow('Video Procesado (Detección de Líneas)', processed_frame)
            out.write(processed_frame)
        
        current_delay = max(1, int(frame_delay / speed_multiplier))
        key = cv2.waitKey(current_delay) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            if paused:
                print("⏸️  Video pausado - Presiona ESPACIO para continuar")
            else:
                print("▶️  Video reanudado")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            print("🔄 Video reiniciado")
        elif key == ord('+') or key == ord('='):
            speed_multiplier = min(speed_multiplier * 1.5, 5.0)
            print(f"⚡ Velocidad: {speed_multiplier:.1f}x")
        elif key == ord('-'):
            speed_multiplier = max(speed_multiplier / 1.5, 0.2)
            print(f"🐌 Velocidad: {speed_multiplier:.1f}x")

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Procesamiento finalizado. Video guardado en {output_video_path}")

if __name__ == "__main__":
    main()