import cv2
import numpy as np
import sys
from utils import (gamma_correction, equalize_histogram_hsv, ajustar_contraste_hsv, 
                   mascara, apply_clahe_hsv, apply_selective_clahe, enhance_yellow_lines)

def process_frame(frame, k=256, gamma=3, alow=0.95, ahigh=0.95, amin=0, amax=255, 
                  clahe_clip=3.0, clahe_grid=(8,8), shadow_threshold=90, pipeline_mode='selective'):
    """
    Procesa un fotograma con diferentes estrategias de mejora.
    
    Parámetros:
        pipeline_mode: 
            'original' - Pipeline sin CLAHE
            'clahe' - CLAHE global (problema de saturación)
            'selective' - CLAHE SOLO en sombras (✅ RECOMENDADO)
            'yellow_enhanced' - Selective + realce de amarillos
            'adaptive' - Combinación óptima para condiciones variables
    """
    # 1. Redimensionar
    frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
    
    # 2. Corrección gamma
    b, g, r = cv2.split(frame)
    gamma_frame = gamma_correction(b, g, r, gamma)
    
    # 3. Aplicar procesamiento según el modo
    if pipeline_mode == 'original':
        # Pipeline original sin cambios
        equalized_image = equalize_histogram_hsv(gamma_frame, k)
        image_contrast = ajustar_contraste_hsv(equalized_image, alow, ahigh, amin, amax)
        
    elif pipeline_mode == 'clahe':
        # CLAHE global (puede saturar zonas claras)
        clahe_image = apply_clahe_hsv(gamma_frame, clipLimit=clahe_clip, tileGridSize=clahe_grid)
        image_contrast = ajustar_contraste_hsv(clahe_image, alow, ahigh, amin, amax)
        
    elif pipeline_mode == 'selective':
        # ✅ SOLUCIÓN ÓPTIMA: CLAHE solo en sombras
        selective_image, _ = apply_selective_clahe(
            gamma_frame, 
            shadow_threshold=shadow_threshold,
            clipLimit=clahe_clip,
            tileGridSize=clahe_grid
        )
        selective_image, shadow_mask = apply_selective_clahe(
            gamma_frame,
            shadow_threshold=shadow_threshold,
            clipLimit=clahe_clip,
            tileGridSize=clahe_grid
        )
        image_contrast = ajustar_contraste_hsv(selective_image, alow, ahigh, amin, amax, mask=shadow_mask)
        
    elif pipeline_mode == 'yellow_enhanced':
        # CLAHE selectivo + realce de líneas amarillas
        selective_image, shadow_mask = apply_selective_clahe(
            gamma_frame,
            shadow_threshold=shadow_threshold,
            clipLimit=clahe_clip,
            tileGridSize=clahe_grid
        )
        yellow_enhanced = enhance_yellow_lines(selective_image)
        image_contrast = ajustar_contraste_hsv(yellow_enhanced, alow, ahigh, amin, amax, mask=shadow_mask)
        
    elif pipeline_mode == 'adaptive':
        # Modo adaptativo: detecta condiciones y ajusta automáticamente
        hsv_test = cv2.cvtColor(gamma_frame, cv2.COLOR_BGR2HSV)
        v_test = cv2.split(hsv_test)[2]
        mean_brightness = np.mean(v_test)
        
        # Si el frame es muy oscuro (mucha sombra), usar CLAHE agresivo
        if mean_brightness < 80:
            selective_image, shadow_mask = apply_selective_clahe(
                gamma_frame,
                shadow_threshold=100,
                clipLimit=4.0,
                tileGridSize=clahe_grid
            )
        # Si tiene iluminación mixta, usar CLAHE selectivo estándar
        else:
            selective_image, shadow_mask = apply_selective_clahe(
                gamma_frame,
                shadow_threshold=shadow_threshold,
                clipLimit=clahe_clip,
                tileGridSize=clahe_grid
            )

        image_contrast = ajustar_contraste_hsv(selective_image, alow, ahigh, amin, amax, mask=shadow_mask)
    
    # 4. Conversión a escala de grises
    gray_frame = cv2.cvtColor(image_contrast, cv2.COLOR_BGR2GRAY)
    
    # 5. Aplicar máscara ROI
    masked_frame = mascara(gray_frame)
    
    return masked_frame

def process_frame_debug(frame, shadow_threshold=90, clahe_clip=3.0, clahe_grid=(8,8)):
    """
    Versión de debug que retorna la máscara de sombras para visualización.
    """
    frame = cv2.resize(frame, None, fx=0.7, fy=0.7)
    b, g, r = cv2.split(frame)
    gamma_frame = gamma_correction(b, g, r, gamma=3)
    
    # Obtener imagen procesada Y máscara de sombras
    selective_image, shadow_mask = apply_selective_clahe(
        gamma_frame,
        shadow_threshold=shadow_threshold,
        clipLimit=clahe_clip,
        tileGridSize=clahe_grid
    )
    
    return selective_image, shadow_mask

def main():
    if len(sys.argv) < 2:
        print("Error: Se debe proporcionar la ruta del video de entrada.")
        print("\n" + "="*70)
        print("USO DEL PROGRAMA")
        print("="*70)
        print("\nFormato:")
        print("  python main.py <video> [params...] [mode]")
        print("\n📋 MODOS DISPONIBLES:")
        print("  'original'        - Sin CLAHE (baseline)")
        print("  'clahe'           - CLAHE global (satura zonas claras)")
        print("  'selective'       - ✅ CLAHE solo en sombras (RECOMENDADO)")
        print("  'yellow_enhanced' - Selective + realce de amarillos")
        print("  'adaptive'        - Ajuste automático según iluminación")
        print("\n🎯 EJEMPLO RECOMENDADO:")
        print("  python main.py video.mp4 256 3 0.95 0.95 0 255 3.0 8 90 selective")
        print("\n⚙️  PARÁMETROS:")
        print("  [1] k              - Niveles de ecualización (256)")
        print("  [2] gamma          - Corrección gamma (3.0)")
        print("  [3] alow           - Percentil bajo (0.95)")
        print("  [4] ahigh          - Percentil alto (0.95)")
        print("  [5] amin           - Valor mínimo contraste (0)")
        print("  [6] amax           - Valor máximo contraste (255)")
        print("  [7] clahe_clip     - Intensidad CLAHE (2.0-4.0)")
        print("  [8] clahe_grid     - Tamaño tile (8 o 16)")
        print("  [9] shadow_thresh  - Umbral sombras (70-100)")
        print("  [10] mode          - Modo de pipeline")
        print("\n💡 AJUSTE DE SOMBRAS:")
        print("  • shadow_thresh = 70  → Detecta sombras muy oscuras")
        print("  • shadow_thresh = 90  → Balance (recomendado)")
        print("  • shadow_thresh = 100 → Detecta sombras más suaves")
        print("\n💪 INTENSIDAD DE REALCE:")
        print("  • clahe_clip = 2.0 → Suave, menos ruido")
        print("  • clahe_clip = 3.0 → Balance (recomendado)")
        print("  • clahe_clip = 4.0 → Agresivo, máximo contraste")
        print("="*70 + "\n")
        sys.exit(1)
    
    input_video_path = sys.argv[1]
    
    # Parámetros del pipeline
    k = float(sys.argv[2]) if len(sys.argv) > 2 else 256
    gamma = float(sys.argv[3]) if len(sys.argv) > 3 else 3
    alow = float(sys.argv[4]) if len(sys.argv) > 4 else 0.95
    ahigh = float(sys.argv[5]) if len(sys.argv) > 5 else 0.95
    amin = float(sys.argv[6]) if len(sys.argv) > 6 else 0
    amax = float(sys.argv[7]) if len(sys.argv) > 7 else 255
    
    # Parámetros CLAHE y sombras
    clahe_clip = float(sys.argv[8]) if len(sys.argv) > 8 else 3.0
    clahe_grid_size = int(sys.argv[9]) if len(sys.argv) > 9 else 8
    clahe_grid = (clahe_grid_size, clahe_grid_size)
    shadow_threshold = int(sys.argv[10]) if len(sys.argv) > 10 else 90
    
    # Modo de pipeline
    pipeline_mode = sys.argv[11] if len(sys.argv) > 11 else 'selective'
    
    # Activar modo debug (muestra máscara de sombras)
    debug_mode = '--debug' in sys.argv
    
    print(f"\n{'='*70}")
    print(f"🎬 PIPELINE DE DETECCIÓN DE LÍNEAS VIALES")
    print(f"{'='*70}")
    print(f"📹 Video: {input_video_path}")
    print(f"🔧 Modo: {pipeline_mode.upper()}")
    print(f"{'='*70}")
    print(f"⚙️  CONFIGURACIÓN:")
    print(f"   • Gamma: {gamma}")
    print(f"   • CLAHE clipLimit: {clahe_clip}")
    print(f"   • CLAHE tileGrid: {clahe_grid}")
    print(f"   • Umbral sombras: {shadow_threshold}")
    print(f"   • Auto-contraste: [{alow}, {ahigh}]")
    if debug_mode:
        print(f"   • Modo DEBUG activado (visualiza máscaras)")
    print(f"{'='*70}\n")
    
    # Configurar salida
    if input_video_path.endswith('.mp4'):
        output_video_path = input_video_path.replace('.mp4', f'_processed_{pipeline_mode}.mp4')
    else:
        output_video_path = f'output_processed_{pipeline_mode}.mp4'
    
    # Abrir video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"❌ Error: No se pudo abrir el video en {input_video_path}")
        sys.exit(1)
    
    # Configurar escritor
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = int(width * 0.7)
    output_height = int(height * 0.7)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (output_width, output_height), isColor=False)
    
    # Variables de control
    paused = False
    speed_multiplier = 1.0
    frame_delay = max(1, int(1000 / fps))
    frame_count = 0
    
    print("⌨️  CONTROLES:")
    print("   [Espacio] - Pausar/Reanudar")
    print("   [+/-]     - Cambiar velocidad")
    print("   [R]       - Reiniciar video")
    print("   [Q]       - Salir")
    print("   [D]       - Toggle modo debug\n")
    print("🎬 Procesando video...\n")
    
    show_debug = debug_mode
    
    # Procesar video
    while cap.isOpened():
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Procesar fotograma
            if show_debug:
                # Modo debug: mostrar máscara de sombras
                processed_color, shadow_mask = process_frame_debug(
                    frame, shadow_threshold, clahe_clip, clahe_grid
                )
                processed_frame = cv2.cvtColor(processed_color, cv2.COLOR_BGR2GRAY)
                processed_frame = mascara(processed_frame)
                
                # Mostrar máscara de sombras
                cv2.imshow('Máscara de Sombras (Debug)', shadow_mask)
            else:
                processed_frame = process_frame(
                    frame, k, gamma, alow, ahigh, amin, amax,
                    clahe_clip, clahe_grid, shadow_threshold, pipeline_mode
                )
            
            # Mostrar resultados
            frame_resized = cv2.resize(frame, (output_width, output_height))
            cv2.imshow('Video Original', frame_resized)
            cv2.imshow('Video Procesado (Detección de Líneas)', processed_frame)
            
            # Guardar
            out.write(processed_frame)
            frame_count += 1
            
            if frame_count % 30 == 0:
                print(f"⏳ Procesados {frame_count} frames...", end='\r')
        
        # Control de teclado
        current_delay = max(1, int(frame_delay / speed_multiplier))
        key = cv2.waitKey(current_delay) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print(f"\n{'⏸️  Pausado' if paused else '▶️  Reanudado'}")
        elif key == ord('r'):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            print("\n🔄 Video reiniciado")
        elif key == ord('d'):
            show_debug = not show_debug
            print(f"\n🐛 Debug: {'ON' if show_debug else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            speed_multiplier = min(speed_multiplier * 1.5, 5.0)
            print(f"\n⚡ Velocidad: {speed_multiplier:.1f}x")
        elif key == ord('-'):
            speed_multiplier = max(speed_multiplier / 1.5, 0.2)
            print(f"\n🐌 Velocidad: {speed_multiplier:.1f}x")
    
    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"\n\n{'='*70}")
    print(f"✅ PROCESAMIENTO COMPLETADO")
    print(f"{'='*70}")
    print(f"📊 Total frames: {frame_count}")
    print(f"💾 Video guardado: {output_video_path}")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()