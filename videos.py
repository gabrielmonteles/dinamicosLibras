import cv2
import os
import shutil
import numpy as np
from glob import glob
import mediapipe as mp

# Inicializa o MediaPipe Holistic (linha de teste de diff)
mp_holistic = mp.solutions.holistic


def landmark_vector_from_results(results):
    """Converte pose e maos em um vetor de landmarks brutos, ignorando rosto."""
    landmark_sets = (
        results.pose_landmarks,
        results.left_hand_landmarks,
        results.right_hand_landmarks)
    expected_sizes = (33, 21, 21)
    groups = []
    for landmark_set, expected_size in zip(landmark_sets, expected_sizes):
        if landmark_set:
            group = np.asarray([(lm.x, lm.y, lm.z)
                                for lm in landmark_set.landmark],
                               dtype=np.float32)
        else:
            group = np.zeros((expected_size, 3), dtype=np.float32)
        groups.append(group)

    return np.concatenate(groups).astype(np.float32).reshape(-1)


def get_landmark_vector(frame, holistic, draw_landmarks=False):
    """Processa um frame e retorna landmarks, opcionalmente desenhando-os."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    if not (results.pose_landmarks or results.left_hand_landmarks or
            results.right_hand_landmarks):
        return None
    if draw_landmarks:
        drawing_utils = mp.solutions.drawing_utils
        pose_style = drawing_utils.DrawingSpec(
            color=(255, 0, 0), thickness=2, circle_radius=3)
        left_hand_style = drawing_utils.DrawingSpec(
            color=(0, 255, 0), thickness=2, circle_radius=3)
        right_hand_style = drawing_utils.DrawingSpec(
            color=(0, 0, 255), thickness=2, circle_radius=3)
        drawing_utils.draw_landmarks(
            frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=left_hand_style,
            connection_drawing_spec=left_hand_style)
        drawing_utils.draw_landmarks(
            frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=right_hand_style,
            connection_drawing_spec=right_hand_style)
        drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=pose_style,
            connection_drawing_spec=pose_style)
    return landmark_vector_from_results(results)


def get_dynamic_square_roi(frame, holistic, padding_factor=1.3,
                           draw_landmarks=False, return_landmarks=False):
    """Recorta a menor regiao quadrada que contem os landmarks detectados."""
    h_frame, w_frame, _ = frame.shape
    # O MediaPipe Holistic trabalha com imagens RGB, enquanto o OpenCV
    # normalmente entrega os frames no formato BGR.
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)

    landmarks = []
    # Ignora landmarks do rosto para manter o foco em pose e mãos, que
    # carregam o movimento relevante dos sinais de LIBRAS.
    if results.left_hand_landmarks:
        landmarks.extend([(lm.x, lm.y)
                         for lm in results.left_hand_landmarks.landmark])
    if results.right_hand_landmarks:
        landmarks.extend([(lm.x, lm.y)
                         for lm in results.right_hand_landmarks.landmark])
    if results.pose_landmarks:
        landmarks.extend([(lm.x, lm.y)
                         for lm in results.pose_landmarks.landmark])

    # Sem deteccao nao existe uma regiao confiavel para recortar.
    if not landmarks:
        return (None, None) if return_landmarks else None

    if draw_landmarks:
        drawing_utils = mp.solutions.drawing_utils
        drawing_utils.draw_landmarks(
            frame, results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS)
        drawing_utils.draw_landmarks(
            frame, results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS)
        drawing_utils.draw_landmarks(
            frame, results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS)

    # Os landmarks usam coordenadas normalizadas (0 a 1); converte-as para
    # pixels da imagem atual para calcular a caixa delimitadora.
    coords = np.array([[lm_x * w_frame, lm_y * h_frame]
                      for lm_x, lm_y in landmarks])
    x_min, y_min = np.min(coords, axis=0)
    x_max, y_max = np.max(coords, axis=0)

    box_w = x_max - x_min
    box_h = y_max - y_min
    center_x = x_min + box_w / 2
    center_y = y_min + box_h / 2

    # Um quadrado baseado no maior lado evita cortar o corpo quando a pessoa
    # ocupa uma area mais larga ou mais alta no frame.
    side_length = max(box_w, box_h)
    square_size = int(side_length * padding_factor)

    # Adiciona margem ao redor da pessoa e centraliza o quadrado no conjunto
    # de landmarks encontrado.
    new_x_min = int(center_x - square_size / 2)
    new_y_min = int(center_y - square_size / 2)
    new_x_max = new_x_min + square_size
    new_y_max = new_y_min + square_size

    # Impede que o recorte tente acessar pixels fora dos limites do frame.
    new_x_min = max(0, new_x_min)
    new_y_min = max(0, new_y_min)
    new_x_max = min(w_frame, new_x_max)
    new_y_max = min(h_frame, new_y_max)

    # Extrai a regiao de interesse (ROI) que sera usada no treinamento.
    square_roi = frame[new_y_min:new_y_max, new_x_min:new_x_max]

    # Um recorte vazio pode ocorrer em casos extremos de coordenadas invalidas.
    if square_roi.size == 0:
        return (None, None) if return_landmarks else None

    if return_landmarks:
        return square_roi, landmark_vector_from_results(results)
    return square_roi


def create_output_directory(base_path, gesture):
    """Cria e retorna a pasta raw para uma nova sequencia."""
    # Cada gesto possui sequencias numeradas, e a pasta raw guarda os frames
    # recortados.
    gesture_path = os.path.join(base_path, gesture)
    os.makedirs(gesture_path, exist_ok=True)
    existing = glob(os.path.join(gesture_path, "sequence_*"))
    # Usa a quantidade de sequencias existentes como proximo indice.
    seq_num = len(existing)
    out_dir = os.path.join(gesture_path, f'sequence_{seq_num}')
    raw_dir = os.path.join(out_dir, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    return raw_dir


# ==================== CAMINHOS ====================
# Vídeos podem ficar no Google Drive (ex.: G: com Drive para Desktop).
# Use uma barra invertida entre pastas no Windows; ex.:
#   r"G:\.shortcut-targets-by-id\1oE-zIqZbRz2ez0t_V-LtSwaX3WOtwg9E\TCC - Aline e Gabi\sinais_treinados"
caminho_videos_originais = r".\sinais_treinados"
# Saída dos frames (raw + aug): melhor no disco local para treino rápido
caminho_local_temporario = ".\iframes"
# Visualizacoes sao somente para inspecao; nao entram no treinamento.
caminho_visualizacoes = r".\landmark_inspection\video_frames"
# =================================================================


def reset_output_directories():
    """Limpa e recria as pastas geradas pelo pre-processamento."""
    output_directories = (
        caminho_local_temporario,
        caminho_visualizacoes,
    )
    for directory in output_directories:
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    # O bloco so executa quando este arquivo e chamado diretamente, evitando
    # iniciar o processamento quando suas funcoes forem importadas.
    reset_output_directories()
    print("Pastas de saída resetadas: iframes e landmark_inspection/video_frames")

    # --- ETAPA DE VERIFICACAO ---
    print("--- INICIANDO VERIFICAÇÃO ---")
    print(f"Procurando por vídeos .mp4 em: '{caminho_videos_originais}'")

    # Procura videos em todas as subpastas; o nome da subpasta identifica o
    # gesto (por exemplo, "oi" ou "obrigado").
    video_files = glob(os.path.join(caminho_videos_originais,
                       '**', '*.mp4'), recursive=True)

    print(f"--> Foram encontrados {len(video_files)} vídeos.")
    print("---------------------------\n")

    # Mantem uma unica instancia do MediaPipe para processar todos os frames
    # e aproveitar o rastreamento entre frames consecutivos.
    with mp_holistic.Holistic(static_image_mode=False,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5) as holistic:
        for vid_path in video_files:
            # A pasta imediatamente acima do arquivo de video representa a
            # classe do sinal que sera usada pelo modelo.
            gesture = os.path.basename(os.path.dirname(vid_path))
            raw_dir = create_output_directory(
                caminho_local_temporario, gesture)
            sequence_name = os.path.basename(os.path.dirname(raw_dir))
            visualization_dir = os.path.join(
                caminho_visualizacoes, gesture, sequence_name, "raw")
            os.makedirs(visualization_dir, exist_ok=True)
            # Abre o video somente para extrair landmarks dos frames.
            cap = cv2.VideoCapture(vid_path)
            # Os videos nao tem a mesma taxa de frames entre si; guardamos o
            # fps para calcular o instante real (em segundos) de cada frame
            # salvo, em vez de confiar apenas no indice do frame.
            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0:
                fps = 30.0
            frame_count = 1
            raw_frame_index = 0
            timestamps = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                timestamp = raw_frame_index / fps
                raw_frame_index += 1
                landmark_vector = get_landmark_vector(
                    frame, holistic, draw_landmarks=True)
                if landmark_vector is None:
                    # Frames sem landmarks nao entram no dataset.
                    continue
                np.save(
                    os.path.join(
                        raw_dir, f'frame_{frame_count:04d}.npy'),
                    landmark_vector)
                timestamps.append(timestamp)
                cv2.imwrite(
                    os.path.join(
                        visualization_dir, f'frame_{frame_count:04d}.jpg'),
                    frame)
                frame_count += 1
            # Timestamps em segundos, alinhados por indice aos frame_*.npy
            # salvos em raw_dir; usados para normalizar sequencias de fps
            # diferentes pelo tempo real, nao pela contagem de frames.
            # Fica fora de raw_dir para nao ser confundido com um frame por
            # ferramentas que leem todo *.npy dessa pasta (ex.: inspecao).
            np.save(
                os.path.join(os.path.dirname(raw_dir), 'timestamps.npy'),
                np.asarray(timestamps, dtype=np.float32))
            # Libera o arquivo de video e os recursos associados ao decoder.
            cap.release()
