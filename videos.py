import cv2
import os
import numpy as np
import random
from glob import glob
import mediapipe as mp

# Inicializa o MediaPipe Holistic
mp_holistic = mp.solutions.holistic


def landmark_vector_from_results(results):
    """Converte pose e maos em um vetor normalizado, ignorando rosto."""
    landmark_sets = (
        results.pose_landmarks,
        results.left_hand_landmarks,
        results.right_hand_landmarks)
    expected_sizes = (33, 21, 21)
    groups = []
    visible_points = []
    for landmark_set, expected_size in zip(landmark_sets, expected_sizes):
        if landmark_set:
            group = np.asarray([(lm.x, lm.y, lm.z)
                                for lm in landmark_set.landmark],
                               dtype=np.float32)
            groups.append(group)
            visible_points.append(group)
        else:
            groups.append(np.zeros((expected_size, 3), dtype=np.float32))

    if results.pose_landmarks:
        pose = groups[0]
        center = (pose[11] + pose[12]) / 2.0
        shoulder_delta = pose[12] - pose[11]
        scale = float(np.linalg.norm(shoulder_delta[:2]))
        angle = float(np.arctan2(shoulder_delta[1], shoulder_delta[0]))
    elif visible_points:
        all_visible = np.concatenate(visible_points)
        center = all_visible.mean(axis=0)
        scale = float(np.max(np.linalg.norm(all_visible[:, :2] - center[:2], axis=1)))
        angle = 0.0
    else:
        return np.zeros(33 * 3 + 21 * 3 + 21 * 3, dtype=np.float32)

    pose, left_group, right_group = canonicalize_landmark_side(
        groups[0], groups[1], groups[2])
    groups = [pose, left_group, right_group]

    scale = max(scale, 1e-6)
    cosine = np.cos(-angle)
    sine = np.sin(-angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)

    normalized = []
    for group, landmark_set in zip(groups, landmark_sets):
        if landmark_set:
            group = group - center
            group[:, :2] = group[:, :2].dot(rotation.T)
            group /= scale
        normalized.append(group)
    return np.concatenate(normalized).astype(np.float32).reshape(-1)


def mirror_landmark_vector(vector):
    """Espelha horizontalmente um vetor de landmarks para simular sinais do lado oposto."""
    pose_size = 33 * 3
    hand_size = 21 * 3
    total = vector.shape[0]
    if total < pose_size + 2 * hand_size:
        return vector.copy()

    points = vector.reshape(-1, 3)
    pose = points[:33]
    left = points[33:33 + 21]
    right = points[33 + 21:33 + 42]

    pose[:, 0] = 1.0 - pose[:, 0]
    left[:, 0] = 1.0 - left[:, 0]
    right[:, 0] = 1.0 - right[:, 0]

    mirrored = np.concatenate([pose, right, left], axis=0)
    return mirrored.reshape(-1).astype(np.float32)


def canonicalize_landmark_side(pose, left_hand, right_hand):
    """Canoniza a lateralidade para deixar sempre o sinal em um lado padrao."""
    pose = pose.copy()
    left_hand = left_hand.copy()
    right_hand = right_hand.copy()

    if np.any(right_hand) and not np.any(left_hand):
        pose[:, 0] = 1.0 - pose[:, 0]
        mirrored_right = right_hand.copy()
        mirrored_right[:, 0] = 1.0 - mirrored_right[:, 0]
        return pose, mirrored_right, np.zeros_like(right_hand)
    return pose, left_hand, right_hand


def get_landmark_vector(frame, holistic):
    """Processa um frame e retorna landmarks pose + maos ou None."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    if not (results.pose_landmarks or results.left_hand_landmarks or
            results.right_hand_landmarks):
        return None
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


def generate_augmentation_params():
    """Sorteia uma configuracao fixa de transformacoes para um video."""
    # Os parametros sao gerados uma vez por pasta de aumento. Assim, todos os
    # frames de uma mesma sequencia recebem uma variacao visual consistente.
    params = {
        'angle': random.uniform(-20, 20),
        'scale': random.uniform(0.8, 1.2),
        'shear': random.uniform(-5, 5),
        'brightness': random.uniform(0.6, 1.4),
        'saturation': random.uniform(0.6, 1.4),
        'contrast': random.uniform(0.8, 1.2),
        'do_flip': random.random() < 0.5,
        'do_blur': random.random() < 0.3,
        'blur_ksize': random.choice([3, 5]),
        'noise_std': random.uniform(0, 10),
    }
    return params


def apply_augmentation(image, params):
    """Aplica transformacoes geometricas, de cor, blur e ruido ao frame."""
    if image is None or image.size == 0:
        return None

    if params.get('do_flip', False):
        image = cv2.flip(image, 1)

    h, w = image.shape[:2]
    cx, cy = w / 2, h / 2

    # 1) Cisalhamento horizontal em torno do centro vertical da imagem.
    theta_s = np.deg2rad(params['shear'])
    sh = np.tan(theta_s)
    # matriz de shear pivoteada no centro verticalmente
    M_shear = np.array([
        [1,       sh, -sh * cy],
        [0,       1,   0]
    ], dtype=np.float32)
    image = cv2.warpAffine(image, M_shear, (w, h),
                           borderMode=cv2.BORDER_REFLECT)

    # 2) Rotacao e escala uniforme, preservando a proporcao da imagem.
    M_rs = cv2.getRotationMatrix2D((cx, cy), params['angle'], params['scale'])
    image = cv2.warpAffine(image, M_rs, (w, h), borderMode=cv2.BORDER_REFLECT)

    # 3) Ajustes de cor no HSV: saturacao e brilho sao alterados sem separar
    # manualmente os canais de cor BGR.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= params['saturation']
    hsv[..., 2] *= params['brightness']
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # O addWeighted aumenta ou reduz o contraste em relacao ao preto.
    image = cv2.addWeighted(
        image, params['contrast'], np.zeros_like(image), 0, 0)

    # 4) Desfoque gaussiano opcional para simular pequenas perdas de nitidez.
    if params['do_blur']:
        k = params['blur_ksize']
        image = cv2.GaussianBlur(image, (k, k), 0)

    # 5) Ruido gaussiano simula variacoes do sensor da camera. A conversao
    # para inteiros maiores evita overflow durante a soma.
    noise = np.random.normal(
        0, params['noise_std'], image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return image


def create_output_directory(base_path, gesture, n_augs=3):
    """Cria e retorna as pastas raw e de aumento para uma nova sequencia."""
    # Cada gesto possui sequencias numeradas. A pasta raw guarda os frames
    # recortados; cada aug_N guarda uma versao aumentada desses mesmos frames.
    gesture_path = os.path.join(base_path, gesture)
    os.makedirs(gesture_path, exist_ok=True)
    existing = glob(os.path.join(gesture_path, "sequence_*"))
    # Usa a quantidade de sequencias existentes como proximo indice.
    seq_num = len(existing)
    out_dir = os.path.join(gesture_path, f'sequence_{seq_num}')
    raw_dir = os.path.join(out_dir, 'raw')
    aug_dirs = [os.path.join(out_dir, f'aug_{i+1}') for i in range(n_augs)]
    os.makedirs(raw_dir, exist_ok=True)
    for d in aug_dirs:
        os.makedirs(d, exist_ok=True)
    return raw_dir, aug_dirs


# ==================== CAMINHOS ====================
# Vídeos podem ficar no Google Drive (ex.: G: com Drive para Desktop).
# Use uma barra invertida entre pastas no Windows; ex.:
#   r"G:\.shortcut-targets-by-id\1oE-zIqZbRz2ez0t_V-LtSwaX3WOtwg9E\TCC - Aline e Gabi\sinais_treinados"
caminho_videos_originais = r".\sinais_treinados"
# Saída dos frames (raw + aug): melhor no disco local para treino rápido
caminho_local_temporario = '.\iframes'
# =================================================================

if __name__ == "__main__":
    # O bloco so executa quando este arquivo e chamado diretamente, evitando
    # iniciar o processamento quando suas funcoes forem importadas.
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
            raw_dir, aug_dirs = create_output_directory(
                caminho_local_temporario, gesture, n_augs=2)
            landmark_dirs = [raw_dir] + aug_dirs
            # Gera uma configuracao independente para cada pasta aug_N.
            aug_params_list = [generate_augmentation_params() for _ in aug_dirs]

            # Abre o video e le seus frames sequencialmente.
            cap = cv2.VideoCapture(vid_path)
            frame_count = 1
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Detecta o corpo e as maos, recortando somente a area util.
                roi, landmark_vector = get_dynamic_square_roi(
                    frame, holistic, return_landmarks=True)
                if roi is None:
                    # Frames sem landmarks nao entram no dataset.
                    continue
                # Todos os exemplos precisam ter o mesmo tamanho para serem
                # aceitos pela rede neural.
                roi_resized = cv2.resize(
                    roi, (224, 224), interpolation=cv2.INTER_AREA)
                # Salva a imagem original recortada antes das transformacoes.
                cv2.imwrite(os.path.join(
                    raw_dir, f'frame_{frame_count:04d}_raw.jpg'), roi_resized)
                for landmark_dir in landmark_dirs:
                    np.save(
                        os.path.join(
                            landmark_dir,
                            f'frame_{frame_count:04d}.npy'),
                        landmark_vector)
                for idx, params in enumerate(aug_params_list):
                    # Cria uma variante aumentada para cada configuracao.
                    aug_img = apply_augmentation(roi_resized, params)
                    aug_landmark = mirror_landmark_vector(landmark_vector) if params.get('do_flip', False) else landmark_vector
                    if aug_img is not None:
                        cv2.imwrite(
                            os.path.join(
                                aug_dirs[idx], f'frame_{frame_count:04d}.jpg'),
                            aug_img
                        )
                        np.save(
                            os.path.join(
                                aug_dirs[idx], f'frame_{frame_count:04d}.npy'),
                            aug_landmark
                        )
                frame_count += 1
            # Libera o arquivo de video e os recursos associados ao decoder.
            cap.release()
