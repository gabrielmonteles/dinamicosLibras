import cv2
import os
import json
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
from PIL import Image

# Usa exatamente a mesma estratégia de ROI aplicada durante o treinamento. Essa
# consistência é importante: o modelo deve receber imagens com o mesmo recorte
# e a mesma região do corpo que viu durante o aprendizado.
from videos import get_dynamic_square_roi, get_landmark_vector
from cnn_gesture_recognition import LandmarkLSTMModel
# MediaPipe Holistic detecta rosto, mãos e pose em cada frame da câmera.
import mediapipe as mp

# ------ Configurações gerais -----------------------------------------------
# SCRIPT_DIR aponta para a pasta deste arquivo, e não necessariamente para a
# pasta atual do terminal. Assim, os arquivos do modelo continuam sendo
# encontrados quando o programa é iniciado a partir de outro diretório.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Pesos do melhor modelo salvo durante o treinamento.
model_path = os.path.join(SCRIPT_DIR, 'cnn_lstm_best_model.pth')
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Checkpoint do melhor modelo nao encontrado: {model_path}")
# Mapeia nomes dos gestos para índices numéricos usados pelo classificador.
class_to_idx_path = os.path.join(SCRIPT_DIR, 'class_to_idx.json')
# Guarda dimensões como max_len, hidden_size e número de entradas da LSTM.
model_config_path = os.path.join(SCRIPT_DIR, 'model_config.json')
# Este caminho não é usado para inferência. Ele só recebe uma cópia das
# sequências capturadas quando save_captures estiver habilitado.
dataset_path = r'C:\Users\Aline\Desktop\tcc_imagens\tcc_imagens'
# Todas as regiões de interesse são redimensionadas para 224 x 224 pixels,
# que é o tamanho esperado pelas camadas convolucionais do modelo.
frame_size = (224, 224)
# Quando True, a sequência capturada é salva com o nome da classe prevista.
save_captures = True
# Índice da câmera fornecido ao OpenCV. A câmera integrada normalmente é 0;
# valores diferentes selecionam outras câmeras conectadas.
camera_id = 0
# Frames descartados depois do ESPAÇO para permitir o posicionamento antes da
# sequência que será enviada ao modelo (15 x 30 ms, aproximadamente 0,45 s).
PREPARATION_FRAMES = 15

# ------ Valores padrão da arquitetura --------------------------------------
# Esses valores permitem iniciar a interface mesmo sem model_config.json.
# Quando o arquivo existe, seus valores têm prioridade na inicialização abaixo.
DEFAULT_MAX_LEN = 100
DEFAULT_CNN_OUTPUT = 32 * 56 * 56
DEFAULT_HIDDEN = 128

# ------ Modelo CNN + LSTM (igual ao treino) -------------------------------
class CNNLSTMModel(nn.Module):
    """Recria a arquitetura usada para produzir as previsões dos gestos."""

    def __init__(self, cnn_output_size, hidden_size, num_classes):
        """Define as camadas convolucionais, temporais e de classificação."""
        super(CNNLSTMModel, self).__init__()
        # A CNN processa cada frame individualmente. Os MaxPool2d reduzem a
        # imagem de 224x224 para 56x56 e aumentam os canais de 3 para 32.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # A LSTM recebe a sequência de vetores produzidos pela CNN e aprende
        # como o gesto evolui ao longo dos frames.
        self.lstm = nn.LSTM(input_size=cnn_output_size, hidden_size=hidden_size, batch_first=True)
        # A camada final transforma o estado temporal em uma pontuação por
        # classe. A maior pontuação será escolhida na inferência.
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """Recebe (lote, frames, canais, altura, largura) e retorna logits."""
        # O modelo foi definido para aceitar uma sequência por vez, mas mantém
        # a dimensão de lote para que a arquitetura seja compatível com o treino.
        batch_size, seq_len, C, H, W = x.size()
        c_out = []
        for i in range(seq_len):
            # Seleciona o frame i de todas as sequências do lote.
            cnn_out = self.cnn(x[:, i])
            # Achata os mapas de características para formar um vetor por frame.
            cnn_out = cnn_out.view(batch_size, -1)
            c_out.append(cnn_out)
        # Empilha os vetores na ordem temporal: lote x frames x características.
        cnn_out_seq = torch.stack(c_out, dim=1)
        # A LSTM analisa todos os frames; a saída de cada instante fica em lstm_out.
        lstm_out, _ = self.lstm(cnn_out_seq)
        # Apenas a saída do último instante representa a sequência completa.
        out = self.fc(lstm_out[:, -1])
        return out

# ------ Carregar o mapeamento de classes -----------------------------------
def load_class_map():
    """Retorna um dicionário que converte índice previsto em nome do gesto."""
    if os.path.isfile(class_to_idx_path):
        # O treinamento salva o mapa como nome -> índice. A interface precisa
        # do mapa inverso, índice -> nome, para exibir o resultado.
        with open(class_to_idx_path, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
        idx_to_class = {int(idx): name for name, idx in class_to_idx.items()}
        return idx_to_class
    if os.path.isdir(dataset_path):
        # Fallback para quando o JSON ainda não existe: usa os nomes das pastas.
        # A ordenação precisa ser a mesma usada pelo treinamento.
        gestures = sorted([d for d in os.listdir(dataset_path)
                          if os.path.isdir(os.path.join(dataset_path, d))])
        return {idx: g for idx, g in enumerate(gestures)}
    # Sem o mapa e sem um dataset local, não é possível traduzir a previsão.
    raise FileNotFoundError(
        f"Não encontrado {class_to_idx_path}. Rode o treino antes (cnn_gesture_recognition.py)."
    )

# ------ Carregar configuração do modelo ------------------------------------
def load_model_config():
    """Carrega os parâmetros necessários para reconstruir a arquitetura treinada."""
    if os.path.isfile(model_config_path):
        with open(model_config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    # Mantém a interface utilizável com os valores originais caso o treinamento
    # não tenha gerado o arquivo de configuração.
    return {
        "max_len": DEFAULT_MAX_LEN,
        "num_classes": None,
        "cnn_output_size": DEFAULT_CNN_OUTPUT,
        "hidden_size": DEFAULT_HIDDEN,
    }

# ------ Inicialização do modelo --------------------------------------------
# O mapa e a configuração devem ser carregados antes de criar a última camada,
# pois o número de classes e as dimensões da LSTM vêm desses arquivos.
class_map = load_class_map()
config = load_model_config()
max_len = config.get("max_len", DEFAULT_MAX_LEN)
num_classes = len(class_map)
cnn_output_size = config.get("cnn_output_size", DEFAULT_CNN_OUTPUT)
hidden_size = config.get("hidden_size", DEFAULT_HIDDEN)
input_mode = config.get("input_mode", "video")
landmark_input_size = config.get("landmark_input_size", 225)
print(f"Modelo carregado: {model_path}")
print(f"Modo de entrada: {input_mode}")

# A arquitetura precisa ser idêntica à usada no treinamento para que os pesos
# armazenados tenham exatamente os mesmos nomes e dimensões.
if input_mode == "landmarks":
    model = LandmarkLSTMModel(input_size=landmark_input_size,
                              hidden_size=hidden_size,
                              num_classes=num_classes)
else:
    model = CNNLSTMModel(cnn_output_size=cnn_output_size,
                         hidden_size=hidden_size, num_classes=num_classes)
# map_location=cpu permite carregar o arquivo mesmo quando CUDA não está
# disponível. A interface atual faz a inferência na CPU.
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# Desliga o comportamento de treinamento, como dropout, caso a arquitetura
# venha a receber esse tipo de camada no futuro.
model.eval()

# A normalização deve ser igual à utilizada no dataset de treinamento.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ------ Salvar uma sequência capturada -------------------------------------
def save_sequence(frames, gesture_name):
    """Salva os frames BGR em uma pasta raw organizada pelo gesto previsto."""
    if not os.path.isdir(dataset_path):
        # Não cria a raiz automaticamente: um caminho incorreto não deve gerar
        # arquivos em um local inesperado.
        return
    gesture_dir = os.path.join(dataset_path, gesture_name)
    os.makedirs(gesture_dir, exist_ok=True)
    # Conta apenas sequências já existentes para escolher o próximo número.
    existing = [d for d in os.listdir(gesture_dir) if os.path.isdir(os.path.join(gesture_dir, d))]
    seq_num = len([d for d in existing if d.startswith("seq_")]) + 1
    seq_folder = os.path.join(gesture_dir, f"seq_{seq_num:03d}", "raw")
    os.makedirs(seq_folder, exist_ok=True)
    for i, frame in enumerate(frames):
        # OpenCV trabalha em BGR; PIL espera RGB para salvar as cores corretamente.
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img.save(os.path.join(seq_folder, f"frame_{i:03d}.jpg"))
    print(f"Sequência salva em: {seq_folder}")

# ------ Captura com ROI dinâmica (MediaPipe), igual ao treino --------------
# O objeto Holistic é reutilizado ao longo do vídeo para aproveitar o
# rastreamento entre frames e evitar recriar o detector a cada imagem.
mp_holistic = mp.solutions.holistic

# Abre a câmera selecionada e prepara o buffer da sequência atual. No Windows,
# CAP_DSHOW costuma evitar falhas de abertura com webcams USB e integradas.
cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
if not cap.isOpened():
    cap.release()
    raise RuntimeError(
        f"Não foi possível abrir a câmera {camera_id}. "
        "Verifique se ela está conectada, não está sendo usada por outro "
        "programa e teste os índices 0, 1 e 2."
    )

frame_buffer = []

print("Pressione ESPAÇO para capturar gesto e prever.")
print("ESC para sair. Use a mesma ROI (rosto/mãos) do treino.")
print(f"max_len={max_len} frames (igual ao treino)")

with mp_holistic.Holistic(static_image_mode=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as holistic:
    while True:
        # Primeiro mostra uma prévia contínua e aguarda uma tecla do usuário.
        ret, frame = cap.read()
        if not ret:
            # Interrompe se a câmera for desconectada ou não puder fornecer frame.
            break

        # Tenta localizar automaticamente rosto e mãos usando MediaPipe.
        roi_preview = get_dynamic_square_roi(frame, holistic)
        if roi_preview is None:
            # Sem detecção, usa um quadrado central como fallback visual para
            # permitir que o usuário se posicione antes de capturar.
            h, w = frame.shape[:2]
            side = min(h, w) // 2
            x0, y0 = (w - side) // 2, (h - side) // 2
            roi_preview = frame[y0:y0+side, x0:x0+side]
            cv2.putText(frame, "Posicione rosto e maos no centro", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        else:
            cv2.putText(frame, "ROI detectada - ESPACO para capturar", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Usa uma cópia para desenhar mensagens sem modificar o frame original.
        display_frame = frame.copy()
        cv2.imshow("Reconhecimento de Gestos - Libras", display_frame)
        # waitKey também permite que o OpenCV processe eventos da janela.
        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        elif key == 32:
            # Uma captura começa sempre vazia para não misturar frames de
            # tentativas anteriores.
            frame_buffer.clear()
            raw_frames = []

            # Dá tempo para a pessoa terminar de se posicionar. Esses frames
            # são processados pelo MediaPipe, mas não entram na sequência.
            capture_cancelled = False
            for preparation_index in range(PREPARATION_FRAMES):
                ret, preparation_frame = cap.read()
                if not ret:
                    capture_cancelled = True
                    break
                cv2.putText(
                    preparation_frame,
                    f"Preparando captura... {PREPARATION_FRAMES - preparation_index}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 165, 255), 2)
                cv2.imshow("Capturando...", preparation_frame)
                if cv2.waitKey(30) & 0xFF == 27:
                    capture_cancelled = True
                    break

            if capture_cancelled:
                break

            while len(frame_buffer) < max_len:
                # Continua lendo até reunir max_len frames com ROI detectada,
                # como no processamento dos vídeos do dataset.
                ret, frame = cap.read()
                if not ret:
                    break
                if input_mode == "landmarks":
                    landmark_vector = get_landmark_vector(frame, holistic)
                    if landmark_vector is None:
                        cv2.imshow("Capturando...", frame)
                        if cv2.waitKey(30) & 0xFF == 27:
                            capture_cancelled = True
                            break
                        continue
                    tensor = torch.from_numpy(landmark_vector).float()
                    raw_frames.append(frame.copy())
                    cv2.imshow("Capturando...", frame)
                else:
                    roi = get_dynamic_square_roi(frame, holistic)
                    if roi is None:
                        # Frames sem landmarks nao entram no dataset nem na
                        # sequencia enviada ao modelo.
                        cv2.imshow("Capturando...", frame)
                        if cv2.waitKey(30) & 0xFF == 27:
                            capture_cancelled = True
                            break
                        continue
                    roi_resized = cv2.resize(
                        roi, frame_size, interpolation=cv2.INTER_AREA)
                    raw_frames.append(roi_resized.copy())
                    img_rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
                    tensor = transform(img_rgb)
                    cv2.imshow("Capturando...", roi_resized)

                frame_buffer.append(tensor)
                # Aproximadamente 30 ms por frame mantém a captura próxima de
                # 33 frames por segundo e mantém a janela responsiva.
                cv2.waitKey(30)
                continue

            if capture_cancelled:
                break

            # Evita acessar frame_buffer[0] quando a câmera não entregou
            # nenhum frame durante a tentativa de captura.
            if not frame_buffer:
                print("Nenhum frame foi capturado; tente novamente.")
                continue

            # Se a câmera entregar menos frames que o esperado, completa a
            # sequência repetindo o último frame para preservar max_len.
            while len(frame_buffer) < max_len:
                frame_buffer.append(frame_buffer[-1].clone())
                raw_frames.append(raw_frames[-1])

            # Empilha os frames e adiciona a dimensão de lote esperada pelo
            # modelo: (1, max_len, 3, 224, 224).
            input_tensor = torch.stack(frame_buffer).unsqueeze(0)

            # A inferência não precisa de gradientes, economizando memória e tempo.
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.softmax(output, dim=1)[0]
                print("Probabilidades:")
                for class_index, probability in enumerate(probabilities):
                    print(f"  {class_map[class_index]}: {probability.item():.4f}")
                # Escolhe o índice com maior logit e converte-o para o nome do gesto.
                _, predicted = torch.max(output, 1)
                predicted_class = class_map[predicted.item()]
                print(f"Gesto reconhecido: {predicted_class}")

                if save_captures and os.path.isdir(dataset_path):
                    # Salva a sequência original somente quando a pasta de destino existe.
                    save_sequence(raw_frames, predicted_class)

                # Mostra o resultado por alguns instantes antes de voltar à prévia.
                display_frame = frame.copy() if ret else display_frame
                cv2.putText(display_frame, f"Gesto: {predicted_class}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.imshow("Reconhecimento de Gestos - Libras", display_frame)
                cv2.waitKey(1500)

cap.release()
cv2.destroyAllWindows()
