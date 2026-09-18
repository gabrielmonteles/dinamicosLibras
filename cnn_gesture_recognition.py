import os
import json
import argparse
import torch
from glob import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

# ------------------- MODELO LSTM -------------------
# A LSTM recebe a sequencia de vetores de landmarks (pose + maos) na ordem
# temporal e aprende o movimento do gesto ao longo da sequencia.


class LandmarkLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])


# ------------------- NORMALIZAÇÃO ESPACIAL -------------------
# Cada frame bruto traz a posicao dos pontos do jeito que a camera captou:
# a pessoa pode estar em qualquer lugar do quadro, mais perto ou mais longe
# da camera, com a camera levemente inclinada. As funcoes abaixo centralizam,
# escalam e rotacionam cada frame para que sobre so a forma relativa do
# gesto (posicao da mao em relacao ao corpo), no mesmo referencial pra
# qualquer video.

POSE_SIZE = 33
LEFT_HAND_SIZE = 21
RIGHT_HAND_SIZE = 21
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12


def normalize_landmarks_frame(vector):
    """Centraliza, escala e rotaciona um unico frame de landmarks.

    Centraliza no meio dos ombros, escala pela distancia entre eles e
    rotaciona para deixar a linha dos ombros na horizontal. Grupos ausentes
    (preenchidos com zeros na extracao) sao mantidos como zero, para nao
    inventar deteccao onde nao houve.
    """
    points = np.asarray(vector, dtype=np.float32).reshape(-1, 3).copy()
    pose = points[:POSE_SIZE]
    left_hand = points[POSE_SIZE:POSE_SIZE + LEFT_HAND_SIZE]
    right_hand = points[POSE_SIZE + LEFT_HAND_SIZE:
                        POSE_SIZE + LEFT_HAND_SIZE + RIGHT_HAND_SIZE]
    groups = (pose, left_hand, right_hand)

    pose_visible = np.any(pose)
    if pose_visible:
        center = (pose[LEFT_SHOULDER_IDX] + pose[RIGHT_SHOULDER_IDX]) / 2.0
        shoulder_delta = pose[RIGHT_SHOULDER_IDX] - pose[LEFT_SHOULDER_IDX]
        scale = float(np.linalg.norm(shoulder_delta[:2]))
        angle = float(np.arctan2(shoulder_delta[1], shoulder_delta[0]))
    else:
        # Sem pose detectada (raro, ja que maos costumam vir junto): usa o
        # centroide dos pontos visiveis como referencia, sem rotacionar.
        visible_groups = [group for group in groups if np.any(group)]
        if not visible_groups:
            return points.reshape(-1)
        visible = np.concatenate(visible_groups)
        center = visible.mean(axis=0)
        scale = float(np.max(np.linalg.norm(visible[:, :2] - center[:2], axis=1)))
        angle = 0.0

    scale = max(scale, 1e-6)
    cosine, sine = np.cos(-angle), np.sin(-angle)
    rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)

    normalized_groups = []
    for group in groups:
        if np.any(group):
            group = group - center
            group[:, :2] = group[:, :2] @ rotation.T
            group = group / scale
        normalized_groups.append(group)

    return np.concatenate(normalized_groups).reshape(-1).astype(np.float32)


def normalize_landmarks_sequence(vectors):
    """Aplica a normalizacao espacial a cada frame de uma sequencia."""
    return np.stack([normalize_landmarks_frame(vector) for vector in vectors])


# ------------------- DATASET POR SEQUÊNCIA -------------------


def resample_sequence(vectors, timestamps, num_samples):
    """Reamostra a sequencia para um numero fixo de passos temporais.

    Interpola cada dimensao do vetor de landmarks ao longo do tempo real
    (em segundos) em vez do indice do frame, entao videos gravados com fps
    diferentes ou de duracao diferente ficam no mesmo eixo de tempo
    normalizado (0 a 1) antes de serem comparados pelo modelo.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    n = len(vectors)
    if n == 1:
        return np.repeat(vectors, num_samples, axis=0)

    timestamps = np.asarray(timestamps, dtype=np.float64)
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        # Sem timestamps validos: distribui os frames existentes de forma
        # uniforme no tempo normalizado.
        t_norm = np.linspace(0.0, 1.0, n)
    else:
        t_norm = (timestamps - timestamps[0]) / duration

    target = np.linspace(0.0, 1.0, num_samples)
    resampled = np.empty((num_samples, vectors.shape[1]), dtype=np.float32)
    for dim in range(vectors.shape[1]):
        resampled[:, dim] = np.interp(target, t_norm, vectors[:, dim])
    return resampled


# ------------------- AUMENTO DE DADOS (SO NO TREINO) -------------------
# A normalizacao espacial ja remove posicao/escala/rotacao de forma
# deterministica, entao nao faz sentido reaplicar essas variacoes aqui.
# O que sobra pra simular sao coisas que a normalizacao nao cobre: o ruido
# de deteccao do MediaPipe (varia frame a frame) e o ritmo interno do
# gesto (a reamostragem so fixa a duracao total, nao a cadencia dentro
# dela).


def warp_timestamps(timestamps, strength=0.15):
    """Distorce levemente o ritmo interno da sequencia, mantendo inicio,
    fim e a ordem dos frames — simula uma pessoa sinalizando um pouco mais
    rapido em um trecho e mais devagar em outro."""
    timestamps = np.asarray(timestamps, dtype=np.float64)
    duration = timestamps[-1] - timestamps[0]
    if duration <= 0:
        return timestamps
    t_norm = (timestamps - timestamps[0]) / duration
    # Deslocamento suave (uma senoide com fase aleatoria) que vale zero nas
    # pontas, para nao alterar o inicio/fim da sequencia.
    phase = np.random.uniform(0, 2 * np.pi)
    offset = strength * np.sin(np.pi * t_norm) * np.sin(t_norm * 2 * np.pi + phase)
    t_warped = np.clip(t_norm + offset, 0.0, 1.0)
    # Garante ordem crescente mesmo apos a distorcao.
    t_warped = np.sort(t_warped)
    return t_warped * duration + timestamps[0]


def jitter_landmarks(vectors, noise_std=0.01):
    """Adiciona ruido gaussiano por frame, simulando a imprecisao de
    deteccao do MediaPipe. Mantem grupos ausentes (zero) como zero."""
    vectors = np.asarray(vectors, dtype=np.float32)
    points = vectors.reshape(vectors.shape[0], -1, 3)
    mask = np.any(points != 0, axis=2)
    noise = np.random.normal(0.0, noise_std, size=points.shape).astype(np.float32)
    points = points + noise * mask[..., None]
    return points.reshape(vectors.shape)


class GestureDataset(Dataset):
    """Indexa sequencias de vetores de landmarks (.npy) por gesto."""

    def __init__(self, base_path, max_len=30, use_raw=True, use_aug=True,
                 train_augment=False, split=None, val_ratio=0.33):
        """Indexa sequencias de landmarks e prepara sua conversao para o modelo."""
        self.sequences = []
        self.timestamps = []
        self.labels = []
        self.sequence_groups = []
        self.class_to_idx = {}

        # A ordem alfabetica torna o mapeamento classe -> indice deterministico.
        gestures = sorted(
            gesture for gesture in os.listdir(base_path)
            if os.path.isdir(os.path.join(base_path, gesture)))
        for idx, gesture in enumerate(gestures):
            gesture_path = os.path.join(base_path, gesture)
            self.class_to_idx[gesture] = idx

            # Cada subpasta sequence_N corresponde a um video processado.
            for seq_folder in sorted(os.listdir(gesture_path)):
                seq_path = os.path.join(gesture_path, seq_folder)
                if not os.path.isdir(seq_path):
                    continue

                # Timestamps (em segundos) de cada frame do video original,
                # salvos por videos.py ao lado de raw/. Usados para reamostrar
                # a sequencia pelo tempo real em vez do indice do frame.
                timestamps_path = os.path.join(seq_path, 'timestamps.npy')
                sequence_timestamps = (
                    np.load(timestamps_path)
                    if os.path.isfile(timestamps_path) else None)

                if use_raw:
                    # Adiciona a sequencia original, sem transformacoes de
                    # aumento previamente geradas.
                    raw_dir = os.path.join(seq_path, 'raw')
                    if os.path.isdir(raw_dir):
                        # timestamps.npy guarda os instantes dos frames, nao
                        # e um frame em si, entao fica de fora do padrao.
                        frames = sorted(glob(os.path.join(raw_dir, 'frame_*.npy')))
                        if frames:
                            self.sequences.append(frames)
                            self.timestamps.append(
                                sequence_timestamps
                                if sequence_timestamps is not None and
                                len(sequence_timestamps) == len(frames)
                                else np.arange(len(frames), dtype=np.float64))
                            self.labels.append(idx)
                            self.sequence_groups.append((idx, seq_folder))

                if use_aug:
                    # Tambem adiciona cada pasta aug_N como uma sequencia
                    # independente da mesma classe.
                    for sub in sorted(os.listdir(seq_path)):
                        if sub.startswith('aug_'):
                            aug_dir = os.path.join(seq_path, sub)
                            if os.path.isdir(aug_dir):
                                frames = sorted(glob(os.path.join(aug_dir, 'frame_*.npy')))
                                if frames:
                                    self.sequences.append(frames)
                                    self.timestamps.append(
                                        sequence_timestamps
                                        if sequence_timestamps is not None and
                                        len(sequence_timestamps) == len(frames)
                                        else np.arange(len(frames), dtype=np.float64))
                                    self.labels.append(idx)
                                    self.sequence_groups.append((idx, seq_folder))

        if split in ("train", "val"):
            groups_by_class = {}
            for group in sorted(set(self.sequence_groups)):
                groups_by_class.setdefault(group[0], []).append(group)

            validation_groups = set()
            for class_groups in groups_by_class.values():
                validation_count = max(1, int(round(len(class_groups) * val_ratio)))
                validation_groups.update(class_groups[:validation_count])

            keep = [
                (group not in validation_groups) == (split == "train")
                for group in self.sequence_groups
            ]
            self.sequences = [sequence for sequence, include in zip(self.sequences, keep) if include]
            self.timestamps = [ts for ts, include in zip(self.timestamps, keep) if include]
            self.labels = [label for label, include in zip(self.labels, keep) if include]
            self.sequence_groups = [group for group, include in zip(self.sequence_groups, keep) if include]

        self.max_len = max_len
        self.train_augment = train_augment

    def __len__(self):
        # O tamanho e o numero total de sequencias indexadas, e nao o numero
        # individual de landmarks existentes nas pastas.
        return len(self.sequences)

    def __getitem__(self, idx):
        try:
            # Recupera os caminhos dos frames, seus timestamps e o indice
            # numerico da classe.
            frames = self.sequences[idx]
            timestamps = self.timestamps[idx]
            label = self.labels[idx]
            vectors = np.stack([np.load(fpath) for fpath in frames])
            # Normaliza cada frame (posicao/escala/rotacao) antes de
            # interpolar no tempo, para a interpolacao misturar frames que
            # ja estao no mesmo referencial espacial.
            vectors = normalize_landmarks_sequence(vectors)
            if self.train_augment:
                # So no treino: distorce o ritmo interno antes de reamostrar
                # e adiciona ruido por frame depois, gerando uma variacao
                # nova a cada epoca em que essa sequencia e usada.
                timestamps = warp_timestamps(timestamps)
            resampled = resample_sequence(vectors, timestamps, self.max_len)
            if self.train_augment:
                resampled = jitter_landmarks(resampled)
            tensor_seq = torch.from_numpy(resampled).float()
            return tensor_seq, torch.tensor(label)
        except Exception as e:
            print(f"Erro no __getitem__ do idx {idx}: {e}")
            raise

# ------------------- TREINAMENTO -------------------


if __name__ == "__main__":
    # Este bloco executa o treinamento somente quando o arquivo e iniciado
    # diretamente, e nao quando suas classes sao importadas por outro modulo.
    start_time = time.time()

    # Reprodutibilidade do treino: fixa os geradores usados pelo PyTorch e
    # NumPy para que os resultados sejam mais comparaveis entre execucoes.
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    parser = argparse.ArgumentParser(description="Treina o classificador LSTM por landmarks.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--use-aug", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.33)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitado, mas nao esta disponivel neste ambiente.")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    print("Treinando em:", device)

    base_path = os.path.join(os.path.dirname(__file__), "iframes")
    # Dataset de treino: inclui raw e, se pedido, aug; aplica ruido/distorcao
    # de ritmo a cada epoca para reduzir overfitting no conjunto pequeno.
    dataset_train = GestureDataset(base_path, max_len=args.max_len,
                                   use_raw=True, use_aug=args.use_aug,
                                   train_augment=True,
                                   split="train", val_ratio=args.val_ratio)
    # Dataset de avaliacao: usa videos diferentes dos usados no treino, sem
    # aumento, para que as metricas mecam generalizacao real.
    dataset_eval = GestureDataset(base_path, max_len=args.max_len,
                                 use_raw=True, use_aug=args.use_aug,
                                 train_augment=False,
                                 split="val", val_ratio=args.val_ratio)
    if not dataset_train or not dataset_eval:
        raise RuntimeError(
            "Nao ha sequencias de landmarks suficientes. "
            "Execute videos.py novamente para gerar os dados necessarios."
        )
    # O DataLoader agrupa sequencias em lotes e embaralha apenas o treino.
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size,
                                 shuffle=True, num_workers=args.workers,
                                 pin_memory=device.type == "cuda")
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size,
                                 shuffle=False, num_workers=args.workers,
                                 pin_memory=device.type == "cuda")

    # Hiperparametros e estruturas para acompanhar o melhor resultado.
    best_acc = 0.0
    metrics_history = []
    landmark_input_size = 225
    hidden_size = 128
    num_classes = len(dataset_train.class_to_idx)
    num_epochs = args.epochs

    model_config = {
        "num_classes": num_classes,
        "max_len": args.max_len,
        "hidden_size": hidden_size,
        "landmark_input_size": landmark_input_size,
    }

    # Cria o modelo e move seus pesos para o mesmo dispositivo dos dados.
    model = LandmarkLSTMModel(landmark_input_size, hidden_size,
                              num_classes).to(device)
    # CrossEntropyLoss e apropriada para classificacao multiclasse; Adam
    # atualiza os pesos usando os gradientes calculados em cada lote.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    print(f"Classes: {dataset_train.class_to_idx}")
    print(f"Total sequências (treino): {len(dataset_train)}")
    print(f"Total sequências (validação): {len(dataset_eval)}")

    # Salva o mapeamento para que a inferencia possa traduzir o indice previsto
    # de volta para o nome do gesto.
    with open("class_to_idx.json", "w") as f:
        json.dump(dataset_train.class_to_idx, f)

    print("Treinamento iniciado.")
    for epoch in range(num_epochs):
        # Cada epoca percorre todas as sequencias uma vez.
        epoch_start = time.time()
        model.train()
        train_labels = []
        train_preds = []
        for sequences, labels in dataloader_train:
            # Move entradas e respostas para CPU ou GPU, conforme escolhido.
            sequences = sequences.to(device)
            labels = labels.to(device)
            # Zera gradientes antigos, calcula a previsao e propaga o erro para
            # atualizar os parametros do modelo.
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_labels.extend(labels.detach().cpu().numpy())
            train_preds.extend(outputs.detach().argmax(1).cpu().numpy())

        train_acc = accuracy_score(train_labels, train_preds)

        # Avaliacao sem gradientes: reduz consumo de memoria e impede alteracoes
        # nos pesos enquanto as previsoes sao medidas.
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for sequences, labels in dataloader_eval:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences)
                # Seleciona o indice da maior pontuacao para cada sequencia.
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        # Calcula metricas macro, dando o mesmo peso a cada classe mesmo que
        # o numero de exemplos por gesto seja diferente.
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds,
                               average='macro', zero_division=0)
        rec = recall_score(all_labels, all_preds,
                           average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        epoch_end = time.time()

        # Guarda as metricas e o tempo da epoca para analise posterior em CSV.
        metrics_history.append({
            "epoch": epoch + 1,
            "loss": loss.item(),
            "train_accuracy": train_acc,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "epoch_time": epoch_end - epoch_start
        })

        if acc > best_acc:
            # Mantem o checkpoint com a maior acuracia observada durante o
            # treinamento e salva a configuracao necessaria para recarrega-lo.
            best_acc = acc
            torch.save(model.state_dict(), 'cnn_lstm_best_model.pth')
            with open("model_config.json", "w") as f:
                json.dump(model_config, f, indent=2)
            print(f"🔖 Novo melhor modelo salvo! Accuracy: {acc:.4f}")

        print(
            f"Treino Accuracy: {train_acc:.4f} | Val Accuracy: {acc:.4f} | "
            f"Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | "
            f"loss: {loss.item()}")
        print(
            f"Epoch {epoch+1}/{num_epochs} - Tempo: {epoch_end - epoch_start:.2f} segundos", end='\n\n')

    # ------------------- SALVAR MODELO -------------------
    # Ao final, salva tambem o estado da ultima epoca, independentemente de
    # ela ser melhor que o checkpoint escolhido durante a avaliacao.

    torch.save(model.state_dict(), 'cnn_lstm_model.pth')
    with open("model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    print("✅ Modelo treinado e salvo como 'cnn_lstm_model.pth'")
    print("✅ Configuração salva em 'model_config.json'")

    df_metrics = pd.DataFrame(metrics_history)
    df_metrics.to_csv("metrics_history.csv", index=False)
    print("📊 Métricas salvas em metrics_history.csv")

    # ------------------- DIAGNÓSTICO -------------------
    # Curvas de treino x validação: divergencia entre as duas indica
    # overfitting; loss que nao cai indica problema de otimizacao/dados.
    fig_curves, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(df_metrics["epoch"], df_metrics["train_accuracy"], label="Treino")
    axes[0].plot(df_metrics["epoch"], df_metrics["accuracy"], label="Validação")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Acurácia")
    axes[0].set_title("Acurácia por época")
    axes[0].legend()
    axes[1].plot(df_metrics["epoch"], df_metrics["loss"])
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss (treino)")
    axes[1].set_title("Loss por época")
    fig_curves.tight_layout()
    fig_curves.savefig("training_curves.png")
    plt.close(fig_curves)
    print("📈 Curvas de treino/validação salvas em training_curves.png")

    # Matriz de confusao e relatorio por classe do melhor checkpoint salvo
    # durante o treino, para localizar com quais gestos o modelo confunde.
    model.load_state_dict(torch.load('cnn_lstm_best_model.pth', map_location=device))
    model.eval()
    best_labels = []
    best_preds = []
    with torch.no_grad():
        for sequences, labels in dataloader_eval:
            sequences = sequences.to(device)
            outputs = model(sequences)
            _, preds = torch.max(outputs, 1)
            best_labels.extend(labels.numpy())
            best_preds.extend(preds.cpu().numpy())

    idx_to_class = {v: k for k, v in dataset_train.class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(num_classes)]

    cm = confusion_matrix(best_labels, best_preds, labels=list(range(num_classes)))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv("confusion_matrix.csv")

    report = classification_report(best_labels, best_preds, labels=list(range(num_classes)),
                                   target_names=class_names, zero_division=0)
    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

    fig_cm, ax = plt.subplots(figsize=(max(6, num_classes), max(6, num_classes)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusão (melhor modelo)")
    threshold = cm.max() / 2 if cm.max() > 0 else 0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig_cm.colorbar(im, ax=ax)
    fig_cm.tight_layout()
    fig_cm.savefig("confusion_matrix.png")
    plt.close(fig_cm)
    print("📊 Matriz de confusão salva em confusion_matrix.png / confusion_matrix.csv")
    print("📄 Relatório por classe salvo em classification_report.txt")

    # Lista as sequencias de validacao que o melhor modelo errou, para
    # investigar diretamente quais videos estao confundindo o modelo.
    mismatches = [
        {
            "sequencia": group[1],
            "gesture_real": class_names[true],
            "gesture_predito": class_names[pred],
        }
        for true, pred, group in zip(best_labels, best_preds, dataset_eval.sequence_groups)
        if true != pred
    ]
    pd.DataFrame(mismatches).to_csv("misclassified_sequences.csv", index=False)
    print(f"🔍 {len(mismatches)} sequência(s) mal classificada(s) em misclassified_sequences.csv")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Tempo total de execução: {total_time:.2f} segundos")
    print(f"Tempo total de execução: {total_time/60:.2f} minutos")
