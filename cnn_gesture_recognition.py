import os
import json
import argparse
import torch
from glob import glob
import numpy as np
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import time
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ------------------- MODELO CNN + LSTM -------------------
# A CNN extrai caracteristicas visuais de cada frame individualmente. A LSTM
# recebe essas caracteristicas na ordem temporal e aprende o movimento do
# gesto ao longo da sequencia.


class CNNLSTMModel(nn.Module):
    def __init__(self, cnn_output_size, hidden_size, num_classes):
        """Monta o modelo que combina extracao espacial e temporal."""
        super(CNNLSTMModel, self).__init__()
        # A entrada de cada frame tem 3 canais RGB. Duas convolucoes seguidas
        # de pooling reduzem a imagem de 224x224 para mapas de 56x56 e
        # aumentam o numero de canais de 3 para 32.
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # Cada frame vira um vetor com cnn_output_size valores. A LSTM
        # processa esses vetores em ordem e produz estados de tamanho
        # hidden_size.
        self.lstm = nn.LSTM(input_size=cnn_output_size,
                            hidden_size=hidden_size, batch_first=True)
        # O ultimo estado temporal e convertido em uma pontuacao por classe.
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """Classifica um lote de sequencias de imagens."""
        # Formato esperado: (lote, frames, canais, altura, largura).
        batch_size, seq_len, C, H, W = x.size()
        c_out = []
        for i in range(seq_len):
            # Aplica a mesma CNN a cada frame, compartilhando seus pesos.
            cnn_out = self.cnn(x[:, i])
            # Achata os mapas de caracteristicas para formar um vetor por
            # frame antes de entrega-lo a LSTM.
            cnn_out = cnn_out.view(batch_size, -1)
            c_out.append(cnn_out)
        # Reorganiza a lista no formato (lote, frames, caracteristicas).
        cnn_out_seq = torch.stack(c_out, dim=1)
        lstm_out, _ = self.lstm(cnn_out_seq)
        # Somente a saida do ultimo frame representa a sequencia completa.
        out = self.fc(lstm_out[:, -1])
        return out


class LandmarkLSTMModel(nn.Module):
    """Versão original de LSTM pura para sequências de landmarks."""

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1])


# ------------------- DATASET POR SEQUÊNCIA -------------------


def sample_frames(frames, num_samples):
    """Seleciona uma quantidade fixa de frames distribuida pelo video."""
    if len(frames) <= num_samples:
        # Sequencias curtas sao completadas repetindo o ultimo frame para que
        # todas tenham o mesmo comprimento aceito pelo modelo.
        return frames + [frames[-1]] * (num_samples - len(frames))
    # Para videos longos, escolhe frames igualmente espacados do inicio ao fim.
    idxs = np.linspace(0, len(frames) - 1, num_samples).astype(int)
    return [frames[i] for i in idxs]


class GestureDataset(Dataset):
    """Com train_augment=True aplica variações de cor/brilho (recomendado para poucos vídeos por classe)."""
    def __init__(self, base_path, max_len=30, use_raw=True, use_aug=True,
                 train_augment=False, split=None, val_ratio=0.33,
                 input_mode="video"):
        """Indexa sequencias de frames e prepara sua transformacao para o modelo."""
        self.sequences = []
        self.labels = []
        self.sequence_groups = []
        self.class_to_idx = {}
        self.input_mode = input_mode

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

                if use_raw:
                    # Adiciona a sequencia original, sem transformacoes de
                    # aumento previamente geradas.
                    raw_dir = os.path.join(seq_path, 'raw')
                    if os.path.isdir(raw_dir):
                        pattern = '*.npy' if input_mode == 'landmarks' else '*.[jp][pn]g'
                        frames = sorted(glob(os.path.join(raw_dir, pattern)))
                        if frames:
                            self.sequences.append(frames)
                            self.labels.append(idx)
                            self.sequence_groups.append((idx, seq_folder))

                if use_aug:
                    # Tambem adiciona cada pasta aug_N como uma sequencia
                    # independente da mesma classe.
                    for sub in sorted(os.listdir(seq_path)):
                        if sub.startswith('aug_'):
                            aug_dir = os.path.join(seq_path, sub)
                            if os.path.isdir(aug_dir):
                                pattern = '*.npy' if input_mode == 'landmarks' else '*.[jp][pn]g'
                                frames = sorted(glob(os.path.join(aug_dir, pattern)))
                                if frames:
                                    self.sequences.append(frames)
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
            self.labels = [label for label, include in zip(self.labels, keep) if include]
            self.sequence_groups = [group for group, include in zip(self.sequence_groups, keep) if include]

        self.max_len = max_len
        self.train_augment = train_augment
        # Normalizacao baseada no ImageNet, compatibilizando a escala das
        # entradas com a usada por modelos de visao treinados nesse conjunto.
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],  # Média ImageNet
                [0.229, 0.224, 0.225]   # Desvio padrão ImageNet
            )
        ])
        # Esta versao acrescenta variacoes aleatorias durante o carregamento;
        # portanto, uma mesma sequencia pode aparecer ligeiramente diferente
        # em cada epoca de treinamento.
        self.train_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        # O tamanho e o numero total de sequencias indexadas, e nao o numero
        # individual de imagens existentes nas pastas.
        return len(self.sequences)

    def __getitem__(self, idx):
        try:
            # Recupera os caminhos dos frames e o indice numerico da classe.
            frames = self.sequences[idx]
            label = self.labels[idx]
            selected_frames = sample_frames(frames, self.max_len)
            tensor_seq = []
            # A avaliacao usa apenas normalizacao; o treino pode usar
            # ColorJitter, conforme train_augment.
            t = self.train_transform if self.train_augment else self.transform
            for fpath in selected_frames:
                # Cada imagem e convertida para RGB, transformada em tensor e
                # normalizada antes de entrar na sequencia.
                if self.input_mode == "landmarks":
                    tensor = torch.from_numpy(np.load(fpath)).float()
                else:
                    img = Image.open(fpath).convert("RGB")
                    tensor = t(img)
                tensor_seq.append(tensor)
            tensor_seq = torch.stack(tensor_seq)
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

    parser = argparse.ArgumentParser(description="Treina o classificador CNN + LSTM.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-len", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--use-aug", action="store_true")
    parser.add_argument("--train-augment", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=0.33)
    parser.add_argument("--input-mode", choices=("video", "landmarks"),
                        default="video")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitado, mas nao esta disponivel neste ambiente.")
    device = torch.device(
        "cuda" if args.device == "cuda" or
        (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    print("Treinando em:", device)

    base_path = os.path.join(os.path.dirname(__file__), "iframes")
    # Dataset de treino: inclui raw e aug e ainda aplica ColorJitter em tempo
    # de carregamento, aumentando a variedade visual dos poucos videos.
    dataset_train = GestureDataset(base_path, max_len=args.max_len,
                                   use_raw=True, use_aug=args.use_aug,
                                   train_augment=args.train_augment,
                                   split="train", val_ratio=args.val_ratio,
                                   input_mode=args.input_mode)
    # Dataset de avaliacao: usa videos diferentes dos usados no treino e sem
    # ColorJitter para que as metricas mecam generalizacao sem aleatoriedade.
    dataset_eval = GestureDataset(base_path, max_len=args.max_len,
                                 use_raw=True, use_aug=args.use_aug,
                                 train_augment=False, split="val",
                                 val_ratio=args.val_ratio,
                                 input_mode=args.input_mode)
    if not dataset_train or not dataset_eval:
        raise RuntimeError(
            f"Nao ha sequencias suficientes para input_mode={args.input_mode}. "
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
    # Para imagens 224x224, os dois MaxPool2d reduzem cada dimensao pela
    # metade duas vezes: 224 -> 112 -> 56. Com 32 canais, o vetor tem
    # 32 * 56 * 56 caracteristicas.
    cnn_output_size = 32 * 56 * 56
    landmark_input_size = 225
    hidden_size = 128
    num_classes = len(dataset_train.class_to_idx)
    num_epochs = args.epochs

    model_config = {
        "num_classes": num_classes,
        "max_len": args.max_len,
        "cnn_output_size": cnn_output_size,
        "hidden_size": hidden_size,
        "input_mode": args.input_mode,
        "landmark_input_size": landmark_input_size,
    }

    # Cria o modelo e move seus pesos para o mesmo dispositivo dos dados.
    if args.input_mode == "landmarks":
        model = LandmarkLSTMModel(landmark_input_size, hidden_size,
                                  num_classes).to(device)
    else:
        model = CNNLSTMModel(cnn_output_size, hidden_size, num_classes).to(device)
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

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Tempo total de execução: {total_time:.2f} segundos")
    print(f"Tempo total de execução: {total_time/60:.2f} minutos")
