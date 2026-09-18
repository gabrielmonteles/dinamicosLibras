# Reconhecimento de sinais em LIBRAS

Projeto de reconhecimento de sinais em LIBRAS usando MediaPipe para extrair
landmarks do corpo e das mãos e uma LSTM para classificar sequências.

## Instalação

No PowerShell, dentro da pasta do projeto:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 1. Organizar os vídeos

Crie uma pasta para cada gesto dentro de `sinais_treinados`. O nome da pasta
será usado como nome da classe:

```text
sinais_treinados/
	oi/
		video_01.mp4
		video_02.mp4
	obrigado/
		video_01.mp4
	desculpa/
		video_01.mp4
```

Use nomes em minúsculas, sem espaços, por exemplo `eu_amo_voce`.

## 2. Gerar os landmarks

Antes de executar `videos.py`, confirme no arquivo que a saída está configurada
para a mesma pasta usada pelo treinamento:

```python
caminho_videos_originais = r".\sinais_treinados"
caminho_local_temporario = r".\iframes"
```

Depois, execute na raiz do projeto:

```powershell
python videos.py
```

O script percorre cada `.mp4`, processa os frames com MediaPipe Holistic e
salva os landmarks em arquivos `.npy`. Cada vetor possui 225 valores:

- 33 pontos da pose;
- 21 pontos da mão esquerda;
- 21 pontos da mão direita;
- 3 coordenadas (`x`, `y`, `z`) por ponto.

A estrutura gerada será semelhante a:

```text
iframes/
	oi/
		sequence_0/
			raw/
				frame_0001.npy
			aug_1/
				frame_0001.npy
			aug_2/
				frame_0001.npy
```

`raw` contém os landmarks originais. `aug_1` e `aug_2` são variações geradas
para aumentar os dados de treinamento. Frames sem landmarks detectados são
ignorados.

## 3. Treinar com landmarks

Execute o treinamento usando explicitamente o modo de landmarks:

```powershell
python cnn_gesture_recognition.py --input-mode landmarks --use-aug --epochs 15
```

Para treinar somente na CPU com menos épocas:

```powershell
python cnn_gesture_recognition.py --input-mode landmarks --device cpu --epochs 5
```

Opções disponíveis:

```text
--device        auto, cpu ou cuda
--max-len       quantidade de frames por sequência; padrão: 30
--batch-size    tamanho do lote; padrão: 1
--epochs        quantidade de épocas; padrão: 15
--use-aug       inclui aug_1 e aug_2 no treinamento
--train-augment aplica variações durante o carregamento
--val-ratio     proporção usada na validação; padrão: 0.33
```

O treinamento lê os arquivos `.npy` de `iframes/` e salva na raiz:

```text
cnn_lstm_best_model.pth
cnn_lstm_model.pth
class_to_idx.json
model_config.json
metrics_history.csv
```

## 4. Executar a aplicação

Depois que o treinamento terminar e os arquivos do modelo existirem, execute:

```powershell
python interface.py
```

Na janela da câmera:

- pressione `Espaço` para capturar uma sequência;
- aguarde a previsão do gesto;
- pressione `Esc` para sair.

A aplicação usa a webcam para extrair landmarks em tempo real. O modelo usado
quando `model_config.json` indica `landmarks` recebe vetores de landmarks, não
imagens.

## Estrutura principal

```text
cnn_gesture_recognition.py  treinamento
videos.py                   extração dos landmarks
interface.py                aplicação de reconhecimento
sinais_treinados/            vídeos organizados por gesto
iframes/                    landmarks em formato .npy
models e métricas            arquivos gerados pelo treinamento na raiz
```

## Inspecionar os landmarks

O script `inspect_landmarks.py` analisa os arquivos `.npy` sem alterar o
dataset. Ele valida dimensões, valores inválidos, pontos ausentes, movimento
entre frames e estatísticas de cada ponto.

Para analisar todo o dataset, incluindo `raw` e `aug_*`:

```powershell
python inspect_landmarks.py
```

Para analisar somente os dados originais:

```powershell
python inspect_landmarks.py --variant raw
```

Para analisar somente um gesto:

```powershell
python inspect_landmarks.py --gesture oi --variant raw
```

Para analisar uma sequência específica:

```powershell
python inspect_landmarks.py --sequence oi/sequence_0 --variant raw
```

Os resultados são salvos em `landmark_inspection/`:

```text
report.json      relatório completo e anomalias encontradas
sequences.csv    estatísticas por sequência
frames.csv       estatísticas por frame
	coordinates/     um CSV com as coordenadas de cada frame
anomalies.csv    somente os frames inválidos ou anômalos
points.csv       média, desvio e ausência de cada ponto
*.png            gráficos das sequências selecionadas
```

Em `coordinates/`, cada arquivo CSV corresponde a um frame e contém uma linha
para cada ponto, com as coordenadas `x`, `y` e `z`. Os arquivos são organizados
por gesto, sequência e variante.

Para visualizar os landmarks sobre o frame real do vídeo, execute primeiro:

```powershell
python videos.py
```

As imagens são salvas separadamente em:

```text
landmark_inspection/video_frames/<gesto>/<sequencia>/raw/frame_0001.jpg
```

Essas imagens são cópias do frame original com pose e mãos desenhadas por cima.
Elas servem apenas para inspeção e não são usadas no treinamento. Os arquivos
`.npy` continuam sendo a única entrada do modelo.

Os arquivos `.png` gerados por `inspect_landmarks.py` são gráficos de sequência
e movimento. O inspetor não gera uma imagem para cada frame. Para ver o frame
real com os landmarks desenhados, use os `.jpg` gerados por `videos.py`.

Para gerar apenas relatórios, sem gráficos detalhados:

```powershell
python inspect_landmarks.py --variant raw --max-plots 0
```