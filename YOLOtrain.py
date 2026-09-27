import os
from datetime import datetime
from pathlib import Path
import sys
import time
import pyautogui
import numpy as np
import cv2 as cv
import torch
from fontTools.misc.cython import returns
from fontTools.misc.plistlib import end_string
from torch.xpu import device
from ultralytics import YOLO

# import pandas as pd

import json
from dataset_tools import encode_image_to_base64, active_window_by_title
import subprocess

# Adiciona o diretório do projeto biblioteca ao sys.path
caminho_projeto_bib = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FillForm'))
sys.path.append(caminho_projeto_bib)


# from DetectLib import InputDataBox, ImageBinary, ShowMe

# Fluxo de trabalho
# 1. Reúna as imagens:
#    Colete as imagens que serão usadas para treinar seu modelo. → OK
#       /PyCharm/YOLOProject/images/
# 2. Criar estrutura de pastas:
#    Siga o esquema apresentado acima (nome_do_seu_dataset, images, labels, train, val).
#       /PyCharm/YOLOProject/dataset/..
#       /PyCharm/YOLOProject/dataset/images   ../train  ../val
#       /PyCharm/YOLOProject/dataset/labes    ../train  ../val
# 3. Anotar imagens:
#    LabelImg para criar os arquivos .txt correspondentes a cada imagem.
#    Desenhar uma caixa delimitadora em torno de cada objeto
#       imagem1.jpg --> imagem1.txt
#       <class> <x_center> <y_center> <width> <height>
#       0 0.5234 0.4871 0.2345 0.1982
#       1 0.7652 0.8123 0.1567 0.2543
# 4. Dividir dados:
#    Separe as imagens e seus rótulos entre as pastas train, val (e test se necessário).
#       /PyCharm/YOLOProject/dataset/images  ../train  ../val
#       /PyCharm/YOLOProject/dataset/labels  ../train  ../val
# 5. Criar arquivo data.yaml: descreva os caminhos train, val e nomes das classes.
#       /PyCharm/YOLOProject/dataset/my_data.yaml
# 6. Treinamento:
#    Use os comandos da Ultralytics para iniciar o treinamento do seu modelo

class YOLOvar:
    data_yaml = "E:/PyCharm/YOLOProject/dataset/mydata.yaml"
    model_pt = "E:/PyCharm/YOLOProject/runs/detect/train-8/weights/best.pt"

    dir_yolo = "E:/PyCharm/YOLOProject"
    dataset = "E:/PyCharm/YOLOProject/dataset"
    dir_jpeg = "E:/PyCharm/YOLOProject/images"
    dir_labelme = "E:/PyCharm/LabelmeProject"
    dir_json = "E:/PyCharm/LabelmeProject/dataset/json"
    file_json = "E:/PyCharm/LabelmeProject/dataset/json"

    dir_yolo_train = 'E:/PyCharm/YOLOProject/dataset/labels/train'
    dir_yolo_val = 'E:/PyCharm/YOLOProject/dataset/labels/val'
    dir_images_train = 'E:/PyCharm/YOLOProject/dataset/images/train'
    dir_images_val = 'E:/PyCharm/YOLOProject/dataset/images/val'
    path_json = "E:/PyCharm/LabelmeProject/dataset/json"
    dic_id = {'Entry': 0, 'Combobox': 1, 'Checkbox': 2, 'Radiobutton': 3, 'Listbox': 4,
              'Button': 5, 'Shape': 6, 'Picture': 7, 'Table': 8, 'Iconbutton': 9,
              'Underline': 10, 'Switch': 11, 'Searchbutton': 12}


class BGR:
    preto = (0, 0, 0)
    branco = (255, 255, 255)
    azul = (255, 0, 0)
    verde = (0, 255, 0)
    vermelho = (0, 0, 255)
    amarelo = (0, 255, 255)
    ciano = (255, 255, 0)
    magenta = (255, 0, 255)
    cinza = (127, 127, 127)
    laranja = (0, 127, 255)
    rosa = (161, 141, 255)
    violeta = (79, 47, 79)
    marrom = (51, 64, 92)
    salmon = (66, 66, 111)
    limao = (50, 205, 50)
    roxo = (142, 35, 107)
    bronze = (112, 147, 219)
    bege = (15, 199, 235)
    neon = (255, 77, 77)
    cobre = (51, 115, 184)
    ouro = (25, 217, 217)
    red = (10, 35, 180)


class Labels:
    names = {0: 'Entry', 1: 'Combobox', 2: 'Checkbox', 3: 'Radiobutton',
             4: 'Listbox', 5: 'Button', 6: 'Shape', 7: 'Picture',
             8: 'Table', 9: 'Iconbutton', 10: 'Underline', 11: 'Switch', 12: 'Searchbutton'}
    id = {'Entry': 0, 'Combobox': 1, 'Checkbox': 2, 'Radiobutton': 3,
          'Listbox': 4, 'Button': 5, 'Shape': 6, 'Picture': 7,
          'Table': 8, 'Iconbutton': 9, 'Underline': 10, 'Switch': 11, 'Searchbutton': 12}
    cores = {
        0: BGR.red, 'Entry': BGR.red,
        1: BGR.vermelho, 'Combobox': BGR.vermelho,
        2: BGR.verde, 'Checkbox': BGR.verde,
        3: BGR.azul, 'Radiobutton': BGR.azul,
        4: BGR.rosa, 'Listbox': BGR.rosa,
        5: BGR.violeta, 'Button': BGR.violeta,
        6: BGR.amarelo, 'Shape': BGR.amarelo,
        7: BGR.ciano, 'Picture': BGR.ciano,
        8: BGR.magenta, 'Table': BGR.magenta,
        9: BGR.roxo, 'Iconbutton': BGR.roxo,
        10: BGR.neon, 'Underline': BGR.neon,
        11: BGR.marrom, 'Switch': BGR.marrom,
        12: BGR.preto, 'Searchbutton': BGR.preto
    }
    entry = 0
    combobox = 1
    checkbox = 2
    radiobutton = 3
    listbox = 4
    button = 5
    shape = 6
    picture = 7
    table = 8
    iconbutton = 9
    underline = 10
    switch = 11
    searchbutton = 12


def predict_best_model(numero=0):

    # 1. Encontra arquivo da imagem original
    image_path = f"images/imagem_{numero:03}.jpg"
    if not os.path.isfile(image_path):
        print("Arquivo não existe:", image_path)
        return 0

    # 2. Carregue o modelo treinado (ajuste o caminho se necessário)
    model = YOLO(YOLOvar.model_pt)

    # 3. Execute a predição
    results = model.predict(
        source=image_path,  # Pode ser uma imagem, pasta ou link
        imgsz=960,          # Resolução idêntica ao treino
        conf=0.25,          # Filtro de confiança (ignora palpites fracos)
        save=True,          # Salva a imagem com os widgets desenhados
        save_txt=False,     # Opcional: Salva as coordenadas em arquivos .txt
        device=0            # Usa sua RTX 5060 Ti para inferência ultra-rápida
    )
    exibir_resultados(results, image_path)
    print("Detecção concluída! Os resultados estão na pasta 'runs/detect/predict/'")
    return results


def avaliar_deteccoes(numero=0):

    # Carregar a imagem original
    image_path = f"images/imagem_{numero:03}.jpg"
    if not os.path.isfile(image_path):
        print("Arquivo não existe:", image_path)
        return 0

    # Carregar o seu modelo treinado
    model = YOLO(YOLOvar.model_pt)

    # Realizar a inferência
    results = model(image_path, conf=0.45, iou=0.8)
    # results = model('images/imagem_002.jpg', save=True)
    # print(results)
    exibir_resultados(results, image_path)
    return results


def exibir_resultados(results, image_path):

    # converte arquivo em imagem ndarray
    img = cv.imread(image_path)

    # Processar os resultados (haverá um item na lista de resultados para cada imagem de entrada)
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # Coordenadas [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy()  # Pontuações de confiança
        classes = result.boxes.cls.cpu().numpy().astype(int)  # Índices das classes
        names = result.names  # Nomes das classes (dicionário)

        # Iterar sobre cada detecção individual
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box
            label = f"{names[cls]}"
            score = f"{names[cls]}: {score:.2f}"
            color = Labels.cores[label]
            # Desenhar a caixa delimitadora
            cv.rectangle(img, (x1, y1), (x2, y2), color, 1)

            # Desenhar o rótulo e a confiança
            cv.putText(img, score, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Exibir a imagem com as detecções
    cv.imshow(f"{image_path}, {img.shape[1]} x {img.shape[0]}", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


def treinar_modelo_dataset_personalizado():

    # Carregar um modelo pré-treinado (recomendado para fine-tuning)
    model = YOLO('yolov8n.pt')
    print(model.model_name)

    # Treinar o modelo com seu dataset personalizado
    model.train(
        data=YOLOvar.data_yaml,  # 'seu_dataset.yaml'
        epochs=300,  # Aumentado para lidar com a alta variação de estilos da internet
        # imgsz=1280, # Força o YOLO a enxergar os widgets pequenos em alta definição
        imgsz=960, # Reduz uso VRAM ~43% comparado a 1280, mas mantém nitidez para widgets
        batch=16,  # 16GB VRAM aguenta batch 32 para o modelo Nano 640px, para mais pixels usar 16
        lr0=0.001,  # Taxa inicial ligeiramente menor para não quebrar o aprendizado de formas finas
        cos_lr=True,  # Decaimento cosseno ajuda o modelo a convergir melhor no final

        # --- Geometria Rígida (Crucial para Interfaces) ---
        # rect=True,  # Ativa treino retangular. Evita esticar/distorcer os recortes das janelas
        rect=False,  # DESATIVADO: Força lotes quadrados fixos, eliminando picos surpresa de memória

        # --- Redução Estável de Resolução e Carga ---
        # --- Parâmetros de Acuracidade e Augmentation Ajustados ---
        optimizer='AdamW',  # Melhor para contornos estruturados e textos nas caixas
        degrees=0.0,  # Desativado: Janelas não giram
        shear=0.0,  # Desativado: Distorções de perspectiva estragam formas de botões
        fliplr=0.0,  # Desativado: Interfaces têm padrão fixo esquerda->direita (ex: texto antes do checkbox)
        scale=0.2,  # Permite zoom de ±20% para simular resoluções de tela diferentes
        workers=2,  # Mantido baixo para proteger a RAM do PC
        cache=False,

        # --- Adaptação para Imagens da Internet ---
        hsv_v=0.4,  # Varia o brilho para simular temas claros (Light) e escuros (Dark Mode)
        hsv_s=0.2,  # Altera saturação (ajuda a lidar com cores berrantes de alguns softwares)
        mosaic=0.3,  # Reduzido drasticamente para não misturar layouts de sistemas diferentes na mesma imagem
        # mosaic=0,5 (padrão 1,0) para evitar colagens que quebrem a lógica visual de janelas

        # --- Estabilização Final ---
        close_mosaic=25,  # Desliga o mosaico nas últimas 25 épocas para focar na geometria exata dos widgets

        # --- Parâmetros de Validação e Monitoramento ---
        val=True,              # Roda a validação automática ao final de cada época
        plots=True,            # Salva matriz de confusão, curvas F1, PR e perdas na pasta /runs
        save=True,             # Garante o salvamento dos melhores pesos (best.pt)

        device=0
    )
    metrics = model.val()
    # print(results)
    # print(metrics)


def treinar_modelo_dataset_personalizado_2():
    # Carregar um modelo pré-treinado (recomendado para fine-tuning)
    model = YOLO('yolov8n.pt')
    print(model.model_name)

    # Treinar o modelo com seu dataset personalizado
    # modelo nano (yolov8n.pt) consome pouca memória VRAM (GPU)

    # Parâmetros de Treinamento Recomendados  NVIDIA GeForce RTX 5060 Ti 16GB VRAM
    # imgsz=640 ou imgsz=1280 tamanho da imagem:
    #   Use 640 como padrão. Se o seu conjunto de dados possui objetos muito pequenos,
    #   mude para 1280 para capturar mais detalhes
    # batch=32 ou batch=64, lotes maiores melhoram a estabilidade da normalização.
    #   * Se usar imgsz=1280, reduza para 16 ou 32 se necessário
    # epochs=150 com patience=50 para parada antecipada se estacionar a acurácia
    # dar mais épocas com uma boa taxa de aprendizado garante melhor ajuste fino.
    # optimizer="AdamW" costuma entregar convergência mais suave e melhor generalização que o SGD padrão
    # lr0=0,01 taxa de aprendizado inicial padrão que funciona bem,
    #  combinada com um agendador de decaimento cosseno (0,001 refinada)
    # close_mosaic=20 desativa o aumento de mosaico nas últimas 20 épocas para estabilizar
    #  o aprendizado final e refinar a localização das caixas
    # cos_lr=True ativa o agendamento de taxa de aprendizado cosseno para estabilizar o fim do treino

    # Ajustes de Aumento de Dados (Augmentation)
    #  Se o seu dataset for pequeno ou medianamente complexo, intensifique a robustez:
    #  mixup=0.1 ou 0.15 mistura imagens para robustez a oclusões
    #  degrees=10.0, scale=0.5, translate=0.1
    #  fliplr=0.5 espelhamento horizontal, se fizer sentido para o seu domínio

    # model=yolov8n.pt data=seu_dataset.yaml epochs=300 imgsz=640 batch=32 optimizer=AdamW close_mosaic=20 device=0

    results = model.train(data=YOLOvar.data_yaml, imgsz=1280, epochs=150, batch=32, patience=50, optimizer="AdamW",
                          lr0=0.01, cos_lr=True, mixup=0.1, close_mosaic=20, device=0)

    metrics = model.val()
    # print(results)
    # print(metrics)


def test_torch():
    x = torch.rand(5, 3)
    print(x)
    print(f"\nVersão numpy: {np.__version__}")
    print(f"\nVersão YOLO: {YOLO._version}")
    print(f"\nVersão Torch: {torch.__version__}")
    print("GPU Disponível: ", torch.cuda.is_available())
    print("GPU Conta:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("GPU Atual:", torch.cuda.current_device())
        print("Device:", torch.cuda.device(0))
        print("Nome da GPU:", torch.cuda.get_device_name(0))
    else:
        print("Nome da GPU: Nenhuma")


def run_labelme_for_image(image_path, popen=True, terminal=False):
    """ Inicia o labelme para anotar uma imagem específica."""
    dir_labelme = f"E:/PyCharm/LabelmeProject"
    # image_path = f"{dir_labelme}/dataset/json/imagem_{numero:03}.json"
    # image_path = f"E:/PyCharm/YOLOProject/images/imagem_{numero:03}.jpg"
    dir_json = f"{dir_labelme}/dataset/json"
    txt_labels = f"{dir_labelme}/labels.txt"
    flags = f"{dir_labelme}/image_flags.txt"

    # 1. Defina o caminho para o executável Python do ambiente virtual do LabelMe
    # python_exe = "E:/PyCharm/LabelmeProject/.venv/Scripts/python.exe"
    python_exe = r"C:/Users/rudma/AppData/Local/Programs/Python/Python313"
    labelme_exe = f"{python_exe}/Scripts/labelme.exe"

    # 2. Caminho executável do labelme
    command = [python_exe, "-m", "labelme", image_path,
               '-O', dir_json, '--labels', txt_labels, '--flags', flags]
    command = labelme_exe   # [, image_path]

    """
    Executa um comando no terminal por entrada do teclado pyautogui
    """
    if terminal:
        print("Executando Labelme.exe via terminal...")
        pyautogui.hotkey('alt', '5')  # Debug
        time.sleep(1)
        pyautogui.hotkey('alt', 'f12')  # Terminal
        # Aguarda para dar tempo de focar no terminal
        time.sleep(2)
        # Simula a digitação (interval=0,1 faz parecer humano)
        # linha de comando com parâmetros do labelme.exe: output, flags...
        pyautogui.write(command, interval=0.01)
        pyautogui.write(' ', interval=0.01)
        pyautogui.write(image_path, interval=0.01)
        pyautogui.write(' --output ', interval=0.01)
        pyautogui.write(dir_json, interval=0.01)
        pyautogui.write(' --flags ', interval=0.01)
        pyautogui.write(flags, interval=0.01)
        pyautogui.write(' --labels ', interval=0.01)
        pyautogui.write(txt_labels, interval=0.01)
        pyautogui.write(' --no-sort-labels', interval=0.01)

        # parametro --nodata não copia imagem inteira codificada em formato base64
        # dentro do arquivo de saída, gera arquivos JSON menores.

        # parametro --config carrega um arquivo de configuração customizado,
        # sobrescrevendo o arquivo padrão ~/.labelmerc.

        # Simula o pressionamento da tecla Enter
        pyautogui.press('enter')
        print(f"Comando '{command}' enviado.")
        time.sleep(2)
        # pyautogui.hotkey('alt', '5')  # Debug
        return True

    else:
        # 3. Use subprocess.run para chamar o labelme.
        try:
            print("Executando Labelme.exe por subprocess...")
            comando = (f"{command} {image_path} --output {dir_json} "
                       f"--flags {flags} --labels {txt_labels} --no-sort-labels")
            # Comando para iniciar o labelme. A flag -m executa um módulo como script.
            work_dir = dir_json   # f"{python_exe}/Scripts"
            if popen:
                # Inicia o .exe e passa direto para a próxima linha
                process = subprocess.Popen(comando, cwd=work_dir)
                print(f"Labelme iniciado com subprocess Popen: {process.pid}")
                # process.wait()  # Trava o Python aqui até o .exe fechar

                print(f"⏳ Aguardando a janela 'Labelme' aparecer...")
                # Loop que verifica a cada 1 segundos se a janela existe
                janela_aberta = False
                tentativas = 0
                while not janela_aberta and tentativas < 20:  # Limite de 20 segundos
                    # Busca todas as janelas abertas no Windows que contêm o título informado
                    janela = active_window_by_title('Labelme*')
                    if janela:
                        janela_aberta = True
                        print("\n✅ Janela detectada com sucesso!")
                    else:
                        print(".", end="", flush=True)  # Indicador visual de carregamento
                        time.sleep(1)
                        tentativas += 1
                    msg = process.poll()
                    if msg is not None:
                        msg = "(fechado)" if msg == 0 else f"Erro: {msg}"
                        print(f"\n⚠️ Programa 'Labelme' parou de rodar {msg}")

                if not janela_aberta:
                    print("\n⚠️ Tempo limite estourou e janela 'Labelme' não foi encontrada.")
                # print("Python liberado para continuar o código!")
                return process
            else:
                # Espera o processo terminar (labelme completar a execução)
                process = subprocess.run(comando,
                    cwd= work_dir,  #
                    shell=True,
                    capture_output=True,
                    check=True,  # Levanta exceção se o comando falhar (código de saída nâo 0)
                    # stdout=subprocess.PIPE, # Captura a saída padrão (stdout)
                    # stderr=subprocess.PIPE, # Captura erros (stderr)
                    text=True  # Decodifica a saída como texto (Python 3.5+)
                )
                msg = 'Sucesso' if process.returncode == 0 else f"Erro: {process.returncode}"
                print("Labelme iniciado com subprocess run:", msg)
                return process
        except FileNotFoundError:
            print(f"Erro: O executável Python ou labelme não foi encontrado, {python_exe}")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar labelme (subprocess pid): {e}")
        return None


def processar_deteccao(image_path, confianca=0.25):

    # Quando está capturando a tela em tempo real (ex: usando pyautogui ou mss)
    # img = cv.imread(image_path)
    # ou quando a imagem precisa sofrer algum pré-processamento antes de ir para o YOLO
    # img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Carregar o seu modelo treinado
    model = YOLO(YOLOvar.model_pt)

    # Realizar a inferência com parâmetros ajustados
    # conf=0,5 (limiar de confiança), iou=0,4 (limiar de NMS)
    results = model(
        source=image_path,  # imagem já salva no disco usar caminho, ou ndarray
        imgsz=960,      # Resolução idêntica ao treino
        conf=0.25,      # Filtro de confiança (ignora palpites fracos)
        device=0,       # Usa sua RTX 5060 Ti para inferência ultra-rápida
        iou=0.8         # limiar de NMS
        )

    return results

# confirma se imagem json precisa ser processada
def confirma_processamento_json(arquivo_json):

    if not os.path.exists(arquivo_json):
        print(f"O arquivo ainda não foi processado")
        return True
    caminho_arquivo = Path(arquivo_json)

    # 1. Obter a data do sistema atual (apenas ano, mês e dia)
    data_atual = datetime.now().date()
    # 2. Obter a data de modificação do arquivo (apenas ano, mês e dia)
    timestamp_modificacao = caminho_arquivo.stat().st_mtime
    data_modificacao = datetime.fromtimestamp(timestamp_modificacao).date()
    # 3. Comparar as duas datas
    if data_modificacao == data_atual:
        print(f"O arquivo já foi modificado hoje! {data_modificacao}")
    else:
        print(f"Existe um arquivo processado em: {data_modificacao}")

    # Confirma processamento da imagem json
    pyautogui.hotkey('alt', '4')  # move cursor para área 'Run'
    time.sleep(1)
    msg = input(f"Arquivo já processado! Prosseguir s/N ?: ")

    if msg.lower() == "s":
        return True
    else:
        return False

# Criar novo arquivo imagem e anotações formato json
def criar_anotacao_formato_json(numero=0, json_path=None):

    dir_labelme = YOLOvar.dir_labelme
    file_json = json_path or f"{YOLOvar.dir_json}/imagem_{numero:03}.json"
    # processar deteccao de objetos modelo treinado 'best.pt'
    image_jpeg = f"{YOLOvar.dir_jpeg}/imagem_{numero:03}.jpg"
    image_path = os.path.basename(image_jpeg)

    if not os.path.exists(image_jpeg):
        print(f"O arquivo {image_path} não existe")
        return False

    # verifica se tem um json processado recentemente
    resp = confirma_processamento_json(file_json)
    if not resp:
        return False

    # retorna uma lista de resultados para cada imagem de entrada
    results = processar_deteccao(image_jpeg)

    img_flags = {
        "Boa_Qualidade": False,
        "Media_Qualidade": False,
        "Baixa_Qualidade": False,
        "Borrada_Imagem": False,
        "Escura_Imagem": False,
        "Classic": False,
        "Modern": False,
        "Office": False,
        "Web": False,
        "Figure": False,
        "Photo": False,
        "SAP": False
    }

    # parâmetros do formato json
    label = 'Entry'
    group_id = 0
    flags = img_flags
    shape_type = 'rectangle'  # 'circle'
    description = ''
    mask = None
    x0, y0, x1, y1 = 0, 0, 0, 0
    points = [[x0, y0], [x1, y1]]
    dic_shape = {"label": label, "points": points, "group_id": group_id,
                 "description": description, "shape_type": shape_type,
                 "flags": flags, "mask": mask}
    img_base64 = "(funcao para gerar arquivo base_64)"
    width, height = results[0].orig_shape[1], results[0].orig_shape[0]
    list_shapes = []  # [dic_shape, dic_shape...]

    # estrutura do arquivo json
    # monta dicionario de conteudo do arquivo json para gravação
    dic_json = {"version": "5.9.1", "flags": flags,
                "shapes": list_shapes,
                "imagePath": image_path,
                "imageData": img_base64,
                "imageHeight": height,
                "imageWidth": width
                }

    # Processar os resultados
    # haverá um item na lista de resultados para cada imagem de entrada
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)  # Coordenadas [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy()  # Pontuações de confiança
        classes = result.boxes.cls.cpu().numpy().astype(int)  # Índices das classes
        names = result.names  # Nomes das classes (dicionário)
        width, height = result.orig_shape[1], result.orig_shape[0]

        list_shapes = []
        # Iterar sobre cada detecção individual, cada shape
        for box, score, cls in zip(boxes, scores, classes):
            x0, y0, x1, y1 = box
            label = f"{names[cls]}"
            # score = f"{names[cls]}: {score:.2f}"
            # color = Labels.cores[label]
            group_id = Labels.id[label]
            points = [[int(x0), int(y0)], [int(x1), int(y1)]]
            dic_shape = {"label": label, "points": points, "group_id": group_id,
                         "description": '', "shape_type": 'rectangle',
                         "flags": {}, "mask": None}
            list_shapes.append(dic_shape)

    # codifica imagem byte para formato base64 'utf-8'
    print(f'Processando imagem {file_json}...')
    img_base64, erro = encode_image_to_base64(image_jpeg)
    if erro:
        print(f'Erro: {erro}\nNão foi possivel gravar arquivo json {file_json}')
        return False

    # atualiza dicionario de conteúdo para gravar arquivo json
    dic_json['shapes'] = list_shapes
    dic_json['imagePath']  = image_path
    dic_json['imageData'] = img_base64
    dic_json["imageWidth"] = int(width)
    dic_json["imageHeight"] = int(height)

    # Abre o arquivo em modo de escrita ('w') e usa json.dump() para salvar os dados
    with open(file_json, 'w', encoding='utf-8') as obj_file:
        json.dump(dic_json, obj_file, ensure_ascii=False, indent=4)
    print(f'Imagem salva: {file_json}')
    return True


# abrir e carregar imagem no labelme
def abrir_imagem_labelme(numero=0, json_path=None):
    dir_labelme = YOLOvar.dir_labelme
    json_path = json_path or f"{dir_labelme}/dataset/json"
    file_json = f"{json_path}/imagem_{numero:03}.json"
    # Coringa '*' antes e/ou depois junto ao parametro para procurar texto em dentro do título
    # * no fim do parametro indica procura por títulos que iniciam com texto
    labelme_aberto = False
    handle = active_window_by_title('Labelme*', close_win=True)
    if handle:
        labelme_aberto = True
    if not labelme_aberto:
        print("Não encontrou janela ativa com Labelme")
        # executar Labelme
        print("Executando programa Labelme.exe")
        proc1 = run_labelme_for_image(file_json)
        if proc1:
            print('Diretório de saída definido (--output):', json_path)
            handle = active_window_by_title('Labelme')
            if handle:
                print('Abrindo diretório de imagens...:', json_path)
                labelme_aberto = True
                """
                pyautogui.hotkey('ctrl', 'u')  # abrir diretório de imagens
                time.sleep(2)
                pyautogui.typewrite(json_path.replace('/', '\\'))
                time.sleep(2)
                pyautogui.press('enter')
                time.sleep(2)
                pyautogui.press('enter')
                time.sleep(2)
                """
            else:
                print("Não foi possível abrir arquivo json no Labelme")

        # if proc1:
        #     print(f"Fechando labelme (PID: {proc1.pid})")
        #     proc1.terminate() # ou proc1.kill()
    else:
        try:
            # Envia o comando do teclado
            # Para teclas específicas (ex: Enter, Ctrl+C):
            print(f"Enviando comando 'Open' para a aplicativo Labelme...")
            time.sleep(1)
            pyautogui.click(100,5, duration=0.3)
            pyautogui.hotkey('ctrl', 's')  # salvar
            time.sleep(1)
            pyautogui.hotkey('ctrl', 'w')  # fechar
            time.sleep(1)
            pyautogui.hotkey('ctrl', 'o')  # abrir
            time.sleep(2)
            # Para texto:
            json_name = os.path.basename(file_json)
            pyautogui.typewrite(json_name)  # interval=0.1
            time.sleep(2)
            pyautogui.press('enter')
            time.sleep(2)

            # pyautogui.press('enter')
            print(f"Comando 'Open' enviado para a aplicativo Labelme")

        except Exception as e:
            print(f"Erro: {e}")

# Transpose, espelhar ou girar a imagem
def espelhar_imagem(numero=0, espelho=0):

    from PIL import Image, ImageOps
    # Carregar a imagem original
    image_jpeg = f"{YOLOvar.dir_jpeg}/imagem_{numero:03}.jpg"
    image_path = os.path.basename(image_jpeg)
    img = Image.open(image_jpeg)
    dic_ = {0 : Image.FLIP_LEFT_RIGHT,
            1: Image.FLIP_TOP_BOTTOM,
            2: Image.ROTATE_90,
            3: Image.ROTATE_180,
            4: Image.ROTATE_270,
            5: Image.TRANSPOSE,
            6: Image.TRANSVERSE,
            7: 'MIRROR'
    }
    if espelho == 7:
        # Espelha horizontalmente usando ImageOps
        img_espelhada = ImageOps.mirror(img)
    else:
        # Transpose imagem
        img_espelhada = img.transpose(dic_[espelho])

    # Salva a nova imagem
    img_espelhada.save(image_jpeg)
    print("Imagem espelhada:", image_jpeg)

def teste_vision(numero=780):

    # 0. Carregar a imagem principal (grande, palheiro)
    image_jpeg = f"{YOLOvar.dir_jpeg}/imagem_{numero:03}.jpg"
    img_principal = cv.imread(image_jpeg)

    # Lista arquivos de imagens do template (imagem pequena, agulha)
    caminho = Path(f"{YOLOvar.dir_jpeg}/templates")
    # Retorna apenas os arquivos (ignorando subpastas)
    templates = [f.name for f in caminho.iterdir() if f.is_file()]

    # 1. Carregar imagem template (imagem pequena, agulha)
    figura = f"{YOLOvar.dir_jpeg}/templates/{templates[0]}"
    template = cv.imread(figura)

    import vision_v2 as vision

    agulha = vision.Vision(agulha_img_path=template)
    rets = agulha.find(palheiro_img=img_principal)
    print(rets)

# Localiza imagens que contem outra imagem menor (rótulo)
# Retorna dicionario com nomes de imagens e cantos do retângulo do rótulo localizado
def pesquisa_imagens_por_rotulos(limite=3):

    # Dicionario com chave nome da imagem e valor os cantos rect do rótulo encontrado
    dic_img = {}
    # Lista arquivos de imagens a serem pesquisadas
    caminho = Path(f"{YOLOvar.dir_jpeg}")
    # Retorna apenas os arquivos (ignorando subpastas)
    imagens = [f.name for f in caminho.iterdir() if f.is_file()]

    # Lista arquivos de imagens dos rótulos usados na pesquisa (imagem pequena)
    caminho = Path(f"{YOLOvar.dir_jpeg}/templates")
    templates = [f.name for f in caminho.iterdir() if f.is_file()]

    conta = 0
    for imagem in imagens[777:]:
        # Carrega imagem principal (grande, palheiro)
        image_jpeg = f"{YOLOvar.dir_jpeg}/{imagem}"
        img_principal = cv.imread(image_jpeg, cv.IMREAD_GRAYSCALE)
        print(f"Localizando na imagem {imagem}...")
        for arquivo in templates:
            # 1. Carregar imagem template (imagem pequena, agulha)
            figura = f"{YOLOvar.dir_jpeg}/templates/{arquivo}"
            template = cv.imread(figura, cv.IMREAD_GRAYSCALE)
            rects = sift_detection(img_principal, template)
            if len(rects) > 0:
                dic_img[imagem] = rects
                print(f"\t✅ encontrado rotulo {arquivo}")
                conta += 1
                break
        if limite and limite == conta:
            break

    print("\nImagens com rótulos localizados:\n", dic_img)

    # Abre o arquivo para escrita na mesma pasta do script
    with (open("imagens_SAP.txt", "w", encoding="utf-8") as arquivo):
        # Itera sobre a chave e o valor ao mesmo tempo
        for chave, valor in dic_img.items():
            lst_pares = np.array(valor, dtype=np.int32).reshape(-1, 2).tolist()
            # Grava no formato "Chave: Valor/rect" e pula uma linha (\n)
            arquivo.write(f"{chave}: {lst_pares}\n")

    print("Arquivo gravado com sucesso!")

    if limite > 0:
        for img, rects in dic_img.items():
            image_jpeg = f"{YOLOvar.dir_jpeg}/{img}"
            img_colorida = cv.imread(image_jpeg)  # Para desenhar o resultado colorido
            # Desenhar o contorno (pode ser um quadrilátero inclinado/escalado) na imagem original
            # Usamos uma linha vermelha (BGR: 0, 0, 255) de espessura 3
            imagem_resultado = cv.polylines(img_colorida, [np.int32(rects)],
                                            True, (0, 0, 255), 3, cv.LINE_AA)

            # Exibir o resultado
            cv.imshow('Objeto Localizado com SIFT', imagem_resultado)
            cv.waitKey()
            cv.destroyAllWindows()


# Detectar objetos em imagem usando sift
def sift_detection(img_principal, img_localizar):

    coordenadas = []
    # Carregar as imagens
    # Lemos diretamente em tons de cinza (0) porque o SIFT analisa apenas a intensidade geométrica,
    # o que anula completamente a diferença de cores entre as duas imagens!
    template = img_localizar   # cv.imread(figura, cv.IMREAD_GRAYSCALE)
    # img_principal = cv.imread(image_jpeg, cv.IMREAD_GRAYSCALE)

    # Inicializar o detector SIFT
    sift = cv.SIFT_create()

    # Encontrar os pontos de interesse (keypoints) e os descritores de ambas as imagens
    kp_template, des_template = sift.detectAndCompute(template, None)
    kp_principal, des_principal = sift.detectAndCompute(img_principal, None)

    # Combinar os pontos de interesse usando o FLANN Matcher (rápido e preciso)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    # Encontra as duas melhores combinações para cada ponto
    matches = flann.knnMatch(des_template, des_principal, k=2)

    # Filtrar apenas as melhores combinações (Teste de Razão de Lowe)
    bons_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            bons_matches.append(m)

    # Verificar se encontramos pontos suficientes para garantir que é o objeto correto
    MIN_PONTOS_CORRESPONDENTES = 4

    if len(bons_matches) >= MIN_PONTOS_CORRESPONDENTES:
        # Extrair a localização dos pontos que deram match em ambas as imagens
        pts_template = np.float32([kp_template[m.queryIdx].pt for m in bons_matches]).reshape(-1, 1, 2)
        pts_principal = np.float32([kp_principal[m.trainIdx].pt for m in bons_matches]).reshape(-1, 1, 2)

        # Encontrar a matriz de Homografia (mapeamento geométrico de perspectiva e escala)
        M, mascara = cv.findHomography(pts_template, pts_principal, cv.RANSAC, 5.0)

        # Obter os cantos da imagem pequena (template)
        h, w = template.shape[:2]
        cantos_template = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        # Projetar os cantos do template na imagem principal usando a matriz geométrica calculada
        cantos_transformados = cv.perspectiveTransform(cantos_template, M)
        # print(f"Sucesso! Imagem localizada usando {len(bons_matches)} pontos de correspondência geométricos.")
        return cantos_transformados

    else:
        # print(f"Não foram encontrados pontos suficientes. Matches bons:"
        #       f" {len(bons_matches)}/{MIN_PONTOS_CORRESPONDENTES}")
        return []


# Detectar objetos em imagem
def detectar_objetos_imagem(numero):

    # 0. Carregar a imagem principal (grande, palheiro)
    image_jpeg = f"{YOLOvar.dir_jpeg}/imagem_{numero:03}.jpg"
    img_principal = cv.imread(image_jpeg, cv.IMREAD_GRAYSCALE)

    # Lista arquivos de imagens do template (imagem pequena, agulha)
    caminho = Path(f"{YOLOvar.dir_jpeg}/templates")
    # Retorna apenas os arquivos (ignorando subpastas)
    templates = [f.name for f in caminho.iterdir() if f.is_file()]
    coordenadas = []

    for arquivo in templates:
        # 1. Carregar imagem template (imagem pequena, agulha)
        figura = f"{YOLOvar.dir_jpeg}/templates/{arquivo}"
        template = cv.imread(figura, cv.IMREAD_GRAYSCALE)

        # Obter as dimensões da imagem pequena (largura e altura)
        h, w, _ = template.shape[:2]
        # 2. Realizar o Template Matching
        resultado = cv.matchTemplate(img_principal, template, cv.TM_CCOEFF_NORMED)

        # 3. Definir um limite de confiabilidade (threshold) de 80%
        limiar = 0.7
        loc = np.where(resultado >= limiar)
        coord = zip(*loc[::-1])
        # 4. Percorrer as ocorrências encontradas e desenhar um retângulo
        for pt in coord: # pt repassa as coordenadas (x, y)
            # Desenhar retângulo verde ao redor da imagem encontrada
            cv.rectangle(img_principal, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
            print(f'Ocorrência {arquivo} encontrada na posição X: {pt[0]}, Y: {pt[1]}')
        if type(coord) is list:
            coordenadas.append(coord)

    # 5. Mostrar a imagem resultante com as marcações
    print(coordenadas)
    cv.imshow('Resultado', img_principal)
    cv.waitKey()
    cv.destroyAllWindows()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    teste_torch = 0
    if teste_torch == 100:
        test_torch()
        exit()
    predicao = 1
    if predicao == 111:
        predict_best_model(numero=28)
        exit()
    avaliacao = 2
    if avaliacao == 222:
        avaliar_deteccoes(numero=28)
        exit()
    anotacao = 3
    if anotacao == 333:
        criar_anotacao_formato_json(numero=28)
        abrir_imagem_labelme(numero=28)
        exit()
    detectar = 4
    if detectar == 4:
        item = 780
        # detectar_objetos_imagem(numero=item)
        # teste_vision()
        pesquisa_imagens_por_rotulos()
        exit()

    # Recriar arquivo json de anotacao com uso IA,
    # aplicando dataset treinado (best.pt) para reconhecer shapes das imagens
    # Sequencia:
    # 1. pegar arquivo /images/imagem_00n.jpg
    # 2. carregar modelo best.pt e aplicar inferência à imagem
    # 3. visualizar imagem e resultados das caixas delimitadores retornadas
    # 4. salvar arquivo de anotacao das caixas delimitadoras em json na pasta dataset/json
    # 5. abrir o programa labelme e editar imagem final


    # Rodar pelo PyCharm altere aqui os argumentos de entrada no sys.argv
    # sys.argv = [sys.argv[0]] + ["--avaliar", "010"]
    # print(sys.argv)

    test_item = 779  # 781   # last 780

    # 656 duplicada
    imagem_json = None
    args = sys.argv
    args = args + ['abrir', f"{test_item}"]
    # args = args + ['espelhar', f"{test_item}"]
    # args = args + ['anotar', f"{test_item}"]

    if len(args) > 1:
        # 745,
        arg1 = args[1].casefold()
        arg2 = args[2] if len(args) > 2 else ''
        item = int(arg2) if arg2.isnumeric() else arg2
        if 'treinar' in arg1:
            treinar_modelo_dataset_personalizado()
        elif 'avaliar' in arg1:
            avaliar_deteccoes(numero=item)
        elif 'abrir' in arg1:
            abrir_imagem_labelme(numero=item)
        elif 'espelhar' in arg1:
            espelhar_imagem(numero=item, espelho=7)
        elif 'anotar' in arg1:
            print(f"Processando item {item}...")
            processado = criar_anotacao_formato_json(numero=item)
            if processado:
                abrir_imagem_labelme(numero=item)
        else:
            print(f"Não processado! Esperado um argumento treinar/avaliar/abrir/anotar {item}...")
    else:
        if test_item > 0:
            processado = criar_anotacao_formato_json(numero=test_item)
            if processado:
                abrir_imagem_labelme(numero=test_item)
        else:
            print("Nenhum parâmetro de entrada foi fornecido.")
            item = None

    print("YOLO train finalizado")

    ident1 = {'Entry': 0,    'Combobox': 1,   'Checkbox': 2,    'Radiobutton': 3,
              'Listbox': 4,  'Button': 5,     'Shape': 6,       'Picture': 7,
              'Table': 8,    'Iconbutton': 9, 'Underline': 10,  'Switch': 11,
              'Searchbutton': 12}
    # print(ident1)
    print("Operação finalizada")
