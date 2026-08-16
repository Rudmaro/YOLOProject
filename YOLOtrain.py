import os
import sys
import time
import numpy as np
import pyautogui
import torch
import cv2
from fontTools.misc.plistlib import end_string
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
    data_yaml = "E:/PyCharm/YOLOProject/dataset/my_data.yaml"
    model_pt = "E:/PyCharm/YOLOProject/runs/detect/train4/weights/best.pt"

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
    dic_id = {'Entry': 0, 'Combobox': 1, 'Checkbox': 2, 'Radiobutton': 3, 'Listbox': 4, 'Button': 5,
              'Shape': 6, 'Picture': 7, 'Table': 8, 'Iconbutton': 9, 'Underline': 10, 'Switch': 11}


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
             8: 'Table', 9: 'Iconbutton', 10: 'Underline', 11: 'Switch', 12: 'Deformado'}
    id = {'Entry': 0, 'Combobox': 1, 'Checkbox': 2, 'Radiobutton': 3,
          'Listbox': 4, 'Button': 5, 'Shape': 6, 'Picture': 7,
          'Table': 8, 'Iconbutton': 9, 'Underline': 10, 'Switch': 11, 'Deformado': 12}
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
        12: BGR.preto, 'Deformado': BGR.preto
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
    deformado = 12


def avaliar_deteccoes(numero=0):
    # Carregar a imagem original
    image_path = f"images/imagem_{numero:03}.jpg"
    img = cv2.imread(image_path)

    # Carregar o seu modelo treinado
    model = YOLO(YOLOvar.model_pt)

    # Realizar a inferência
    results = model(img, conf=0.45, iou=0.8)
    # results = model('images/imagem_002.jpg', save=True)
    # print(results)

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
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

            # Desenhar o rótulo e a confiança
            cv2.putText(img, score, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Exibir a imagem com as detecções
    cv2.imshow(f"{image_path}, {img.shape[1]} x {img.shape[0]}", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def treinar_modelo_dataset_personalizado():
    # Carregar um modelo pré-treinado (recomendado para fine-tuning)
    model = YOLO('yolov8n.pt')
    print(model.model_name)

    # Treinar o modelo com seu dataset personalizado
    results = model.train(data=YOLOvar.data_yaml, epochs=30)  # imgsz=640
    metrics = model.val()
    print(results)
    print(metrics)


def test_torch():
    x = torch.rand(5, 3)
    print(x)
    print(f"\nVersão numpy: {np.__version__}")
    print(f"\nVersão YOLO: {YOLO._version}")
    print("GPU Disponível: ", torch.cuda.is_available())
    print("Nome da GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Nenhuma")


def run_labelme_for_image(image_path, wait=False, terminal=True):
    """ Inicia o labelme para anotar uma imagem específica."""
    dir_labelme = f"E:/PyCharm/LabelmeProject"
    # image_path = f"{dir_labelme}/dataset/json/imagem_{numero:03}.json"
    # image_path = f"E:/PyCharm/YOLOProject/images/imagem_{numero:03}.jpg"
    dir_json = f"{dir_labelme}/dataset/json"
    txt_labels = f"{dir_labelme}/labels.txt"
    flags = f"{dir_labelme}/image_flags.txt"

    # 1. Defina o caminho para o executável Python do ambiente virtual do LabelMe
    venv_python_exe = "E:/PyCharm/LabelmeProject/.venv/Scripts/python.exe"
    venv_python_exe = f"{dir_labelme}/.venv/Scripts/labelme.exe"

    # 2. Caminho para o "script" principal do labelme dentro do venv
    command = [venv_python_exe, "-m", "labelme", image_path,
               '-O', dir_json, '--labels', txt_labels, '--flags', flags]
    command = venv_python_exe   # [, image_path]

    """
    Executa um comando no terminal por entrada do teclado pyautogui
    """
    if terminal:
        pyautogui.hotkey('alt', '5')  # Debug
        time.sleep(1)
        pyautogui.hotkey('alt', 'f12')  # Terminal
        # Aguarda para dar tempo de focar no terminal
        time.sleep(2)
        # Simula a digitação (interval=0.1 faz parecer humano)
        pyautogui.write(command, interval=0.01)
        pyautogui.write(' ', interval=0.01)
        pyautogui.write(image_path, interval=0.01)
        # Simula o pressionamento da tecla Enter
        pyautogui.press('enter')
        print(f"Comando '{command}' enviado.")
        time.sleep(1)
        # pyautogui.hotkey('alt', '5')  # Debug
        return True

    else:
        # 3. Use subprocess.run para chamar o labelme.
        try:
            # Comando para iniciar o labelme. A flag -m executa um módulo como script.
            if wait:
                process = subprocess.run(command,
                    cwd=dir_labelme,   # work_dir
                    shell=True,
                    capture_output=True,
                    check=True,  # Levanta exceção se o comando falhar (código de saída != 0)
                    # stdout=subprocess.PIPE,  # Captura a saída padrão (stdout)
                    # stderr=subprocess.PIPE,  # Captura erros (stderr)
                    text=True  # Decodifica a saída como texto (Python 3.5+)
                )
                print(f"Labelme iniciado com PID: {process.pid}")
                return process
            else:
                process = subprocess.Popen(command, cwd=dir_labelme)
                print(f"Labelme iniciado com PID: {process.pid}")
                return process
        except FileNotFoundError:
            print(f"Erro: O executável Python ou labelme não foi encontrado, {venv_python_exe}")
        except subprocess.CalledProcessError as e:
            print(f"Erro ao executar labelme: {e}")
        return None


def processar_deteccao(image_path):
    # Carregar a imagem original
    # image_path = f"images/imagem_{numero:03}.jpg"
    img = cv2.imread(image_path)

    # Carregar o seu modelo treinado
    model = YOLO(YOLOvar.model_pt)

    # Realizar a inferência com parâmetros ajustados
    # conf=0,5 (limiar de confiança), iou=0,4 (limiar de NMS)
    results = model(img, conf=0.5, iou=0.8)
    # results = model('images/imagem_002.jpg')  # save=True
    # print(results)
    return results


def criar_anotacao_formato_json(numero=0, json_path=None):

    dir_labelme = YOLOvar.dir_labelme
    file_json = json_path or f"{YOLOvar.dir_json}/imagem_{numero:03}.json"
    # processar deteccao de objetos modelo treinado 'best.pt'
    image_jpeg = f"{YOLOvar.dir_jpeg}/imagem_{numero:03}.jpg"
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
        "Photo": False
    }
    # parametros do formato json
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
    image_path = image_jpeg
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
        return

    # atualiza dicionario de conteudo para gravar arquivo json
    dic_json['shapes'] = list_shapes
    dic_json['imagePath']  = image_jpeg
    dic_json['imageData'] = img_base64
    dic_json["imageWidth"] = int(width)
    dic_json["imageHeight"] = int(height)

    # Abre o arquivo em modo de escrita ('w') e usa json.dump() para salvar os dados
    with open(file_json, 'w', encoding='utf-8') as obj_file:
        json.dump(dic_json, obj_file, ensure_ascii=False, indent=4)
    print(f'Imagem salva: {file_json}')


# abrir e carregar imagem no labelme
def abrir_imagem_labelme(numero=0, json_path=None):
    dir_labelme = YOLOvar.dir_labelme
    json_path = json_path or f"{dir_labelme}/dataset/json"
    file_json = f"{json_path}/imagem_{numero:03}.json"
    handle = active_window_by_title('Labelme')
    if handle is None:
        # executar Labelme
        proc1 = run_labelme_for_image(file_json)
        if proc1:
            time.sleep(3)
            handle = active_window_by_title('Labelme')
            if handle:
                pyautogui.hotkey('ctrl', 'u')  # abrir diretório
                time.sleep(2)
                pyautogui.typewrite(json_path.replace('/', '\\'))
                time.sleep(2)
                pyautogui.press('enter')
                time.sleep(2)
        if handle is None:
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


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    test_torch()

    exit
    quit
    end

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

    test_item = 676   # last 675
    # 656 duplicada
    imagem_json = None
    args = len(sys.argv)
    if args > 1:
        # 745,
        arg1 = sys.argv[1].casefold()
        arg2 = sys.argv[2] if len(sys.argv) > 2 else ''
        arg2 = int(arg2) if arg2.isnumeric() else arg2

        if 'treinar' in arg1:
            treinar_modelo_dataset_personalizado()
        elif 'avaliar' in arg1:
            avaliar_deteccoes(numero=arg2)
        else:
            if arg1.isnumeric():
                item = int(arg1)
            elif 'anotar' in arg1:
                item = arg2 if arg2 else None
            else:
                item = None
            if item:
                print(f"Processando item {item}...")
                criar_anotacao_formato_json(numero=item)
                abrir_imagem_labelme(numero=item)
    else:
        if test_item > 0:
            criar_anotacao_formato_json(numero=test_item)
            abrir_imagem_labelme(numero=test_item)
        else:
            print("Nenhum parâmetro de entrada foi fornecido.")
            item = None

    print("YOLO train finalizado")

    ident1 = {'Entry': 0,    'Combobox': 1,   'Checkbox': 2,    'Radiobutton': 3,
              'Listbox': 4,  'Button': 5,     'Shape': 6,       'Picture': 7,
              'Table': 8,    'Iconbutton': 9, 'Underline': 10,  'Switch': 11,
              'Deformado': 12}
    print(ident1)
