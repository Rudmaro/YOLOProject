# Grupo de funções auxiliares para preparar imagens e anotações do dataset personalizado
# Desenhar uma caixa delimitadora em torno de cada objeto
# imagem_001.jpg -→ imagem_001.txt
#    <class> <x_center> <y_center> <width> <height>
#    0 0.5234 0.4871 0.2345 0.1982
#    1 0.7652 0.8123 0.1567 0.2543

import cv2 as cv
import numpy as np
import os
import sys
import time
import pandas as pd
# from Demos.print_desktop import hwnd
from PIL import Image
import json
import base64

from ctypes import windll
import win32con  # , win32ui, win32api
import win32gui

# from fontTools.pens.pointPen import PointToSegmentPen
# from fsspec.registry import default
# from torch.distributed.tensor import empty
# from torchvision.utils import save_image

# Adiciona o diretório do projeto biblioteca ao sys.path
# projeto = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'FillForm'))
# sys.path.append(projeto)
# import DetectLib
# from DetectLib import InputDataBox, ImageBinary, ShowMe


def find_app(name: str, only_once=True):
    names = []
    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            texto = win32gui.GetWindowText(hwnd)
            if type(texto) is str and name.upper() in texto.upper():
                # print(hex(hwnd), texto)
                names.append(texto)
    win32gui.EnumWindows(winEnumHandler, None)
    if names:
        return names[0] if only_once else names
    else:
        return  ''


def get_active_window_name():
    window = win32gui.GetForegroundWindow()
    active_window_name = win32gui.GetWindowText(window)
    return active_window_name


def active_app_win32gui(app_tit: str):
    handle = None
    windll.user32.SetProcessDPIAware()
    active_window_name = win32gui.GetWindowText(win32gui.GetForegroundWindow())
    app_name = find_app(name=app_tit)
    try:
        if active_window_name != app_name:
            # Captura a tela da aplicação (handler)
            handle = win32gui.FindWindow(None, app_name)
            # win32gui.ShowWindow(handle, win32con.SW_SHOW)
            win32gui.ShowWindow(handle, win32con.SW_MINIMIZE)
            win32gui.ShowWindow(handle, win32con.SW_RESTORE)
            win32gui.ShowWindow(handle, win32con.SW_SHOWNOACTIVATE)
            time.sleep(1)
            # print('Navegador:', win_name)
        elif app_name:
            handle = win32gui.FindWindow(None, app_name)
            win32gui.ShowWindow(handle, win32con.SW_SHOW)
    except Exception as erro:
        print(f"{erro}\nJanela do aplicativo com titulo '{app_tit}' não encontrada")
        return None
    return handle

def active_window_by_title(window_name, close_win=False):
    try:
        # Encontra o identificador da janela com base no nome
        find_name = window_name.replace('*', '')
        windows_names = find_app(name=find_name, only_once=False)
        if not windows_names:
            return None
        ret_name = ''
        for win_name in windows_names:
            if win_name.startswith(find_name) and window_name.startswith(find_name):
                ret_name = win_name
                break
            if win_name.endswith(find_name) and window_name.endswith(find_name):
                ret_name = win_name
                break
        if not ret_name:
            if window_name.startswith('*') and window_name.endswith('*'):
                ret_name = windows_names[0]
            else:
                return None
        handle = win32gui.FindWindow(None, ret_name)
        if not handle:
            print(f"Janela '{window_name}' não encontrada.")
            return None
        if close_win:
            # Envia uma mensagem para a janela fechar de forma limpa
            print(f"Fechando janela '{window_name}'...")
            win32gui.PostMessage(handle, win32con.WM_CLOSE, 0, 0)
            # Alternativa mais agressiva:
            win32gui.PostMessage(handle, win32con.WM_DESTROY, 0, 0)
            return None
        else:
            # Se a janela estiver minimizada, restaura
            if win32gui.IsIconic(handle):
                win32gui.ShowWindow(handle, win32con.SW_RESTORE)
                win32gui.ShowWindow(handle, win32con.SW_MAXIMIZE)
            else:
                win32gui.ShowWindow(handle, win32con.SW_SHOW)
                win32gui.ShowWindow(handle, win32con.SW_MAXIMIZE)

            # Ativa a janela, trazendo-a para o primeiro plano
            win32gui.SetForegroundWindow(handle)
            # Espera um momento para a janela ser restaurada completamente
            time.sleep(2)

        return handle
    except IndexError:
        print(f"Janela com título '{window_name}' não encontrada.")
        return None
    except Exception as e:
        print(f"Erro: {e}")
        return None

def get_box_labels(filename, show=False):
    df = pd.DataFrame()
    erro = ''
    try:
        print(f"Processando arquivo {filename} ...")
        if type(filename) is str:
            img = ImageBinary(path=filename)
        else:
            img = filename
        w, h = img.width, img.height

        # classe Entry menores
        size = [(w * 0.02, h * 0.02), (w * 0.5, h * 0.15)]  # Entry
        entry = InputDataBox(image=img, size_rect=size, area_perc=20, max_search=1)
        df_box, err1 = entry.get_boxes()
        # recupera dados do desvio padrao e media
        std, med = entry.desvio, entry.media

        # classe Entry maiores
        size = [(w * 0.4, h * 0.02), (w * 0.97, h * 0.15)]  # Entry
        entry = InputDataBox(image=img, size_rect=size, area_perc=20, max_search=1)
        df_bx2, err2 = entry.get_boxes()

        # classe Checkbox options
        size = [(w * 0.01, h * 0.02), (h * 0.15, h * 0.15)]  # Checkbox
        check = InputDataBox(image=img, size_rect=size, area_perc=20, max_search=1, desvio=std, media=med)
        df_chk, err3 = check.get_boxes()

        # junta as classes de box Entry e Checkbox
        df = pd.concat([df_box, df_bx2, df_chk], ignore_index=True)
        df = df.drop_duplicates(ignore_index=True)

        erro = str(err1) + str(err2) + str(err3)
        if err1 and err2 and err3:
            print(f"Erro no processamento: {erro}")
        else:
            # classifica df pelo formato da caixa (quadrada ou retangulo)
            df = box_format_classify(df)
            # obtem retangulo medio de retangulos muitos proximos (sobrepostos)
            df = remove_retangulos_sobrepostos(df)
            df['width'] = img.width
            df['height'] = img.height

    except IOError as e:
        erro = str(e)
        print(f"Erro renomear {filename} : {erro}")
    except Exception as e:
        erro = str(e)
        print(f"Erro inesperado {filename}: {erro}")
    finally:
        # df['note'] = str(len(df))
        if show:
            df2 = df[df.columns[0:5]].copy()
            ShowMe(path=filename, df=df2)  # , scale=0, thickness=2
        return df, erro

# classifica conforme formato da caixa (quadrada ou retangular)
def box_format_classify(df: pd.DataFrame):

    _std = df['h'].std(ddof=1) or 1
    _mean = df['h'].mean()
    _min = (df['h'] * 0.8)   # minima altura para checar o quadrado
    _max = (df['h'] * 1.2)   # maxima altura para checar o quadrado
    z_score = (df['h'] - _mean) / _std
    # definindo o padrao da altura
    h_padrao = (z_score > -1.4) & (z_score < 1.65)
    # deformado: largura muito curto altura ou altura muito abaixo da media
    deformado = (df['w'] < (df['h'] * 0.78)) | (z_score < -1.4)
    # df['Deformado1'] = df['w'] < (df['h'] * 0.78)
    # df['Deformado2'] = df['z_score'] < -1.4
    # df['Listbox'] = df['z_score'] > 1.65
    checkbox = h_padrao & (df['w'] > _min) & (df['w'] < _max)

    df['label'] = 'Entry'
    df.loc[deformado, 'label'] = 'Deformado'
    df.loc[z_score > 1.65, 'label'] = 'Listbox'  # altura muito acima da média
    df.loc[checkbox, 'label'] = 'Checkbox'
    df = df.loc[:, 'x':'label']

    return df

# Anotacao de objetos em imagens: (x, y, h, w, label)
def image_annotation(df_in:pd.DataFrame, df_out:pd.DataFrame, num_img=None, lst_img=None, show=False):

    if lst_img is None:
        lst_img = [nr for nr in range(len(df_in))]
    num_img = num_img or len(lst_img)
    df_new = pd.DataFrame(columns=df_out.columns)
    idx: int; i:int; indice: int
    conta = 1
    row: pd.Series
    for row in df_in.itertuples():
        idx = row.Index
        if conta > num_img:
            break
        nome_arq = row.new_image
        if nome_arq is np.nan:
            continue
        if not os.path.isfile(nome_arq):
            continue
        seq = int(os.path.basename(nome_arq).split('.')[0][-3:])
        if seq not in lst_img:
            continue
        # processa labels
        df_box, erro = get_box_labels(nome_arq, show=show)
        if erro:
            dados = {'image_file': [nome_arq], 'seq': [int(idx)], 'idx': [0], 'label': ['erro'],
                      'x': [0], 'y': [0], 'w': [0], 'h': [0], 'width': [0], 'height': [0],
                     'note': [str(erro)]}
            df_box = pd.DataFrame(dados)
        else:
            df_box['image_file'] = nome_arq
            df_box['seq'] = idx
            df_box['idx'] = df_box.index.values
            df_box['note'] = str(len(df_box))

        # atualiza df_new
        df_new = pd.concat([df_new, df_box], ignore_index=True)
        # df_new = df_new.append(df_box)
        conta += 1

    # Concatenar os dataframes out com new
    df_new = pd.concat([df_new, df_out], ignore_index=True)
    # Remover linhas duplicadas
    df_new = df_new.drop_duplicates(subset=['image_file', 'x', 'y', 'w', 'h'], ignore_index=True)
    # atualiza coluna idx
    df2 = df_new.drop_duplicates(subset=['seq'])
    for row in df2.itertuples():
        ser_seq = df_new['seq'] == row.seq
        ser_idx = df_new.loc[ser_seq, 'idx']
        df_new.loc[ser_seq, 'idx'] = [i for i in range(len(ser_idx.index))]

    return  df_new


def identificar_formato_imagem(caminho_arquivo):
    """Identifica o formato lendo os bytes iniciais do arquivo."""
    try:
        with open(caminho_arquivo, 'rb') as f:
            header = f.read(8)  # Lê os primeiros 8 bytes
        if header.startswith(b'\x89\x50\x4e\x47'):
            return ".PNG"
        elif header.startswith(b'\xff\xd8\xff'):
            return ".JPG"
    except Exception as e:
        print(f"Erro ao ler iniciais do arquivo: {e}")
        return ''

def remover_arquivo(nome_arquivo, novo_arquivo):
    if nome_arquivo.casefold() == novo_arquivo.casefold():
        return False
    # tempo de 5 segundo para remover arquivo
    tempo = time.time() + 5
    while time.time() < tempo:
        time.sleep(1)
        try:
            os.remove(nome_arquivo)
            print(f"Arquivo {nome_arquivo} renomeado com sucesso!")
            return True
        except IOError as e:
            print(f"Erro ao remover: {e}")
        except Exception as e:
            print(f"Erro inesperado ao remover: {e}")
    print(f"Não removido arquivo antigo {nome_arquivo}!")
    return False


def sequenciar_imagens(diretorio, df: pd.DataFrame, sep='_'):
    conta = 1
    lista = os.listdir(diretorio)
    lista.sort()
    df = df.sort_values(by='new_image')

    for idx, row in df.iterrows():
        nome_arq = row.new_image
        if nome_arq is np.nan:
            df.loc[idx, 'note'] = 'Arquivo nulo; ' + str(df.loc[idx, 'note'])
            continue
        arq_atual = diretorio + '/' + nome_arq
        if not os.path.isfile(arq_atual):
            df.loc[idx, 'note'] = 'Não existe arquivo; ' + str(df.loc[idx, 'note'])
            continue
        sep1, sep2 = ('-', '_') if '-' in nome_arq else ('_', '-')
        base, part2 = nome_arq.split(sep1)
        extensao = part2[part2.find('.'):]  if '.' in part2 else ''
        novo_nome = f"{diretorio}/{base}{sep}{conta:03}{extensao}"
        while os.path.isfile(novo_nome):
            conta += 1
            novo_nome = f"{diretorio}/{base}{sep}{conta:03}{extensao}"
        try:
            os.rename(arq_atual, novo_nome)
            df.loc[idx, 'new_image'] = novo_nome
            conta += 1
        except IOError as e:
            df.loc[idx, 'note'] = f'{str(e)}; ' + str(df.loc[idx, 'note'])
            print(f"Erro renomear {arq_atual} : {e}")
        except Exception as e:
            df.loc[idx, 'note'] = f'{str(e)}; ' + str(df.loc[idx, 'note'])
            print(f"Erro inesperado {arq_atual}: {e}")

    return df


def formato_valido(base_name, sep='.'):
    extensao = base_name.casefold()
    while sep in extensao:
        texto, extensao = extensao.split(sep, 1)
    dic_ext = {'jpg': '.jpg', 'jpeg': '.jpg', 'png': '.jpg', 'avif': '.jpg', 'webp': '.jpg',
               'tiff': '.jpg', 'tif': '.jpg',  'jxr': '.jpg', 'jpe': '.jpg', 'gif': '.jpg', 'bmp': '.jpg'}
    if extensao in dic_ext:
        return dic_ext[extensao]
    return ''

def salvar_imagens_formato_jpeg(diretorio, df: pd.DataFrame, num_images: int = None):

    lista = os.listdir(diretorio)
    lista.sort()
    for idx, row in df.iterrows():
        new_path = str(row.new_image)
        if '.JPG' in new_path.upper():
            continue
        nome_arq = row.image_file
        # if row.status not in [200, 64] or nome_arq is np.nan:
        #     continue
        # if not os.path.exists(nome_arq):
        #    continue
        if not os.path.exists(new_path):
            continue
        # caminho_completo = nome_arq  # diretorio + '/' + nome_arq
        caminho_completo = new_path
        raiz, extensao = os.path.splitext(caminho_completo)
        try:
            img = Image.open(caminho_completo)
            extensao = formato_valido(img.format)
            if img.format and not extensao:
                print(f"Formato {img.format}:  {nome_arq} não é tipo esperado!")
                img.close()
                remover_arquivo(caminho_completo, '')
                df.loc[idx, 'new_image'] = 'removido'
                df.loc[idx, 'note'] = f"? = {img.format}"
                continue
            # imagem com palete de cores
            if img.mode in ("P", "PA"):
                df.loc[idx, 'note'] = "mode PA"
                # print(f"Arquivo {nome_arq} com palete de cores:", img.format)
                # img.close()
                # remover_arquivo(caminho_completo, '')
                # df.loc[idx, 'new_file'] = 'removido'
                # continue
            if img.size[0] < 150 and img.size[1] < 150:
                print("Tamanho da imagem pequena (< 150):", img.format)
                img.close()
                remover_arquivo(caminho_completo, '')
                df.loc[idx, 'note'] = "size < 150"
                df.loc[idx, 'new_image'] = 'removido'
                continue
            # Convert the image to 'RGB' mode
            if img.mode != "RGB":
                img = img.convert("RGB")
            caminho_imagem = f"{raiz}{extensao}"
            img.save(caminho_imagem, format='JPEG')
            img.close()
            df.loc[idx, 'new_image'] = os.path.basename(caminho_imagem)
            remover_arquivo(caminho_completo, caminho_imagem)

        except Exception as e:
            df.loc[idx, 'note'] = str(e)
            print(f"Erro ao tentar ler arquivo no Pillow: {e}")

    return df

# aumenta o tamanho das caixas de entradas, incremento % altura
def box_scaling(df_in: pd.DataFrame, scale: float = 1.25) -> pd.DataFrame:

    df = df_in.copy()
    #df['image_file'] = df_in['image_file']
    #df['seq'] = df_in['seq']
    #df['idx'] = df_in['idx']
    #df['label'] = df_in['label']
    #df['width'] = df_in['width']
    #df['height'] = df_in['height']
    #df['note'] = df_in['note']
    gap = ((df['h'] * scale) - df['h']).astype(int) / 2
    gap = gap.apply(lambda x: x if x > 0 else 1)
    x0, y0 = df['x'] - gap, df['y'] - gap
    x1, y1 = x0 + df['w'] + gap + gap, y0 + df['h'] + gap + gap

    x0 = x0.apply(lambda x: x if x > 0 else 0)
    y0 = y0.apply(lambda y: y if y > 0 else 0)
    df['x'], df['y'] = x0, y0

    # usa colunas w, h para salvar x1, y1 temporariamente
    df['w'], df['h'] = x1, y1
    # verifica se ultrapassa limites do tamanho da janela
    x1 = df.apply(lambda row: row['w'] if row['w'] < row['width'] else row['width'], axis=1)
    y1 = df.apply(lambda row: row['h'] if row['h'] < row['height'] else row['height'], axis=1)
    df['w'], df['h'] = x1 - x0, y1 - y0

    df = df.astype({'x': int, 'y': int, 'w': int, 'h': int, 'width': int, 'height': int})

    return df


def mostrar_imagens(df:pd.DataFrame, start=0, end=3, lst_img=None, wait=True):

    for idx, row in df.iterrows():
        if row['note'] != 'Escalado':
            print(f"{row['note']}: {row['width']}x{row['height']}")

    if lst_img is None:
        lst_img = [nr for nr in range(len(df))]
    else:
        start, end = lst_img[0], lst_img[-1] + 1
    df1 = df[(df['seq'] >= start) & (df['seq'] <= end)]
    for seq in range(start, end):
        df2 = df1[df1['seq'] == seq]
        if df2.empty:
            continue
        if seq not in lst_img:
            continue
        image_file = df2.iloc[0, 0]  # 'image_file'
        img = ImageBinary(path=image_file)
        df_box = df2.loc[:, 'x':'h']
        nome = f'{seq}: {len(df_box)}, {img.width} x {img.height}'
        print(nome)
        ShowMe(image=img.image, df=df_box, win_name=nome, wait_close=wait)  # , scale=0, thickness=2

# normalizar tamanho da imagen max e min
def normalizar_imagem(df:pd.DataFrame, min_size=512, max_size=1024, num_img=0):

    conta = 1
    for indice, row in df.iterrows():
        if num_img and conta > num_img:
            break
        if row['note'] is not np.nan and 'normalizado' in row['note'].casefold():
            continue
        path = row['new_image']
        if not os.path.isfile(path):
            path = 'images/' + path
            if not os.path.isfile(path):
                continue
        conta += 1
        img = cv.imread(path)
        (w, h) = (img.shape[1], img.shape[0])
        new_width, new_height = w, h
        dim = False
        if w < min_size:
            new_width = min_size
            new_height = h = int((h * (min_size / w)))
            w = new_width
            dim = True
        if h > max_size:
            new_height = max_size
            new_width = int((w * (max_size / h)))
            dim = True
        if h < min_size:
            new_height = min_size
            new_width = int((w * (min_size / h)))
            dim = True
        if dim:
            new_size = (new_width, new_height)
            img = cv.resize(img, new_size, interpolation=cv.INTER_AREA)
            cv.imwrite(path, img, [cv.IMWRITE_JPEG_QUALITY, 100])
            df.loc[indice, 'note'] = f"Normalizado {new_width}x{new_height} ***"
        else:
            df.loc[indice, 'note'] = f"Normalizado {new_width}x{new_height}"
    return df


def avaliar_imagem(item=9, normalizar=False, show=True, param_size=False):

    jpeg_file = 'data/csv/img_forms_jpeg.csv'
    df: pd.DataFrame
    df = pd.read_csv(jpeg_file, sep='|')
    df = df.sort_values(by='new_image', ignore_index=True)
    df = df[df['new_image'].str.contains(f'imagem_{item:03}')]
    img_path = df['new_image'].values[0]
    if not(os.path.isfile(img_path)):
        img_path = 'images/' + img_path

    # Normalizar tamanho
    if normalizar:
        normalizar_imagem(df)
    else:
        # img_path = f"images_jpg/imagem_{item:03}.jpg"
        img_path = f"images/imagem_{item:03}.jpg"

    # Normaliza rects da classe Checkbox conforme dados da Entry
    img = ImageBinary(path=img_path)
    w, h = img.width, img.height
    size = [(w * 0.02, h * 0.02), (w * 0.5, h * 0.15)]  # Entry
    entry = InputDataBox(image=img, size_rect=size, area_perc=20, max_search=1)
    df_box, err1 = entry.get_boxes()
    std, med = entry.desvio, entry.media

    if param_size:
        size = [(w * 0.4, h * 0.02), (w * 0.97, h * 0.15)]  # Entry
        entry = InputDataBox(image=img, size_rect=size, area_perc=20, max_search=1)
        df_bx2, err2 = entry.get_boxes()
    else:
        df_bx2, err2 = pd.DataFrame(), ''

    size = [(w * 0.01, h * 0.02), (h * 0.15, h * 0.15)]   # Checkbox
    check = InputDataBox(image=img, size_rect=size, area_perc=20, max_search=1, desvio=std, media=med)
    df_chk, err3 = check.get_boxes()

    if err1 and err2 and err3:
        return None, None, None

    # junta duas classes de box
    df_box = pd.concat([df_box, df_bx2, df_chk], ignore_index=True)
    df_box = df_box.drop_duplicates(ignore_index=True)

    # Detectar RadioButton
    # radio1 = InputRadio(im_bin=img, metodo='HOUGH', minRadius=1, maxRadius=20)
    # radio2 = InputRadio(im_bin=img, metodo='CONTOURS', minRadius=1, maxRadius=20, thresh=188, maxVal=255)

    # classifica df pelo formato da caixa (quadrada ou retangulo)
    df_box = box_format_classify(df_box)

    # escalar retangulos
    cols = ['image_file', 'seq', 'idx', 'label', 'x', 'y', 'w', 'h', 'width', 'height', 'note']
    df_scl = pd.DataFrame(columns=cols)
    df_scl = pd.concat([df_scl, df_box], ignore_index=True)
    df_scl['image_file'] = img_path
    df_scl['seq'] = item
    df_scl['idx'] = [el for el in range(len(df_scl))]
    df_scl['width'] = w
    df_scl['height'] = h
    df_box = box_scaling(df_scl, scale=1.25)
    df_box = df_box[df_box.columns[3:8]]
    df_box = df_box.reindex(columns=['x', 'y', 'w', 'h', 'label'])

    win_name = f'Rects={len(df_box)} {img.width}x{img.height}'
    print(f'Rects={len(df_box)}  ({img.width}x{img.height})')
    if show:
        ShowMe(image=img.image, df=df_box, win_name=win_name, wait_close=show)
    return img, df_box, win_name


def comparar_avaliar(item=9, param_size=False):
    print(f'Avaliando imagem: {item}')
    # img1, df1, name1 = avaliar_imagem(item=item, normalizar=False, show=False)
    img1, df1, name1 = avaliar_imagem(item=item, normalizar=False, param_size=True, show=False)
    # img2, df2, name2 = avaliar_imagem(item=item, normalizar=False, param_size=True, show=False)

    df1 = remove_retangulos_sobrepostos(df1)
    name1 = f"{name1} --> Rects {len(df1)}"

    show = ShowMe(image=img1.image, df=df1, win_name=name1, show=False)
    show.show_me(img1.image, df=df1, win_name=name1, wait=True)
    # show.show_me(img2.image, df=df2, win_name=name2, wait=True)


def distancia_2_rects(rect1, rect2):
    """Calcula a menor distância euclidiana entre 2 cantos de 2 rects (x, y, w, h)"""
    x1, y1, w1, h1 = rect1
    x2, y2, w2, h2 = rect2
    distancia1 = int(((x2 - x1)**2 + (y2 - y1)**2) ** 0.5)
    x1, y1 = x1 + w1, y1 + h1
    x2, y2 = x2 + w2, y2 + h2
    distancia2 = int(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
    return min([distancia1, distancia2])

def remove_retangulos_sobrepostos(df: pd.DataFrame):
    """Remove sobreposicao de retangulos em uma distância minima"""
    idx_pares = []
    # Itera sobre todos os pontos, comparando cada ponto com todos os outros à frente
    row1: pd.Series
    for row1 in df.itertuples():
        rect1 = [row1.x, row1.y, row1.w, row1.h]
        df2 = df.loc[row1.Index + 1:, :]
        for row2 in df2.itertuples():
            rect2 = [row2.x, row2.y, row2.w, row2.h]
            dist = distancia_2_rects(rect1, rect2)
            # Se distância for menor que altura do menor h, de mesmo label
            if dist < min([rect1[3], rect2[3]]) and row1.label == row2.label:
                rect_mean = obter_retangulo_medio(rect1, rect2)
                new_row = rect_mean + [row1.label]
                idx_pares.append([row1.Index, row2.Index, new_row])
    for i, j, row in idx_pares:
        df.loc[i, :] = row
    idx_pares = list(set([j for i, j, r in idx_pares]))
    for j in idx_pares:
        if j in df.index:
            df.drop([j], axis=0, inplace=True)

    return df

def obter_retangulo_medio(rect1, rect2):
    """Calcula o retângulo médio a partir de uma lista de retângulos."""
    (x, y, w, h) = rect1
    (x1, y1, w1, h1) = rect2
    pts1 = (x, y, x + w, y + h)
    pts2 = (x1, y1, x1 + w1, y1 + h1)
    retangulos = np.array([pts1, pts2])
    media_coords = np.mean(retangulos, axis=0)
    x, y, x1, y1 = tuple(int(c) for c in media_coords)
    w, h = x1 - x, y1 - y
    rect_mean = [x, y, w, h]
    return rect_mean


def main_tools(functions=None, num_img=0, lst_img=None, show=False):

    # functions = ['salvar_jpg', 'normalizar', 'sequenciar', 'anotar', 'escalar', 'mostrar']
    if functions is None:
        functions = ['mostrar']

    links_file = 'data/csv/img_forms_links.csv'
    jpeg_file = 'data/csv/img_forms_jpeg.csv'
    label_file = 'data/csv/img_forms_labels.csv'
    scale_file = 'data/csv/img_forms_scale.csv'

    df_jpg: pd.DataFrame
    df_lab: pd.DataFrame
    df_scl: pd.DataFrame
    cols = ['image_file', 'seq', 'idx', 'label', 'x', 'y', 'w', 'h', 'width', 'height', 'note']
    col_types = {'image_file': str, 'seq': int, 'idx': int, 'label': str, 'x': int, 'y': int,
                 'w': int, 'h': int, 'width': int, 'height': int, 'note': str}

    df_frm = pd.read_csv(links_file, sep='|')
    df_frm = df_frm.sort_values(by='new_image', ignore_index=True)
    df_frm = df_frm.fillna('')
    if os.path.isfile(jpeg_file):
        df_jpg = pd.read_csv(jpeg_file, sep='|')
    else:
        df_jpg = df_frm.copy()
    df_jpg = df_jpg.fillna('')

    if 'salvar_jpeg' in functions:
        df_jpg = salvar_imagens_formato_jpeg('images', df_jpg)
        df_jpg = df_jpg.sort_values(by='new_image', ignore_index=True)
        df_jpg.to_csv(jpeg_file, index=False, sep='|', columns=df_jpg.columns, header=True)
    if 'normalizar' in functions:
        normalizar_imagem(df_jpg, num_img=num_img)
        df_jpg.to_csv(jpeg_file, index=False, sep='|', columns=df_jpg.columns, header=True)
    if 'sequenciar' in functions:
        df_jpg = sequenciar_imagens('images', df_jpg)
        df_jpg.to_csv(jpeg_file, index=False, sep='|', columns=df_jpg.columns, header=True)
    if 'anotar' in functions:
        if os.path.exists(label_file):
            df_lab = pd.read_csv(label_file, sep='|')
        else:
            df_lab = pd.DataFrame(columns=cols).astype(col_types)
        df_lab = df_lab.sort_values(by=['image_file', 'seq', 'idx'], ignore_index=True)
        df_lab = image_annotation(df_jpg, df_lab, num_img=num_img, lst_img=lst_img, show=show)
        df_lab = df_lab.sort_values(by=['image_file', 'seq', 'idx']).astype(col_types)
        df_lab.to_csv(label_file, index=False, sep='|', columns=df_lab.columns, header=True)
    if 'escalar' in functions:
        if os.path.exists(scale_file):
            df_scl = pd.read_csv(scale_file, sep='|')
        else:
            df_lab = pd.read_csv(label_file, sep='|')
            df_scl = df_lab.copy()
        df_scl = box_scaling(df_scl)
        df_scl.astype(col_types).to_csv(scale_file, sep='|', index=False, columns=df_scl.columns, header=True)
    if 'mostrar' in functions:
        df_scl = pd.read_csv(scale_file, sep='|')
        mostrar_imagens(df_scl, start=0, end=15, lst_img=lst_img)

    if 'exportar_YOLO' in functions:
        if os.path.exists(scale_file):
            df_scl = pd.read_csv(scale_file, sep='|')
            exportar_df_scaled_to_yolo(df_scl)

    if 'exportar_json' in functions:
        if os.path.exists(scale_file):
            df_scl = pd.read_csv(scale_file, sep='|')
            exportar_json()


def exportar_df_scaled_to_yolo(df_scl):
    # <dir_yolo>
    #   label.txt
    #   image_001.txt
    #       id    x_center  y_center   weight     height
    #       2     0.830368  0.829574   0.080276   0.045113

    labels_yolo = 'data/YOLO/labels_Rects_YOLO.csv'
    cols = ['file_txt', 'seq', 'id', 'x_center', 'y_center', 'width', 'height']
    dir_yolo = 'data/YOLO'
    # dict id: labels  (classes)
    dic_lab = {0: 'Entry', 1: 'Combobox', 2: 'Checkbox', 3: 'Radiobutton',
              4: 'Listbox', 5: 'Button', 6: 'Shape', 7: 'Picture', 8: 'Deformado'}
    dic_id = {'Entry': 0, 'Combobox': 1, 'Checkbox': 2, 'Radiobutton': 3,
              'Listbox': 4, 'Button': 5, 'Shape': 6, 'Picture': 7, 'Deformado': 8}

    # grava arquivo txt de labels (classes)
    with open(dir_yolo + '/' + 'labels.txt', 'w', newline='') as txt_file:
        txt_file.write('\n'.join(dic_id.keys()))

    # prepara df yolo para exportação
    df_yolo = pd.DataFrame(columns=cols)
    df_yolo['file_txt'] = df_scl['image_file'].apply(lambda x: os.path.basename(x)[:-3] + 'txt')
    df_yolo['seq'] = df_scl['seq']
    df_yolo['id'] = df_scl['label'].apply(lambda x: dic_id[x])
    # calcular as coords da label de cada linha e normalizar
    df_yolo['x_center'] = (df_scl['x'] + (df_scl['w'] / 2)) / df_scl['width']
    df_yolo['y_center'] = (df_scl['y'] + (df_scl['h'] / 2)) / df_scl['height']
    df_yolo['width'] = df_scl['w'] / df_scl['width']
    df_yolo['height'] = df_scl['h'] / df_scl['height']

    df_yolo.to_csv(labels_yolo, sep='|', index=False, columns=df_yolo.columns, header=True)

    # salvar os files txt de imagem com coords label
    df: pd.DataFrame
    cols = ['id', 'x_center', 'y_center', 'width', 'height']
    seqs = set(df_yolo['seq'].values)
    for seq in seqs:
        df = df_yolo.loc[df_yolo['seq'] == seq, :]
        file_txt = dir_yolo + '/' + df.iloc[0, 0]
        df.to_csv(file_txt, index=False, sep=' ', columns=cols, header=False, float_format='%.6f')


def exportar_json(item=None, offset=None):
    # <dir_yolo>
    #   label.txt
    #   image_001.txt
    #       id    x_center  y_center   weight     height
    #       2     0.830368  0.829574   0.080276   0.045113

    scale_file = 'data/csv/img_forms_scale.csv'
    if os.path.exists(scale_file):
        df_scl = pd.read_csv(scale_file, sep='|')
    else:
        print(f'Arquivo não encontrado: {scale_file}')
        return

    labels_json = 'E:/PyCharm/LabelmeProject/dataset/json/labels_forms_json.csv'
    dir_images = 'images'
    dir_json = 'E:/PyCharm/LabelmeProject/dataset/json'
    # dict id: labels  (classes)
    dic_lab = {0: 'Entry', 1: 'Combobox', 2: 'Checkbox', 3: 'Radiobutton',
               4: 'Listbox', 5: 'Button', 6: 'Shape', 7: 'Picture', 8: 'Table',
               9: 'Iconbutton', 10: 'LineEntry', 11: 'Deformado'}
    dic_id = {'Entry': 0, 'Combobox': 1, 'Checkbox': 2, 'Radiobutton': 3,
              'Listbox': 4, 'Button': 5, 'Shape': 6, 'Picture': 7, 'Table': 8,
              'Iconbutton': 9, 'LineEntry': 10, 'Deformado': 11}

    # parametros do formato json
    label = 'Entry'
    group_id  = 0
    flags = {}
    shape_type = 'rectangle'  # 'circle'
    description = ''
    mask = 'null'
    x0, y0, x1, y1 = 0, 0, 0, 0
    points = f"[['{x0}', '{y0}'] ,['{x1}', '{y1}']]"
    dic_shape = {"label": label, "points": points, "group_id": group_id,
                 "description": description,  "shape_type": shape_type,
                 "flags": flags, "mask": mask}
    image_path = "images/imagem_000.jpg"
    img_base64 = "(funcao para gerar arquivo base_64)"
    width, height  = 1024, 768
    list_shapes = [dic_shape, dic_shape]
    # estrutura do arquivo json
    dic_json = {"version": "5.9.1", "flags": flags,
                "shapes": list_shapes,
                "imagePath": image_path,
                "imageData": img_base64,
                "imageHeight": height,
                "imageWidth": width
                }

    # grava arquivo txt de labels (classes)
    # with open(dir_json + '/' + 'labels.txt', 'w', newline='') as txt_file:
    #     txt_file.write('\n'.join(dic_id.keys()))

    if item is not None:
        df_scl = df_scl[df_scl['seq']==item]

    if df_scl.empty:
        print(f'Dataframe está vazio! Item: {item}. Verifique arquivo {scale_file}')
        return

    # prepara um df json
    cols = ['image_file', 'file_json', 'seq', 'group_id', 'label', 'x0', 'y0', 'x1', 'y1', 'width', 'height']
    df_json = pd.DataFrame(columns=cols)
    df_json['file_json'] = df_scl['image_file'].apply(lambda x: os.path.basename(x)[:-3] + 'json')
    df_json['image_file'] = df_scl['image_file']
    df_json['seq'] = df_scl['seq']
    df_json['group_id'] = df_scl['label'].apply(lambda x: dic_id[x])
    df_json['label'] = df_scl['label']

    # atualiza as coords da label padrão json (Labelme)
    df_json['x0'] = df_scl['x']
    df_json['y0'] = df_scl['y']
    df_json['x1'] = df_scl['x'] + df_scl['w']
    df_json['y1'] = df_scl['y'] + df_scl['h']
    df_json['width'] = df_scl['width']
    df_json['height'] = df_scl['height']

    if offset is not None:
        # offset = [left, top, right, bottom]
        df_json['x0'] = df_json['x0'].apply(lambda x: int(x) - offset[0])
        df_json['y0'] = df_json['y0'].apply(lambda y: int(y) - offset[1])
        df_json['x1'] = df_json['x1'].apply(lambda x: int(x) + offset[2])
        df_json['y1'] = df_json['y1'].apply(lambda y: int(y) + offset[3])

    # grava df json como csv
    if item is None:
        df_json.to_csv(labels_json, sep='|', index=False, columns=df_json.columns, header=True)

    # cria os arquivos formato json para cada imagem na pasta dir_json
    row: pd.Series
    df: pd.DataFrame
    if item is None:
        seqs = set(df_json['seq'].values)
    else:
        seqs = [item]
    # processa todas imagens, cria um arquivo json por imagem
    for seq in seqs:
        df = df_json.loc[df_json['seq'] == seq, :]
        row = df.iloc[0, :]
        width, height = int(row.width), int(row.height)
        file_json = dir_json + '/' + row.file_json
        image_path = row.image_file  # dir_images + '/' +

        print(f'Processando imagem {file_json}...')
        img_base64, erro = encode_image_to_base64(image_path)
        if erro:
            df_json['note'] = erro
            continue

        list_shapes = []
        # acumula as labels da imagem em uma lista
        for row in df.itertuples():
            # index = row.Index
            label = row.label
            group_id = row.group_id
            x0, y0, x1, y1 = row.x0, row.y0, row.x1, row.y1
            points = [[x0, y0] ,[x1, y1]]
            dic_shape = {"label": label, "points": points, "group_id": group_id,
                         "description": '', "shape_type": 'rectangle',
                         "flags": {}, "mask": None}
            list_shapes.append(dic_shape)

        # monta dicionario de conteudo do arquivo json para gravação
        dic_json = {"version": '5.9.1', "flags": {},
                    "shapes": list_shapes,
                    "imagePath": image_path,
                    "imageData": img_base64,
                    "imageHeight": height,
                    "imageWidth": width
                    }
        # Abre o arquivo em modo de escrita ('w') e usa json.dump() para salvar os dados
        with open(file_json, 'w', encoding='utf-8') as obj_file:
            json.dump(dic_json, obj_file, ensure_ascii=False, indent=4)

def decode_base64_image_to_byte(base64_image_string, image_path):
    """" Args:
         base64_image_string: the Base64 encoded string of an image
         image_path: nome do arquivo imagem de saida
         Return: save the image file in byte format
    """
    # Convert the Base64 string to bytes
    # The .encode('ascii') or .encode('utf-8') is necessary if your base64 string
    # is a Python string type and b64decode expects a bytes-like object.
    image_bytes = base64.b64decode(base64_image_string.encode('utf-8'))
    # Now 'image_bytes' contains the raw binary data of the image.
    # You can then save these bytes to a file, for example:
    try:
        with open(image_path, "wb") as f:
            f.write(image_bytes)
        return True
    except Exception as e:
        print(f"Erro: {e}")
        return False

def encode_image_to_base64(image_path):
    """Encodes an image file to a Base64 string.
        Args:
        image_path (str): The path to the image file.
        Returns:
        str: The Base64 encoded string of the image, or None if an error occurs.
    """
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            base64_encoded_data = base64.b64encode(image_data)
            base64_string = base64_encoded_data.decode('utf-8')
            return base64_string, ''
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, 'FileNotFoundError'
    except Exception as e:
        print(f"An error occurred: {e}")
        return None, str(e)


# Remove de Retângulos duplicados
def remove_duplicate_shapes(lista_shapes, **kwargs):

    lst_shapes = []
    rects = []
    for shape in lista_shapes:
        if shape['shape_type'] == 'rectangle':
            box = np.array(shape['points'], dtype=np.float32).ravel()
            x, y = box[[0, 2]].min(), box[[1, 3]].min()
            w, h = abs(box[2] - box[0]), abs(box[3] - box[1])
            rects.append([x, y, w, h])

    # normal 0.2, limite de similaridade para fusão de retângulos
    eps = kwargs.get('eps', 0.1)
    # número mínimo de retângulos semelhantes necessários para formar um grupo
    group_thr = kwargs.get('groupThreshold', 1)
    # Duplica o retângulo para garantir retorno de pelo menos um groupRectangles
    rects2: np.ndarray = np.concatenate((rects, rects), axis=0)
    # Agrupamento de retângulos próximos, redução para um retângulo medial
    gr_rects, pesos = cv.groupRectangles(rects2, groupThreshold=group_thr, eps=eps)

    for shape in lista_shapes:
        if shape['shape_type'] == 'rectangle':
            box = np.array(shape['points'], dtype=np.float32).ravel()
            x, y = box[[0, 2]].min(), box[[1, 3]].min()
            w, h = abs(box[2] - box[0]), abs(box[3] - box[1])
            rect = [x, y, w, h]
            if rect in gr_rects:
                lst_shapes.append(shape)
        else:
            lst_shapes.append(shape)

    return lst_shapes

if __name__ == '__main__':

    # main_tools(['mostrar'], lst_img=lista, show=True)
    # None  # [2 , 6, 8, 10, 128]

    # main_tools(['anotar'], show=False, lst_img=lista)

    # main_tools(['escalar'], show=False, lst_img=lista)

    # main_tools(['exportar_YOLO'], show=False)

    # main_tools(['exportar_json'], show=False)

    lista = []  # [745]
    for item in lista:
        exportar_json(item=item, offset=[0, 0, 1, 1])

    # comparar_avaliar(item=i, param_size=True)

    print('Finalizado tarefa')
