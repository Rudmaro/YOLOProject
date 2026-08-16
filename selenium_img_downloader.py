# import requests
import os

import pandas as pd
import base64
import logging
import socket
# import urllib3
# from itertools import cycle
from urllib.error import HTTPError, URLError
from urllib.request import urlopen, Request


# Function to check if result is base64
def check_for_base64(source):
    possible_header = source.split(',')[0]
    if possible_header.startswith('data') and ';base64' in possible_header:
        image_type = possible_header.replace('data:image/', '').replace(';base64', '')
        return image_type
    return False


def extensao_valida(base_name, sep='.'):
    extensao = base_name.casefold()
    while sep in extensao:
        texto, extensao = extensao.split(sep, 1)
    ext = {'jpg': '.jpg', 'jpeg': '.jpg', 'png': '.png', 'avif': '.avif', 'webp': '.webp',
           'tiff': '.tiff', 'tif': '.tif', 'jxr': '.jxr', 'jpe': '.jpg', 'gif': '.gif', 'bmp': '.bmp'}
    if extensao in ext:
        return ext[extensao]
    return ''

def download_url(url_img, headers, output_path, termo):
    # proxies = {'http': 'http://10.10.1.10:3128', 'https': 'https://10.10.1.11:1080'}
    # proxy_pool = cycle(proxies)
    request = Request(url_img, headers=headers)
    try:
        # proxy = next(proxy_pool)
        # proxies={"http": proxy, "https": proxy}
        # response = requests.get(url_img, headers=headers, verify=False)
        with urlopen(request, timeout=10) as response:
            if response.status == 200:
                with open(output_path, 'wb') as f:
                    while True:
                        chunk = response.read(1024)
                        if not chunk:
                            break  # End of file
                        f.write(chunk)
                print(f"Sucesso: {output_path}")
                return 1,  200, 'Sucesso'
            else:
                print(f'Falha! status code: {response.status_code}')
                logging.error(f"{termo}|{response.status_code}")
                status = response.status_code
                reason = response.reason
    except HTTPError as error:
        print('Erro HTTP:', error.status, error.reason, url_img)
        logging.error(f"{termo}|{error.status}, {error.reason}")
        status = error.status
        reason = error.reason
    except URLError as error:
        print('Erro URL:', error.reason)
        logging.error(f"{termo}|{error.reason}")
        status = 0
        reason = error.reason
    except TimeoutError:
        print("Request timed out")
        logging.error(f"{termo}|Request timed out")
        status = 10
        reason = 'timeout'
    except Exception as error:
        print('Falha:', error)
        logging.error(f"{termo}|{error}")
        status = 0
        reason = error
    return 0, status, reason

# Function to perform the image downloads
def download_img(df: pd.DataFrame, diretorio, sep='-', conta=1):
    headers = {
        "Accept-Language": 'es-ES,es;q=0.9,pt-BR;q=0.8,pt;q=0.7,en-US;q=0.5,en;q=0.3',
        "Accept": 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/jpeg,image/jpg,'
                  'image/png,image/bmp,image/webp,*/*;q=0.8',
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0"                      
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/132.0.6827.61 Mobile Safari/537.36"}
    caminho_raiz = diretorio + '/imagem' + sep
    logging.basicConfig(filename='output/log/logging.log', filemode='w',
                        format='%(name)s | %(levelname)s | %(message)s')

    for row in df.itertuples(index=True, name='Pandas'):

        row: pd.Series = row
        if row.status in [200, 64]:
            continue
        image_file  = ''
        url_imagem = row.src_link
        termo_query = row.search_terms

        # Check if src content is base64 instead of URL link
        is_b64 = check_for_base64(url_imagem)
        # If it is base64, then retrieve the image with the following if statement
        if is_b64:
            content = base64.b64decode(url_imagem.split(';base64')[1])
            try:
                # O tipo de imagem pode ser extraído do cabeçalho: 'image/jpeg'
                header, base64_str = url_imagem.split(";", 1)
                extensao = extensao_valida(header, sep='/')
                destino = f"{caminho_raiz}{conta:03}{extensao}"
                # with open('output/images/' + row.search_terms + '.png'.format(image_format), 'wb') as f:
                #     f.write(content)
                with open(destino, 'wb') as f:
                    f.write(content)
                valor, status, reason = 1, 64, 'Sucesso'
                conta = conta + valor
                image_file = destino
                print(f"File downloaded successfully to {destino}")
            except Exception as e:
                print('Erro baixar base64:', e)
                logging.error(termo_query + ' | ' + str(e))
                valor, status, reason = 0, 640, str(e)
        # Else, if it is a direct URL, then perform urlretrieve to download the image
        else:
            # destino = 'output/images/' + row.search_terms  + extensao  # '.png'
            # caminho, cabeca = urllib.request.urlretrieve(row.src_link)
            # caminho = os.rename(caminho, destino)
            extensao = extensao_valida(url_imagem[-9:])
            destino = f"{caminho_raiz}{conta:03}{extensao}"
            # Baixa a imagem
            valor, status, reason = download_url(url_imagem, headers, destino, termo_query)
            conta = conta + valor
            if valor > 0:
                image_file = destino
        df.loc[row.Index, 'status'] = status
        df.loc[row.Index, 'reason'] = reason
        df.loc[row.Index, 'image_file'] = image_file

    return df


if __name__ == '__main__':
    # List of img src for the image downloads
    dir_imagens = 'images'
    input_file = 'output/links/img_forms_links.csv'
    output_file = 'output/links/img_forms_links.csv'
    df_img: pd.DataFrame
    df_img = pd.read_csv(input_file, sep='|')
    if 'status' not in df_img.columns:
        df_img['status'] = 99
    if 'reason' not in df_img.columns:
        df_img['reason'] = ''
    if 'image_file' not in df_img.columns:
        df_img['image_file'] = ''
    socket.setdefaulttimeout(15)

    lst_arq = os.listdir(dir_imagens)
    lst_arq.sort()
    if len(lst_arq) > 0:
        sep = '-' if '-' in lst_arq[-1] else '_'
        base1, base2 = lst_arq[-1].split(sep)
        conta_ini = int(base2[0:3]) + 1
    else:
        conta_ini = 1

    # Calling the image download function
    df_img = download_img(df_img, diretorio=dir_imagens, sep='-', conta=conta_ini)

    df_img.to_csv(output_file, index=False, sep='|', columns=df_img.columns)
