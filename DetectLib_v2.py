"""
    Conjunto de classe e funções para detectar objetos em imagens.
    Utiliza opencv para captura de contornos
"""
import cv2 as cv
import numpy as np
import pandas as pd
from numpy import ndarray
import pytesseract
from fuzzywuzzy import fuzz
from pandas.core.window.doc import kwargs_scipy
from torch.distributed.tensor import empty


# from numpy.core.multiarray import ndarray


# Classe para definir geometria retangular
class Rect:
    def __init__(self, **kwargs):
        """
        Mantém dados da geometria retangular
        Args:
            rect (list): [x, y, width, height]
            x or left (int): coordenada x
            y or top (int): coordenada y
            w or width (int): largura do caixa
            h or height (int): altura da caixa
            pts2 (tuple): 2 cantos ((x0, y0), (x3, y3))
            pts (tuple): 4 pontos, ((x0, y0), (x1, y1), (x2, y2), (x3, y3))
        :return:
            arr_points (ndarray): array of points
        """
        x = kwargs.get('x', 0)
        y = kwargs.get('y', 0)
        h = kwargs.get('h', 0)
        w = kwargs.get('w', 0)
        if not w:
            x = kwargs.get('left', x)
            y = kwargs.get('top', y)
            h = kwargs.get('height', h)
            w = kwargs.get('width', w)
            if 'rect' in kwargs:
                x, y, w, h = kwargs.get('rect', [0, 0, 0, 0])
            elif 'pts2' in kwargs:
                (x, y), (x3, y3) = kwargs.get('pts2', ((0, 0), (0, 0)))
                w, h = x3 - x, y3 - y
            elif 'pts' in kwargs:
                (x, y), _, _, (x3, y3) = kwargs.get('pts', ((0, 0), 0, 0, (0, 0)))
                w, h = x3 - x, y3 - y
        x2 = x + w
        y2 = y + h
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.x2 = x2
        self.y2 = y2
        self.pt0 = (x, y)
        self.pt1 = (x2, y)
        self.pt2 = (x2, y2)
        self.pt3 = (x, y2)
        self.rect = [x, y, w, h]
        self.area = w * h
        self.perimeter = (2 * w) + (2 * h)
        self.points = [self.pt0, self.pt1, self.pt2, self.pt3]
        self.arr_points = self.get_corners_rect(self.points)

    @staticmethod
    def get_corners_rect(pts: list) -> ndarray:
        arr_pts: ndarray = np.array(pts, dtype=np.int32)
        return arr_pts

    @staticmethod
    def get_corners_from_rects(rect_list: ndarray) -> ndarray:
        rects = [[(x, y), (x + w, y), (x + w, y + h), (x, y + h)] for x, y, w, h in rect_list]
        rects = np.array(sum(rects, []), np.int32)
        return rects

    def point_in_rect(self, pt: tuple or list, rect: list = None) -> bool:
        x, y = pt
        x0, y0, w0, h0 = self.rect if rect is None else rect
        x1, y1 = x0 + w0, y0 + h0
        cond = x0 <= x <= x1 and y0 <= y <= y1
        return cond


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


class LabelsId:
    entry = 0
    combobox = 1
    checkbox = 2
    radiobutton = 3
    listabox = 4
    button = 5
    shape = 6
    picture = 7
    table = 8
    iconbutton = 9
    underline = 10
    switch = 11
    deformado = 12


class LabelsCor:
    cor_id = {
        0: BGR.red,
        1: BGR.vermelho,
        2: BGR.verde,
        3: BGR.azul,
        4: BGR.rosa,
        5: BGR.violeta,
        6: BGR.amarelo,
        7: BGR.ciano,
        8: BGR.magenta,
        9: BGR.roxo,
        10: BGR.neon,
        11: BGR.marrom,
        12: BGR.preto
    }


class ShowMe:
    df: pd.DataFrame
    df_sector: pd.DataFrame
    row: pd.Series
    image: ndarray
    scale_width: int
    scale_height: int
    draw_rect: bool
    put_text: bool
    draw_line: bool
    draw_marker: bool
    rects: list
    win_name: str
    scale: float
    thickness: int
    num_boxes: int
    box2label: bool
    wait_close: bool
    show: bool
    dic_entries: dict = {'Entry': 1, 'Checkbox': 2, 'Listbox': 3, 'Deformado': 6}

    def __init__(self, **kwargs):
        """
        Visualizar imagens com objetos desenhados
        Args:
            :type image: ndarray image
            :type path: str path image
            :type img_read:  canal de leitura da imagem: GRAY, BGR, RBG
            :type df: dataframe de dados textos e caixas
            :type rects: lista de retangulos tipo Rect
            :type row: apenas uma linha do dataframe
            :type df_sector: df de setores
            :type draw_rect: True/False para desenha retangulos
            :type put_text: True/False para desenhar textos
            :type draw_line: True/False para desenhar linha liga texto - caixa
            :type marker: True/False para desenhar marcas
            :type win_name: nome da janela
            :type scale: fator de escala = 0,5 reduzir,2 = aumentar
            :type thickness: espessura da linha
            :type num_boxes: int 1, 2 ou 3 caixa(s) de entrada(s)
            :type box2label: True para desenhar caixa com label, False somente caixa
            :type wait_close: True/False aguardar clique teclado para fechar janela
            :type show: True/False para abrir janela agora
        """
        self.path = kwargs.get('path', '')
        img_read = kwargs.get('img_read', cv.IMREAD_COLOR_BGR)
        if self.path:
            self.image: ndarray = cv.imread(self.path, img_read)
        else:
            self.image: ndarray = kwargs.get('image', np.zeros((0, 0), dtype=np.uint8))
        if len(self.image) == 0:
            return
        self.box2label = kwargs.get('box2label', False)
        self.scale = kwargs.get('scale', 0)
        self.put_text = kwargs.get('put_text', True)
        self.draw_line = kwargs.get('draw_line', True)
        self.draw_rect = kwargs.get('draw_rect', True)
        self.draw_marker = kwargs.get('marker', True)
        self.wait_close = kwargs.get('wait_close', True)
        self.show = kwargs.get('show', True)
        self.thickness = kwargs.get('thickness', 1)
        self.df = kwargs.get('df', pd.DataFrame())
        self.num_boxes = kwargs.get('num_boxes', 1)
        self.df_sector = kwargs.get('df_sector', pd.DataFrame())
        self.row = kwargs.get('row', pd.Series())
        self.rects = kwargs.get('rects', [])
        if len(self.image.shape) == 2:  # GrayScale
            self.image = cv.cvtColor(self.image, cv.COLOR_GRAY2BGR)
        self.scale_height = int(self.image.shape[0] * self.scale)
        self.scale_width = int(self.image.shape[1] * self.scale)
        self.cores = {0: BGR.red, 1: BGR.vermelho, 2: BGR.verde, 3: BGR.azul,
                      4: BGR.rosa, 5: BGR.violeta, 6: BGR.amarelo, 7: BGR.ciano,
                      8: BGR.magenta, 9: BGR.preto, 10: BGR.roxo, 11: BGR.marrom}
        win_name = f"{self.path}, Rects: {len(self.df) or len(self.rects)}"
        self.win_name = kwargs.get('win_name', win_name)
        if self.show:
            self.show_me(self.image, self.df, win_name=self.win_name, wait=self.wait_close)

    def show_me(self, image, df, path='', win_name='', wait=True):

        if path:
            image: ndarray = cv.imread(path, cv.IMREAD_COLOR_BGR)
        else:
            image: ndarray = self.image if image is None else image
        if len(image) == 0:
            return
        df = self.df.copy() if df is None else df.copy()
        df.reset_index(inplace=True)
        if not self.row.empty:
            df = self.row.to_frame().T
        elif df.empty:
            df = pd.DataFrame(data=self.rects, columns=['class', 'x', 'y', 'w', 'h'])
        for idf, row in self.df_sector.iterrows():
            if 'sector' in row.index:
                classe, x1, y1, x3, y3 = row.loc['sector':'y3']
                rects = [[classe, x1, y1, x3 - x1, y3 - y1]]
                for i, rect in enumerate(rects):
                    classe = rect[0]
                    cor = self.cores[classe]
                    cv.rectangle(image, rect[1:], cor, self.thickness)

        for idx, row in df.iterrows():
            rects = []
            if self.draw_rect:
                if 'class0' in row.index:
                    rects = [(row.loc[f'class{i}':f'h{i}']).tolist() for i in range(self.num_boxes)]
                elif 'sector' in row.index:
                    classe, x1, y1, x3, y3 = row.loc['sector':'y3']
                    rects = [[classe, x1, y1, x3 - x1, y3 - y1]]
                elif 'class' in row.index:
                    rects = [(row.loc['class':'h']).tolist()]
                elif 'x' in row.index:
                    rects = (row.loc['x':'h']).tolist()
                    entry: int = 0
                    if 'label' in row.index:
                        entry = self.dic_entries[row.label]
                    rects = [[entry] + rects]  # inseri classe
                for i, rect in enumerate(rects):
                    classe = int(rect[0])
                    cor = self.cores[classe]
                    cv.rectangle(image, rec=rect[-4:], color=cor, thickness=self.thickness)
                    # cv.putText(image,str(classe), (rect[1], rect[2]), cv.FONT_HERSHEY_COMPLEX, 0.6, cor)

            if self.put_text and 'text' in row.index:
                rect = rects[0][1:]
                classe = rects[0][0]
                cor = self.cores[classe]
                rect_txt = (row.loc['left':'height']).tolist()
                cv.rectangle(image, rect_txt, cor, self.thickness)
                pto = (rect_txt[0], rect_txt[1])
                cv.putText(image, str(classe), pto, cv.FONT_HERSHEY_COMPLEX, 0.6, cor)
                # cv.drawMarker(img1, (x0, y1), cor, cv.MARKER_CROSS, 5)
                if self.draw_line:
                    pt2 = (rect[0], rect[1])
                    cv.line(image, pto, pt2, cor, 1)
                if self.draw_marker:
                    pto = (rect[0], rect[1] + (rect[3] // 2))
                    cv.drawMarker(image, pto, BGR.magenta, cv.MARKER_CROSS, 5)

        win_name = win_name or f'{self.win_name} Rects:{len(self.df)} ({image.shape})'
        if self.scale:
            cv.namedWindow(win_name, cv.WINDOW_NORMAL)
        cv.imshow(win_name, image)
        if self.scale:
            cv.resizeWindow(win_name, self.scale_width, self.scale_height)
        if wait:
            cv.waitKey()
            cv.destroyAllWindows()


# Classe para encapsular o tesseract e char_whitelist
class OCRTesseract:
    output = ''
    config: str = 'ASCII'
    language: str = ''
    tesseract_cmd: str = ''
    abc = r'abcçdefghijklmnopqrstuvwxyz'
    xyz = r'ABCÇDEFGHIJKLMNOPQRSTUVWXYZ'
    acento = r'áéíóúÁÉÍÓÚãõüâêîôû'
    dig = r'0123456789'
    char = r'" !\"#$%&\'()*+,-./:;<=>?@[\]^_`{|}~"'  # tabular não incluído
    ANY = r'--oem 3 --psm 6'
    Num = r'-c tessedit_char_whitelist=0123456789 --oem 3 --psm 6'
    Float = r'-c tessedit_char_whitelist=0123456789., --oem 3 --psm 6'
    Upp = r'-c tessedit_char_whitelist=' + xyz + ' --oem 3 --psm 6'
    Low = r'-c tessedit_char_whitelist=' + abc + ' --oem 3 --psm 6'
    Alfa = r'-c tessedit_char_whitelist=' + abc + xyz + ' --oem 3 --psm 6'
    AlfaNum = r'-c tessedit_char_whitelist=0123456789' + abc + xyz + ' --oem 3 --psm 6'
    # ASCII = r'--oem 3 --psm 6 -c tessedit_char_whitelist=' + dig + abc + xyz + acento + char
    ASCII = r'--oem 3 --psm 6'

    def __init__(self, **kwargs):
        """
        Inicializa a classe, opcionalmente configurando o caminho do executável do Tesseract.
        Args:
            image (ndarray): A imagem.
            language (str): O idioma para o OCR (ex: 'por' para português).
            config (str): configuração do tesseract
            output_type: tipo da variável de retorno do tesseract
        """
        tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'
        if 'tesseract_cmd' in kwargs:
            tesseract_cmd = kwargs.get('tesseract_cmd', tesseract_cmd)
        elif 'path' in kwargs:
            tesseract_cmd = kwargs.get('path', tesseract_cmd)
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.tesseract_cmd = tesseract_cmd

        idioma = kwargs.get('lang', '')
        # Configuração de melhor atuação
        config = kwargs.get('config', self.ASCII)
        output = kwargs.get('output', pytesseract.Output.DATAFRAME)

        self.config = config
        self.output = output
        self.language = idioma

    def detect_text(self, image: ndarray, **kwargs) -> pd.DataFrame:
        """
        Extrai texto de uma imagem usando pytesseract.
        Args:
            image (ndarray): A imagem.
            lang (str): O idioma para o OCR (ex: 'por' para português).
            config (str): configuração do tesseract
            output_type: tipo da variável de retorno do tesseract
        Returns:
            str: O texto extraído da imagem.
        """
        data = pd.DataFrame()
        idioma = self.language
        idioma = kwargs.get('lang', idioma)
        try:
            # Get detailed information including bounding boxes for words
            if idioma:
                data = pytesseract.image_to_data(image, config=self.config, lang=idioma,
                                                 output_type=pytesseract.Output.DATAFRAME)
            else:
                data = pytesseract.image_to_data(image, config=self.config,
                                                 output_type=pytesseract.Output.DATAFRAME)
            return data
        except pytesseract.TesseractNotFoundError:
            print("\nErro: O executável do Tesseract não foi encontrado. Verifique a instalação.")
            return data
        except FileNotFoundError:
            print("\nErro: O arquivo de imagem não foi encontrado.")
            return data
        except Exception as e:
            print(f"\nOcorreu um erro: {e}")
            return data


# Classe rect hierarchy
class RectHierarchy:
    inside = 0  # interno
    outside = 1  # externo
    without = 2  # sem hierarquia


# Classe retângulo rect:[x, y, width, height]
class Rectangle:
    def __init__(self, **kwargs):
        if 'rect' in kwargs:
            x, y, width, height = kwargs.get('rect', [0, 0, 0, 0])
        else:
            x = kwargs.get('x', 0)
            y = kwargs.get('y', 0)
            width = kwargs.get('w' if 'w' in kwargs else 'width', 0)
            height = kwargs.get('h' if 'h' in kwargs else 'height', 0)
        x2 = x + width
        y2 = y + height
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.x2 = x2
        self.y2 = y2
        self.pt1 = (x, y)
        self.pt2 = (x2, y)
        self.pt3 = (x2, y2)
        self.pt4 = (x, y2)
        self.hierarchy = 0

    def hierarchy(self, rect=None, **kwargs) -> int:
        """
            Verifica relação/hierarquia do rect de entrada com rect instanciado pela classe
            Args:
                :type rect: object Rectangle [x, y, width, height]
                kwargs: x, y, w, h, width, height
            Returns:
                int: Nível de hierarquia RectHierarchy
        """
        if rect is not None:
            x, y, width, height = rect
        elif 'rect' in kwargs:
            x, y, width, height = kwargs.get('rect', [0, 0, 0, 0])
        else:
            x = kwargs.get('x', 0)
            y = kwargs.get('y', 0)
            width = kwargs.get('w' if 'w' in kwargs else 'width', 0)
            height = kwargs.get('h' if 'h' in kwargs else 'height', 0)
        x2, y2 = x + width, y + height
        if (self.x >= x and self.y >= y and
                self.x2 <= x2 and self.y2 <= y2):
            self.hierarchy = RectHierarchy.inside
        elif (x >= self.x and y >= self.y and
              x2 <= self.x2 and y2 <= self.y2):
            self.hierarchy = RectHierarchy.outside
        else:
            self.hierarchy = RectHierarchy.without

        return self.hierarchy


# Técnica de limiarização para leitura dos contornos, preparar imagem binaria
class ImageBinary:
    image: ndarray
    img_gray: ndarray
    img_bin: ndarray
    img_blur: ndarray
    thresh: ndarray
    path: str
    GAUSSIAN = cv.ADAPTIVE_THRESH_GAUSSIAN_C
    MEAN = cv.ADAPTIVE_THRESH_MEAN_C
    THRESH = 'THRESH_BINARY'
    THRESH_INV = 'THRESH_BINARY_INV'
    THRESH_OTSU = 'THRESH_OTSU'
    BLUR = 'GaussianBlur'
    CANNY = 'Canny'
    RGB = 'RGB'
    height: int = 0
    width: int = 0

    def __init__(self, image: ndarray = None, **kwargs):
        """
            Tratamento da imagem com técnicas de limiarização para leitura de contornos.
            Tratamento imagem binaria adaptiveThreshold e saída canny
            Args:
                :type image: ndarray object image
                :type path: path str image
                :type method: sequence of str or single str method [GAUSSIAN, MEAN, THRESH]
            Returns:
            image: object image
        """
        # BGR2GRAY: contornos são encontrados com mais precisão em imagens de canal único 8 bits
        self.path = kwargs.get('path', '')
        if self.path:
            self.image = cv.imread(self.path, cv.IMREAD_COLOR_BGR)
            if self.image is None:
                print(f'Caminho de imagem invalido: {self.path}')
                return
            self.img_gray = cv.imread(self.path, cv.IMREAD_GRAYSCALE)
        else:
            self.image = image
            if len(image.shape) > 2:
                self.img_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            else:
                self.img_gray = image.copy()
        self.height = self.img_gray.shape[0]
        self.width = self.img_gray.shape[1]
        # self.get_image_bin(method=method)

    def get_image_thresh(self, method: str = THRESH, **kwargs) -> np.ndarray:
        """
            Args:
                'ksize' tuple: gaussianblur parametro (7, 7)
                'sigmaX' int: gaussianblur parametro default 0
            Returns:
                image: object image thresh ndarray
        """
        thresh = kwargs.get('thresh', 127)
        maxVal = kwargs.get('maxVal', 255)
        # Usar este thresh para distinguir textos, polylines pequenas (< 90 por exemplo)
        if method == self.THRESH:
            ret, img_thresh = cv.threshold(self.img_gray, thresh, maxVal, cv.THRESH_BINARY)
        elif method == self.THRESH_INV:
            ret, img_thresh = cv.threshold(self.img_gray, thresh, maxVal, cv.THRESH_BINARY_INV)
        elif method == self.THRESH_OTSU:
            thresh = kwargs.get('thresh', 0)
            ret, img_thresh = cv.threshold(self.img_gray, thresh, maxVal, cv.THRESH_OTSU)
        elif method in [self.THRESH_INV, self.THRESH_OTSU]:
            thresh = kwargs.get('thresh', 0)
            ret, img_thresh = cv.threshold(self.img_gray, thresh, maxVal, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        # Desfoque gaussiano para reduzir ruído e ajudar na detecção de contornos.
        elif method == self.BLUR:
            ksize = kwargs.get('ksize', (7, 7))
            sigmaX = kwargs.get('sigmaX', 0)
            img_thresh = cv.GaussianBlur(self.img_gray, ksize=ksize, sigmaX=sigmaX)
            self.img_blur = img_thresh
        else:
            img_thresh = self.img_gray.copy()

        self.thresh = img_thresh
        return self.thresh

    def get_image_bin(self, method, **kwargs) -> np.ndarray:
        """
            Args:
                'ksize' tuple: gaussianblur parametro (7, 7)
                'sigmaX' int: gaussianblur parametro default 0
            Returns:
                image: object image binary ndarray
        """
        if type(method) is not list:
            method = [method]
        if self.BLUR in method:
            ksize = kwargs.get('ksize', (7, 7))
            sigmaX = kwargs.get('sigmaX', 0)
            img_gray = cv.GaussianBlur(self.img_gray, ksize=ksize, sigmaX=sigmaX)
            method = method[1:]
            if len(method) == 0:
                return img_gray
        if self.CANNY in method:
            return cv.Canny(self.img_gray, threshold1=100, threshold2=200)
        if self.RGB in method:
            return self.image
        if self.THRESH in method:
            return self.get_image_thresh(self.THRESH)
        if self.THRESH_INV in method:
            return self.get_image_thresh(self.THRESH_INV)

        adap_method = method[0]
        img_adp = cv.adaptiveThreshold(self.img_gray, maxValue=200,
                                       adaptiveMethod=adap_method,
                                       thresholdType=cv.THRESH_BINARY,
                                       blockSize=5, C=2)
        img_bin = img_adp.copy()
        # Aplicar sequencia abaixo para obter melhores resultados
        # Canny + dilated + eroded
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv.erode(img_bin, kernel, iterations=1)
        # eroded = cv.Canny(eroded, threshold1=10, threshold2=20)
        canny = cv.Canny(img_bin, threshold1=10, threshold2=20)
        canny[eroded > 0] = 255
        if 'dilate' in method:
            imag_blur = cv.GaussianBlur(self.img_gray, (7, 7), sigmaX=0)
            dilate = cv.dilate(imag_blur, kernel, iterations=1)
            dilate = cv.Canny(dilate, threshold1=10, threshold2=20)
            canny[dilate > 0] = 255
        self.img_bin = canny.copy()
        return self.img_bin


# Detectar possíveis caixas de entrada de dado
class InputDataBox:
    # variáveis de retorno e gerais
    df_box: pd.DataFrame
    rects_found: ndarray
    rects: ndarray
    image: ImageBinary

    size_min: tuple
    size_max: tuple
    size_rect: tuple
    start_perc: int
    end_perc: int
    desvio: tuple
    media: tuple
    # % de ocupação de rects internos para excluir o rect externo
    area_perc: int
    epsilon: float
    vertices: int
    all_search = ['gauss', 'mean', 'canny', 'thresh', 'thresh_inv']
    default_search = ['gauss', 'mean']
    search_types: list = default_search

    def __init__(self, image: ImageBinary, **kwargs):
        """
        Captura caixa de entrada de dados da imagem de entrada
        Args:
            :type image: object image
            'size_min' (tuple): min size rect (width: int, height: int)
            'size_max' (tuple): max size rect (width: int, height: int)
            'size_rect'(list): list of min/max size rect [(w_min, h_min), (w_max, h_max)]
            'start_perc' (int): % to remove first items of the list input data
            'end_perc' (int): % to remove ends items of the list input data
            'area_perc' (int): int % of internal rect occupations
            'epsilon' (float): fator de multiplicação comprimento do arco (approxPolyDP) find_rects
            'vertices' (int): número máximo de vértices aceitáveis como polyline
            'max_search' (bool): True=all_search, False=default_search
            'search_types' (list): tipos de imagens binarias para processar pesquisa
            'desvio' (tuple): desvio padrao w e h (w_desvio, h_desvio)
            'media' (tuple): media altura e comprimento tuple(w_media, h_media)
        Returns:
        df_box dataframe: df de retangulos [[x, y, w, h], ...]
        """
        # guarda variável imagem original
        self.image = image
        width, height = image.width, image.height

        # limitação de tamanho dos retângulos a serem localizado na imagem
        # padrão inicial: minimo de 10% e máximo de 95% da largura e altura
        size_min, size_max = (width * 0.01, height * 0.01), (width * 0.95, height * 0.95)
        size_rect = kwargs.get('size_rect', [])
        if len(size_rect):
            size_min, size_max = size_rect
        else:
            size_min = kwargs.get('size_min', size_min)
            size_max = kwargs.get('size_max', size_max)
            size_rect = (size_min, size_max)
        # percentual de redução da lista para remover itens iniciais e finais
        start_perc = kwargs.get('start_perc', 0)
        end_perc = kwargs.get('end_perc', 0)
        # % de area ocupada dos retangulos internos para excluir o rect externo
        area_perc = kwargs.get('area_perc', 60)
        # propriedades usadas na captura de retangulos (Rects)
        self.epsilon = kwargs.get('epsilon', 0.01)
        self.vertices = kwargs.get('vertices', 10)
        # img_types = ['gauss', 'mean', 'canny', 'thresh', 'thresh_inv']
        self.search_types = kwargs.get('search_types', self.default_search)
        if kwargs.get('max_search', False):
            self.search_types = self.all_search
        self.size_min = size_min
        self.size_max = size_max
        self.size_rect = size_rect
        self.start_perc = start_perc
        self.end_perc = end_perc
        self.area_perc = area_perc
        self.desvio = kwargs.get('desvio', (0, 0))
        self.media = kwargs.get('media', (0, 0))

    # inicia o processo e retorna dataframe das caixas localizadoras
    def get_boxes(self) -> tuple[pd.DataFrame, str]:
        self.df_box = pd.DataFrame()
        self.rects_found = self.find_rects()  # 906/500   1212/616
        if len(self.rects_found) > 0:
            # ShowMe(self.image.img_gray, rects=self.rects_found)
            self.rects = self.drop_duplicate_rects(rects=self.rects_found)  # 208/280   192/276
            # ShowMe(self.image.img_gray, rects=self.rects)
            self.rects = self.normalize_rects(rects=self.rects)  # 56/56   40/48
            # dataframe retangulos para retorno
            self.df_box = pd.DataFrame(self.rects, columns=['x', 'y', 'w', 'h'])
        return self.df_box, ''

    # Detectar forma geométrica retangular utilizando opencv
    def find_rects(self, **kwargs) -> ndarray:
        """
            Contornos capturados (opencv) de 2 imagens binarias adaptive method gaussian_c e mean_c
            Args:
                'search_types' (list): tipos imagens ['gauss', 'mean', 'canny', 'thresh', 'thresh_inv']
                'epsilon' (float): fator de multiplicação comprimento do arco (approxPolyDP)
                'vertices' (int): número máximo de vértices aceitáveis como polyline
        """
        epsilon = kwargs.get('epsilon', self.epsilon)
        vertices = kwargs.get('vertices', self.vertices)
        img_types = kwargs.get('search_types', self.search_types)
        # Normalizar comprimento aceitável de um contorno / retângulo válido
        # Limita comprimento do contorno para evitar dimensões extremas, e reduzir ruídos
        comp_min = int(self.size_min[0] * 2 + self.size_min[1] * 2)
        comp_max = int(self.size_max[0] * 2 + self.size_max[1] * 2)

        # guarda lista de formas retangulares válidas
        rects = []
        contours = tuple(np.zeros(shape=(2, 1, 2)))
        contornos: tuple
        dic_types = {'gauss': cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                     'mean': cv.ADAPTIVE_THRESH_MEAN_C,
                     'canny': 'Canny',
                     'thresh': 'THRESH_BINARY',
                     'thresh_inv': 'THRESH_BINARY_INV',
                     'rgb': 'RGB'}
        # Retorna os contornos
        for key in img_types:
            img: ndarray = self.image.get_image_bin(method=dic_types[key])
            contornos, _ = cv.findContours(img, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
            if contours[0].any():
                contours = contours + contornos
            else:
                contours = contornos

        # Para cada contorno, aproxime sua forma poligonal approxPolyDP()
        for contour in contours:  # hole_contours:
            comp = int(cv.arcLength(contour, True))
            # Dispensar curvaturas (contornos) de comprimento não normalizados
            if comp < comp_min or comp > comp_max:
                continue
            # Forma poligonal para reduzir o número de vértice (Ramer-Douglas-Peucker)
            # epsilon → tolerância: distancia máxima permitida entre curva original e versão aproximada
            # menor alta precisão: preserva + detalhes, mantém vértices significativos
            # maior baixa precisão: + agressiva, redução significativa de vértices
            approx = cv.approxPolyDP(curve=contour, epsilon=epsilon * comp, closed=True)
            # aceita Polylines com até 10 vertices
            if len(approx) < vertices + 1:
                # Converte aproximação para retângulo
                rect = cv.boundingRect(approx)
                rects.append(rect)

        return np.array(rects, np.int32)

    # Remove de Retângulos duplicados
    def drop_duplicate_rects(self, rects: ndarray, **kwargs) -> ndarray:
        """
        Agrupa e remove retangulos de tamanhos duplicados
        Args:
            rects (ndarray): lista de retangulos (x, y, w, h)
            'eps' (float): limite de similaridade para a fusão de retângulos, normal 0.2
            'groupThreshold' (ibt): número mínimo de retângulos semelhantes necessários
                                    para formar um grupo
        """
        eps = kwargs.get('eps', 0.1)
        group_thr = kwargs.get('groupThreshold', 1)
        # Duplica o retângulo para garantir retorno de pelo menos um groupRectangles
        # rects = 1100 --> 107
        rects2: ndarray = np.concatenate((rects, rects), axis=0)
        # Agrupamento de retângulos próximos, redução para um retângulo medial
        gr_rects, pesos = cv.groupRectangles(rects2, groupThreshold=group_thr, eps=eps)
        return np.array(gr_rects)

    # Normalização dos Retângulos
    def normalize_rects(self, rects: ndarray) -> ndarray:
        """
            Adequar lista para manter somente retangulos dentro dos limites de tamanho
        """
        # Retângulos com dimensão atípicas são removidos, distantes do desvio padrão
        # Média (μ) e o desvio padrão (σ)
        if self.desvio[1] == 0 or self.media == 0:
            h_desvio, h_media = self.find_std(rects=rects, start_perc=5, end_perc=5, sort_col=3)
            w_desvio, w_media = self.find_std(rects=rects, start_perc=75, end_perc=0, sort_col=2)
        else:
            w_desvio, h_desvio = self.desvio
            w_media, h_media = self.media
        self.desvio = (w_desvio, h_desvio)
        self.media = (w_media, h_media)

        # Filtra por tamanho do retângulo dentro dos limites min e max
        rects_dentro = rects[rects[:, 3] > self.size_min[1]]
        rects_dentro = rects_dentro[rects_dentro[:, 3] < self.size_max[1]]
        rects_dentro = rects_dentro[rects_dentro[:, 2] > self.size_min[0]]
        rects_dentro = rects_dentro[rects_dentro[:, 2] < self.size_max[0]]

        # Remove tamanhos atípicos que estão acima de um desvio padrão
        rects_dentro = rects_dentro[abs(rects_dentro[:, 3] - h_media) / (h_desvio or 1) <= 1]
        rects_dentro = rects_dentro[(rects_dentro[:, 2] - w_media) / (w_desvio or 1) <= 1]
        # Mantém comprimentos maiores que a metade da altura media
        rects_dentro = rects_dentro[rects_dentro[:, 2] > (h_media / 2)]

        # Filtra retângulo dentro de retângulo
        sorted_indices = (-rects_dentro[:, 2]).argsort()  # ordem por width Desc
        rects_dentro = rects_dentro[sorted_indices]
        index_del = []
        for idx, rect in enumerate(rects_dentro):
            x, y, w, h = rect
            rects = np.delete(rects_dentro, idx, axis=0)
            rects = rects[rects[:, 0] >= x]
            rects = rects[rects[:, 1] >= y]
            rects = rects[(rects[:, 0] + rects[:, 2]) <= (x + w)]
            rects = rects[(rects[:, 1] + rects[:, 3]) <= (y + h)]
            # possui retangulos internos, então remove este rect
            if len(rects) > 0:  # se pelo menos existe 1 retângulo interno?
                ww = np.sum(rects[:, 2])
                hh = np.sum(rects[:, 3])
                # remove se área somada dos retangulos internos for menos da metade do rect externo
                if (w * h) / (ww * hh) < (100 / self.area_perc):
                    index_del.append(idx)

        # remove retângulo externo se haver dois de retangulos internos
        if len(index_del) > 0:
            rects_dentro = np.delete(rects_dentro, index_del, axis=0)

        self.rects = rects_dentro
        return rects_dentro

    # Calcula desvio padrão de uma lista de valores
    def find_std(self, rects: ndarray, start_perc: int = None, end_perc: int = None, sort_col: int = None):

        if start_perc is None:
            start_perc = self.start_perc
        if end_perc is None:
            end_perc = self.end_perc
        # Aplica percentual de redução na lista para remover itens iniciais e finais
        # rects2 = rects[rects[:, sort_col].argsort()]   # mantém dimensão matriz original
        # retorna array de uma coluna (achatada)
        rects2 = np.sort(rects[:, sort_col]) if sort_col else np.sort(rects)

        # remove itens iniciais da lista
        ini = int(start_perc * 0.01 * len(rects))
        # remove itens finais da lista
        fim = len(rects) - int(end_perc * 0.01 * len(rects))
        rects2 = rects2[ini:fim]
        desvio = np.std(rects2)  # desvio padrão
        desvio = np.round(desvio, 2) if desvio else 0
        media = np.round(np.mean(rects2), 2)
        return desvio, media


# Localiza textos na imagem via pytesseract e retorna dataframe com caixas de textos
class DetectTesseract:
    # variáveis de retorno e gerais
    df_txt: pd.DataFrame
    dados: pd.DataFrame
    labels: list
    tesseract_cmd: str
    image: ImageBinary

    def __init__(self, image: ImageBinary, **kwargs):
        """
            Localiza textos na imagem via pytesseract
            Args:
                :type image: object image
                labels (list): list of fields names
                tesseract_cmd (str): path tesseract command exe
            Returns:
                df_box dataframe: df com caixa de texto [['text' 'left' 'top'...]...]
                Filtrado com base nos nomes listados no argumento labels
        """
        # guarda variável imagem original
        self.image = image
        self.labels = kwargs.get('labels', [])
        self.labels = [item.casefold() for item in self.labels]
        tesseract_cmd = r'c:\Program Files\Tesseract-OCR\tesseract.exe'
        tesseract_cmd = kwargs.get('tesseract_cmd', tesseract_cmd)
        self.tesseract_cmd = tesseract_cmd
        self.dados = self.get_all_texts()

    def get_all_texts(self) -> pd.DataFrame:
        """ Retorna df de todos os textos encontrados """
        data: pd.DataFrame
        tesseract = OCRTesseract(path=self.tesseract_cmd)
        # Get detailed information including bounding boxes for words
        data = tesseract.detect_text(self.image.img_gray)
        # Exclude blocks and empty text
        data = data[pd.notna(data['text'])]
        data = data[data['conf'] != -1]
        data = data[data['text'].str.strip() != '']
        data['text'] = data['text'].str.casefold()
        data['text'] = data['text'].str.replace('—', '')  # Saida-—————
        self.dados = data

        return data

    def get_filters_texts(self, data: pd.DataFrame = None, labels: list = None) -> pd.DataFrame:
        """ Retorna df filtrado com base na lista de nomes em nomes_lst """
        if labels is not None:
            self.labels = [item.casefold() for item in labels]
        if data is None:
            data = self.dados
        new_data: pd.DataFrame = pd.DataFrame()
        for nome in self.labels:
            namelist = nome.split()
            for name in namelist:
                df_ord = pd.DataFrame(dtype=object)
                prob = 95
                while df_ord.empty and prob > 74:
                    df_ord = data[data['text'].map(lambda x: fuzz.ratio(x, name) > prob)]
                    if df_ord.empty:
                        prob = prob - 5
                    else:
                        df_ord.loc[:, 'text'] = name
                if not df_ord.empty:
                    new_data = pd.concat([new_data, df_ord], axis=0)
        new_data.drop_duplicates(inplace=True, ignore_index=True)
        new_data.sort_index(inplace=True, ignore_index=True)

        new_data = self.connect_texts(data=new_data)

        return new_data

    def agrupar_textos(self, index: int, df_1: pd.DataFrame, df_2: pd.DataFrame) -> list:
        """ Agrupamento de textos que estão próximos """
        grupo = []
        for idx, row in df_1.iterrows():
            # largura de um carácter * 3 (espaços em branco)
            gap_x = (row.loc['width'] // len(row.loc['text'])) * 3
            gap_y = row.loc['height'] // 2
            x = row.loc['left'] + row.loc['width']
            y = row.loc['top']
            df_3 = df_2[(df_2['left'] > x) & (df_2['left'] < x + gap_x) &
                        (df_2['top'] > y - gap_y) & (df_2['top'] < y + gap_y)]
            if len(df_3) == 1:
                indice = int(df_3.index[0])
                grupo.append((index, idx, indice))
            elif not df_3.empty:
                # ponto caixa de texto base (esquerda)
                pto_txt = np.array([x, y])
                # pontos caixa de texto a direita próximos do campo base
                pts_x, pts_y = df_3['left'], df_3['top']
                pontos = np.array(list(zip(pts_x, pts_y)), np.int32)
                dist_euc = distancia_euclidiana_pontos(pto_txt, pontos)
                indice = int(df_3.index[np.argmin(dist_euc)])
                grupo.append((index, idx, indice))
        return grupo

    def connect_texts(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Retorna novo df de textos reagrupando aqueles que estão próximos,
            separados por espaços vazios de 3 chars """
        new_data = data.copy()
        new_data['link'] = 0
        pares_lista = []

        for texto in self.labels:
            namelist = texto.split()
            if len(namelist) == 1:
                continue
            df_nomes = [new_data[new_data['text'] == txt] for txt in namelist]
            for i, df in enumerate(df_nomes[:-1]):  # não pega último df
                pares = self.agrupar_textos(i, df, df_nomes[i + 1])
                pares_lista.append(pares)
                for k, idx1, idx2 in pares:
                    new_data.loc[idx1, 'link'] = idx2

        # atualiza dataframe de saida
        df2 = new_data[new_data['link'] > 0]
        for idx, row in df2.iterrows():
            indice = row.loc['link']
            texto = row.loc['text']
            comp = row.loc['width']
            while indice > 0:
                texto = texto + ' ' + new_data.loc[indice, 'text']
                comp = comp + new_data.loc[indice, 'width']
                # Replace rows in new_data where indices match those in df2
                new_data.loc[idx, 'text'] = texto
                new_data.loc[idx, 'width'] = comp
                new_indice = new_data.loc[indice, 'link']
                new_data.loc[indice, 'link'] = -1
                indice = new_indice
        # mantém as linhas de textos unicos ou agrupados
        new_data = new_data[new_data['link'] >= 0]

        return new_data


# Conecta textos com caixas de entrada de dado da imagem de entrada
class Text2BoxLinks:
    # variáveis de retorno e gerais
    df3cxa: pd.DataFrame
    labels: list
    df_box: pd.DataFrame
    df_txt: pd.DataFrame
    cols_txt: list
    cols_cxa: list

    def __init__(self, df_txt: pd.DataFrame, df_box: pd.DataFrame, **kwargs):
        """
            Localiza textos na imagem e conecta com caixas de entrada de dado
            Args:
                :type image: object image
                :type df_txt: dataframe texts/labels
                :type df_box: dataframe caixas de entrada de dado
                'labels' (list): list of fields names
            Returns:
                df3cxa (dataframe): df caixa de texto + 3 caixas de entradas
        """
        self.labels = kwargs.get('labels', [])
        self.labels = [item.casefold() for item in self.labels]
        self.df_box = df_box
        self.df_txt = df_txt
        self.create_df3box()

    # prepara dataframe de saida df3cxa
    def create_df3box(self):
        self.cols_txt = ['text', 'left', 'top', 'width', 'height']
        self.cols_cxa = ['class0', 'x0', 'y0', 'w0', 'h0', 'dist0',
                         'class1', 'x1', 'y1', 'w1', 'h1', 'dist1',
                         'class2', 'x2', 'y2', 'w2', 'h2', 'dist2']
        if self.df_txt.empty:
            self.df3cxa = pd.DataFrame(columns=self.cols_cxa)
        else:
            self.df3cxa = pd.DataFrame(self.df_txt[self.cols_txt], columns=self.cols_txt + self.cols_cxa)
        self.df3cxa.fillna(0, inplace=True)
        self.df3cxa[self.cols_cxa] = self.df3cxa[self.cols_cxa].astype(int)

    def get_boxes2text_links(self) -> pd.DataFrame:
        """ Conectar caixa de texto (rótulo/label) com 3 caixas de entrada dado """

        # lista rects caixa de entrada, tempo = 0.150 ms
        # self.df_box.values

        for index, row in self.df_txt.iterrows():
            box_txt = Rect(rect=row.loc['left':'height'])

            # filtro lista rects box caixa de entrada = 0.185 (mais lento)
            # zona = BoxProximity()
            # box_cxa_lst = zona.get_boxes_in_region(box_txt, df_box)

            # retorna 3 caixas mais proxima ao texto,
            rects = self.get_3boxes2text(box_txt, self.df_box.values)
            rects = sum(rects, [])  # achata matriz, uma dimensão
            self.df3cxa.loc[index, 'class0':'dist2'] = rects

        return self.df3cxa

    # retorna lista rects das 3 caixas de entrada mais próxima do texto referencia
    def get_3boxes2text(self, box_txt: Rect, boxes_cxa: ndarray = None) -> list:
        """
            localiza 3 caixas de entrada mais próximos ao texto,
            comparativo de distancias entre 4 cantos da caixa textos com 4 cantos da caixa entrada
        Args:
            :type box_txt: rect da caixa texto (x, y, w, h).
            :type boxes_cxa: ndarray de n rect da caixa de entrada [(x1, y1, w1, h1), ...]
        Returns:
            rects (ndarray): lista text+class+rect+dist das 3 caixas,
             ordenadas pela distancia de cada caixa ao texto
             [[left, top, width, height, class0, x0, y0, w0, h0, dist0], ... ([..., dist2]]
        """
        referencias: ndarray = box_txt.arr_points
        if boxes_cxa is None:
            boxes_cxa = self.df_box.values
        pontos = box_txt.get_corners_from_rects(rect_list=boxes_cxa)

        dists_quadradas: np.ndarray = np.array([], np.int32)
        for referencia in referencias:
            differences = pontos - referencia
            soma: np.ndarray = np.sum(differences ** 2, axis=1)
            dists_quadradas = np.append(dists_quadradas, soma)
        # Encontra o índice da menor distância
        # flat argmin index = np.argmin(dists_quadradas) or 0  # index menor dist.
        # Lista indices ordenadas das menores distâncias.
        flat_sort_index: np.ndarray = np.argsort(dists_quadradas)
        matrix_shape = [4, len(boxes_cxa), 4]  # 4 cantos
        # rects: np.ndarray = np.array([left, top, width, height], np.int32)
        rects, dists = [], []
        for flat_index in flat_sort_index:
            # Convert the flattened index to 2D coordinates
            ind_txt, ind_cxa, ind_pts = np.unravel_index(flat_index, matrix_shape)
            # print(ind_txt, ind_cxa, ind_pts)
            # pt_texto = referencias[ind_txt]
            # lista rect text = left, top, width, height +
            # [xi, yi, wi, hi] retângulo das 3 caixas de entradas mais próximas
            rect = boxes_cxa[ind_cxa].tolist()
            if rect not in rects:
                rects.append(rect)
                dists.append(int(dists_quadradas[flat_index] ** 0.5))
            if len(rects) > 2:  # limite de 3 caixas de entradas
                break
        # adiciona distancias aos retangulos
        [rect.append(dists[i]) for i, rect in enumerate(rects)]
        # completa tamanho do array com zeros, se for inferior a 3
        [rects.append([0, 0, 0, 0, 0]) for i in range(3 - len(rects), 0, -1)]
        # inclui 0 para class0..2 em cada rect da lista rects
        [rect.insert(0, 0) for i, rect in enumerate(rects)]
        # arr_rects: ndarray = np.array(rects, np.int32)

        return rects


# Inserir nova coluna antes ou depois do nome da coluna referência
def df_new_column(df, new_col, col_ref, after=False, value=0):
    if new_col not in df.columns:
        ref_col_index = df.columns.get_loc(col_ref) + after
        df.insert(loc=ref_col_index, column=new_col, value=0)
        df.fillna(value, inplace=True)
    return df


# Classifica a proximidade entre duas caixas: texto x caixa de entrada
#  FAR
#          OUT                   w_setor= 2*offset_x + w_text/2
# +--------------Section---------------------+
# |  [ LT   ]  [     TOP      ]  [  RT   ]   |
# |  [ LEFT ]  [ CENTER [INT] ]  [ RIGHT ]   | h_setor= 2*offset_y + 1*h_text
# |  [ BT   ]  [    BOTTOM    ]  [  BT   ]   |
# +------------------------------------------+
class BoxProximity:
    # Define rotulo de proximidade do texto com a caixa de entrada
    INT = 0  # "INT"  # texto 100% no interior da caixa
    CENTER = 1  # "CENTER" caixa e texto na região central - intersecção
    RIGHT = 2  # "RIGHT"  # caixa na direita
    BOTTOM = 3  # "BOTTOM"  # caixa abaixo do texto
    LEFT = 4  # "LEFT"  # caixa na esquerda
    TOP = 5  # "TOP" # caixa acima do texto
    LT = 6  # "LT"   # Caixa esquerda superior
    RT = 7  # "RT"   # Caixa direita superior
    LB = 8  # "LB"   # Caixa esquerda inferior
    RB = 9  # "RB"   # Caixa direita inferior
    OUT = 10  # "OUT" # caixa fora, acima de 2 * largura e 2 * altura caixa
    FAR = 11  # "FAR" # Caixa muito fora, distancia sem interesse

    # Valor do rótulo de proximidade
    proximity = None
    side = None
    lado = None
    # distancia (horizontal, vertical) entre caixas texto e entrada
    offset: tuple
    shape: tuple
    dic_sector: dict = {}  # dicionario de regiões
    df_sector: pd.DataFrame
    rect_txt: Rect
    rect_cxa: Rect

    def __init__(self, rect_txt: Rect, **kwargs):
        """
            Classifica a proximidade entre duas caixas: texto x caixa de entrada
            args:
                :type rect_txt: Rect da caixa texto (rótulos)
                :type rect_cxa: Rect caixa entrada usado para calculo dos setores
            return:
                df3box dataframe: df com 3 caixa de de textos classificadas
        """
        # propriedades dos limites de tamanho da caixa
        self.rect_cxa = kwargs.get('rect_cxa', None)
        self.rect_txt = rect_txt
        # distancia max de proximidade entre bordas caixa texto e caixa entrada
        # um caractere (largura 1 digito) = altura texto = h = 1 space
        self.get_text2box_proximity(self.rect_txt, self.rect_cxa)
        # self.dic_sector = self.sector_map(self.rect_txt, self.rect_cxa)
        # self.W_MEAN, self.H_MEAN = self.media(rect_txt, rect_cxa)

    def media(self, rect_txt: Rect, rect_box: Rect) -> tuple:
        if rect_txt is None:
            rect_txt = self.rect_txt
        if rect_box is None:
            rect_box = self.rect_cxa
        x, y, w, h = rect_txt.rect
        x1, y1, w1, h1 = rect_box.rect
        w_mean, h_mean = (w1 + w) // 2, (h1 + h // 2)
        return w_mean, h_mean

    def get_central_sector(self, box_txt: Rect, box_cxa: Rect) -> list:

        xt, yt, wt, ht = box_txt.rect
        xc, yc, wc, hc = box_cxa.rect
        hs = ht if ht > hc else hc  # a maior altura da caixa ou texto
        offset_y = (hs // 6) or 1  # altura max da borda superior, dist topo texto x sector
        offset_x = 2 * hs  # [ caixa ]<-x->[LABEL] dist max de vão horizontal
        hs = hs + (2 * offset_y)
        ws = (wt // 2) + offset_x  # metade da largura texto/label + 1 offset
        # ajusta altura da borda superior em relação ao canto esq/sup texto
        offset_y = (hs - ht) // 2
        # tuple (x, y) distancia max proximidade borda entre caixas texto e entrada
        self.offset = (offset_x, offset_y)
        self.shape = (hs, ws)
        # pontos do setor central
        x1, y1 = xt - offset_x, yt - offset_y  # canto superior esquerdo setor
        return [x1, y1, ws, hs]

    # Região rotulada (4 pontos do retângulo) envolta da caixa texto
    def sector_map(self, rect_txt: Rect, rect_cxa: Rect) -> dict:
        # recalcula altura e largura do setor
        x1, y1, ws, hs = self.get_central_sector(rect_txt, rect_cxa)
        # x1, y1 canto superior esquerdo setor
        x2, y2 = x1 + ws, y1  # canto superior direito
        x3, y3 = x2, y2 + hs  # canto inferior direito
        x4, y4 = x1, y1 + hs  # canto inferior esquerdo

        # |  LT   |  TOP   |   RT   |
        # |  LEFT |  C [ label ]    | RIGHT
        # |  LB   | BOTTOM |   RB   |
        x, y, w, h = rect_txt.rect
        pt_int = [(x, y), (x + w, y + h)]
        dic_set = {self.INT: pt_int,
                   self.CENTER: [(x1, y1), (x4, y4)],
                   self.BOTTOM: [(x4, y4), (x3, y3 + hs)],
                   self.RIGHT: [(x2, y2), (x3 + ws, y3)],
                   self.TOP: [(x1, y1 - hs), (x2, y2)],
                   self.LEFT: [(x1 - ws, y1), (x4, y4)],
                   self.LB: [(x4 - ws, y4), (x4, y4 + hs)],
                   self.RB: [(x3, y3), (x3 + ws, y3 + hs)],
                   self.LT: [(x1 - ws, y1 - hs), (x1, y1)],
                   self.RT: [(x2, y2 - hs), (x2 + ws, y2)],
                   self.OUT: [(x1 - 2 * ws, y1 - 2 * hs), (x3 + 2 * ws, y3 + 2 * hs)]
                   }
        self.dic_sector = dic_set
        # Setor e 2 cantos do retângulo, esq top e dir inf
        self.df_sector = pd.DataFrame(columns=['left', 'top', 'width', 'height', 'sector', 'x1', 'y1', 'x3', 'y3'])
        for key, value in self.dic_sector.items():
            (x1, y1), (x3, y3) = value
            self.df_sector.loc[len(self.df_sector)] = rect_txt.rect + [key, x1, y1, x3, y3]
        return self.dic_sector

    def point_in_rect(self, pt: tuple or list, rect_pts: list) -> bool:
        x, y = pt
        (x0, y0), (x1, y1) = rect_pts
        cond = x0 <= x <= x1 and y0 <= y <= y1
        return cond

    def get_text2box_proximity(self, box_txt: Rect, box_cxa: Rect) -> int:
        """
            args:
            :type box_txt: Rect caixa de texto
            :type box_cxa: Rect caixa entrada dado
        """
        if self.rect_cxa is None or sum(self.rect_cxa.rect) == 0:
            self.proximity = self.FAR
            return self.proximity
        # precisa redefinir os setores para acoplar nova caixa entrada
        self.dic_sector = self.sector_map(rect_txt=box_txt, rect_cxa=box_cxa)
        xt, yt, wt, ht = box_txt.rect
        x, y, w, h = box_cxa.rect
        # Verifica se o ponto inferior esquerdo da caixa pertence a uma região
        pt_lc = (x, y + (h // 2))
        pt_rc = (x + w, y + (h // 2))
        # self.INT: pt_lc
        pts = {self.CENTER: pt_lc,
               self.RIGHT: pt_lc, self.BOTTOM: pt_lc,
               self.LEFT: pt_rc, self.TOP: pt_lc,
               self.LB: pt_rc, self.LT: pt_rc,
               self.RT: pt_lc, self.RB: pt_lc,
               self.OUT: pt_lc}
        # Pula o primeiro item do iterador, desconsidera self.INT
        dic_setor = iter(self.dic_sector.items())
        next(dic_setor)
        # define como 'FAR' caso não achar correspondências
        proximity = self.FAR
        for key, value in dic_setor:
            if self.point_in_rect(pts[key], value):
                proximity = key
                break
        # Verifica se texto esta inteiramente dentro da caixa
        if proximity == self.CENTER:
            box_reg = [(x, y), (x + w, y + h)]
            if self.point_in_rect((xt, yt), box_reg) and self.point_in_rect((xt, yt + ht), box_reg):
                proximity = self.INT
        # calcula distancia
        # dist = np.int32(np.sqrt(np.sum(np.array(x, y) - np.array(x1, y1)) ** 2))
        self.proximity = proximity
        return self.proximity

    # retorna lista das caixas que estão dentro da região/área retangular
    @staticmethod
    def get_boxes_in_region(rect_txt: Rect, df_box_cxa: pd.DataFrame, **kwargs) -> list:
        box_w = np.max(df_box_cxa['w'])
        box_h = np.max(df_box_cxa['h'])
        xt, yt, wt, ht = rect_txt.rect
        fator_borda = kwargs.get('fator_borda', 5)
        gap = fator_borda * ht
        df_box_cxa = df_box_cxa[df_box_cxa['x'] < (xt + box_w + gap)]
        df_box_cxa = df_box_cxa[df_box_cxa['y'] < (yt + box_h + gap)]
        df_box_cxa = df_box_cxa[(df_box_cxa['x'] + df_box_cxa['w']) < (xt + wt + box_w + gap)]
        df_box_cxa = df_box_cxa[(df_box_cxa['y'] + df_box_cxa['h']) < (yt + ht + box_h + gap)]
        box_cxa_lst = df_box_cxa[['x', 'y', 'w', 'h']].values

        return box_cxa_lst

    def show_region(self, img, box_texto, rects_caixa):
        img1 = img.copy()
        cores = {self.INT: (125, 125, 125), self.BOTTOM: (0, 255, 0), self.TOP: (0, 120, 255),
                 self.RIGHT: (0, 150, 200), self.LEFT: (200, 150, 0), self.OUT: (255, 0, 255)}
        for key, value in self.dic_sector.items():
            (x0, y0), (x1, y1) = value
            rect = [x0, y0, x1 - x0, y1 - y0]
            cor = cores[key] if key in cores.keys() else (125, 255, 125)
            cv.rectangle(img1, rect, cor, 1)
            # cv.putText(img1, key, (x0, y1), cv.FONT_HERSHEY_COMPLEX, 0.6, cor)
            # cv.drawMarker(img1, (x0, y1), cor, cv.MARKER_CROSS, 5)

        cv.rectangle(img1, box_texto, (0, 0, 255), 1)
        for i in range(3):
            cls, x, y, w, h = rects_caixa[i]
            cor = cores[cls] if cls in cores.keys() else (125, 255, 125)
            cv.rectangle(img1, [x, y, w, h], cor, 1)
            cv.drawMarker(img1, (x, y + h // 2), cor, cv.MARKER_CROSS, 5)
            cv.drawMarker(img1, (x + w, y + h // 2), cor, cv.MARKER_CROSS, 5)
            cv.putText(img1, str(cls), (x, y + h // 2), cv.FONT_HERSHEY_COMPLEX, 0.6, cor)

        win_name = f'Regions:{len(self.dic_sector)}'
        # cv.namedWindow(win_name, cv.WINDOW_NORMAL)
        cv.imshow(win_name, img1)
        # cv.resizeWindow(win_name, img1.shape[1] // 2, img1.shape[0] // 2)
        cv.waitKey()
        cv.destroyAllWindows()


# Classificar caixas por nivel de proximidade
class BoxClass:
    # variáveis de retorno e gerais
    df3box: pd.DataFrame
    df_sector: pd.DataFrame
    row_sector: pd.Series
    label: str = ''

    def __init__(self, df3box: pd.DataFrame, **kwargs):
        """
        Classifica caixas de entrada conforme posicionamento junto ao texto label
            Args:
                :type df3box: dataframe caixa de texto com 3 caixas de entrada de dado
            Returns:
                df3box dataframe: df da caixa texto (nome campo) e retângulos de 3 caixa de entrada
                classificados por ordem de proximidade
        """
        self.df3box = df3box

    def get_box_proximity(self, **kwargs) -> BoxProximity:
        """
            Classe de posicionamento caixa de entrada mais proxima ao texto label
            Args:
                :type index: indice da row (posição) no dataframe df3box
                :type label: nome do campo (label) coluna 'text'
                :type row: linha do dataframe df3box
            Returns:
                classe da caixa entrada mais proxima ao texto
        """
        row = kwargs.get('row', None)
        index = kwargs.get('index', None)
        label = kwargs.get('label', '').casefold()
        val_index = None
        if row is None:
            if label:
                val_index = (self.df3box['text'] == label.casefold()).idxmax()
                row = self.df3box.loc[val_index, :]
            else:
                row = self.df3box.iloc[index, :]
        else:
            # Use .eq() para comparar a Series com o DataFrame e .all(axis=1) para
            cond_igualdade_total = self.df3box.eq(row).all(axis=1)
            # Obtenha os índices usando a máscara booleana, [0] pega o primeiro item
            val_index = self.df3box.index[cond_igualdade_total].tolist()[0]

        # caixa de texto do rótulo (nome do campo)
        rect_txt = Rect(rect=row.loc['left':'height'])
        # Classificar df_class pela posição da caixa de entrada
        # rect 3 caixas de entrada da row df3cxa (x0..2, y0..2, w0..2, h0..2)
        rect_cxa = Rect(rect=row.loc['x0':'h0'])
        proxx = BoxProximity(rect_txt=rect_txt, rect_cxa=rect_cxa)
        if val_index:
            self.df3box.loc[val_index, 'class0'] = proxx.proximity
        else:
            self.df3box.iloc[index, 'class0'] = proxx.proximity
        self.df_sector = proxx.df_sector
        self.label = row.loc['text']
        self.row_sector = row

        return proxx

    def get_classification(self) -> pd.DataFrame:

        # classifica (rótula) por proximidades do texto com a caixa entrada
        for idx, row in self.df3box.iterrows():
            # rotulo = row.loc['text']
            # caixa de texto do rótulo (nome do campo)
            rect_txt = Rect(rect=row.loc['left':'height'])
            # pto_txt = box_txt.arr_points

            # Classificar df_class pela posição e distância da caixa de entrada
            # Inicia classe BoxProximity para rotular cada caixa de entrada
            for i in range(3):
                # rect 3 caixas de entrada da row df3cxa (x0..2, y0..2, w0..2, h0..2)
                rect_cxa = Rect(rect=row.loc[f'x{i}':f'h{i}'])
                if sum(rect_cxa.rect) > 0:
                    proxx = BoxProximity(rect_txt=rect_txt, rect_cxa=rect_cxa)
                    # classe = proxx.get_text2box_proximity(box_txt=rect_txt, box_cxa=rect_cxa)
                    self.df3box.at[idx, f'class{i}'] = proxx.proximity
        return self.df3box


# Recortar a imagem
class CropBox:
    # variáveis de retorno e gerais
    arr_image: ndarray
    df3box: pd.DataFrame
    df_crop_box: pd.DataFrame
    width: int
    height: int
    crop_img: ndarray
    add_label = False
    # divisor de altura da caixa texto para calculo da borda no tamanho do recorte
    # gap = (altura_texto // border_factor) or 2
    border_factor: int = 5
    grayscale: bool
    index: int
    label: str
    crop_rect: list[tuple]  # pontos [(x1, y1), (x2, y2)]

    def __init__(self, image: ImageBinary, df3box: pd.DataFrame, **kwargs):
        """
            Recorte da imagem na área mínima que envolve o texto e caixa mais proxima
            Args:
                :type df3box: dataframe caixa de texto com 3 caixas de entrada de dado
                :type border_factor: divisor altura caixa para borda do recorte, 5 default
                :type add_label: boolean, 0 or int se o texto (rótulo) fará parte do recorte da imagem
                :type grayscale: True default para escala de cinza
                :type index: default 0, indice do df3box
                :type label: nome do campo da caixa
            Returns:
                list[tuple]: lista de pontos da area de recorte [(x1, y1), (x2, y2)]
        """
        self.width = image.width
        self.height = image.height
        self.df3box = df3box
        self.index = kwargs.get('index', 0)
        self.label = kwargs.get('label', "")
        self.add_label = kwargs.get('add_label', False)
        self.border_factor = kwargs.get('border_factor', 5)
        self.grayscale = True  # kwargs.get('grayscale', True)
        self.arr_image = image.img_gray  # if self.grayscale else image.image
        if df3box.empty:
            self.crop_rect = []
        else:
            self.crop_rect = self.get_cropped_area_points(index=self.index)

    def get_cropped_image(self, **kwargs) -> ndarray:
        """
            Retorna imagem recortada por uma area definida em crop_rect
            Args:
                :type rect: Rect, list[int], list[tuple] da area recortada (saída)
                :type image: ndarray image type
            Returns:
                cropped_image ndarray: image recortada
        """
        image = kwargs.get('image', self.arr_image)
        rect = kwargs.get('rect', None)
        if rect is None:
            self.index = kwargs.get('index', 0)
            self.label = kwargs.get('label', "")
            crop_rect = self.get_cropped_area_points(index=self.index, label=self.label)
            (x1, y1), (x2, y2) = crop_rect
        elif type(rect) is list:
            if type(rect[0]) is tuple:
                (x1, y1), (x2, y2) = rect
            else:
                x1, y1, w, h = rect
                x2, y2 = x1 + w, y1 + h
        elif type(rect) is Rect:
            (x1, y1), _, (x2, y2), _ = rect.points

        # recorte da imagem para tamanho menor envolta do texto
        self.crop_img = image[y1:y2, x1:x2]
        self.crop_rect = [(x1, y1), (x2, y2)]
        return self.crop_img

    def cropped_boxes(self) -> pd.DataFrame:
        self.df_crop_box = self.df3box.copy()
        for idx, row in self.df3box.iterrows():
            crop_rect = self.get_cropped_area_points(row=row)
            # w = crop_rect[1][0] - crop_rect[0][0]
            # h = crop_rect[1][1] - crop_rect[0][1]
            # print(idx, "w x h :", w, 'x', h, crop_rect[0], crop_rect[1])
            self.df_crop_box.loc[idx, 'x1_crop'] = crop_rect[0][0]
            self.df_crop_box.loc[idx, 'y1_crop'] = crop_rect[0][1]
            self.df_crop_box.loc[idx, 'x2_crop'] = crop_rect[1][0]
            self.df_crop_box.loc[idx, 'y2_crop'] = crop_rect[1][1]
        return self.df_crop_box

    # Retorna pontos da area de recorte da imagem associada ao texto e caixa
    def get_cropped_area_points(self, index: int = None, label: str = None, row: pd.Series = None) -> list:

        if row is None:
            if label:
                index = (self.df3box['text'] == label.casefold()).idxmax()
                row = self.df3box.loc[index, :]
            else:
                row = self.df3box.iloc[index, :]
        # pontos da caixa
        x0c, y0c, wc, hc = row.loc['x0':'h0'].tolist()
        x1c, y1c = x0c + wc, y0c + hc
        # pontos do texto
        if self.add_label:
            x0t, y0t, wt, ht = row.loc['left':'height'].tolist()
            x1t, y1t = x0t + wt, y0t + ht
        else:
            # iguala valores para compensar o cálculo sem texto
            x0t, y0t, x1t, y1t, wt, ht = x0c, y0c, x1c, y1c, wc, hc
        # borda do recorte
        gap = (hc // self.border_factor) or 2
        # pontos do recorte da imagem
        x1 = (x0c if x0c < x0t else x0t) - gap
        y1 = (y0c if y0c < y0t else y0t) - gap
        x2 = (x1c if x1c > x1t else x1t) + gap
        y2 = (y1c if y1c > y1t else y1t) + gap
        # delimita dimensão de recorte no array imagem, evita estouro
        x1 = 0 if x1 < 0 else x1
        y1 = 0 if y1 < 0 else y1
        x2 = self.width if x2 > self.width else x2
        y2 = self.height if y2 > self.height else y2

        # recorte da imagem para tamanho menor envolta do texto
        self.crop_rect = [(x1, y1), (x2, y2)]

        return self.crop_rect


# Função principal do aplicativo
def localiza_caixas(path, labels=None, max_search=0, show=False):
    df_txt = pd.DataFrame()
    df_chk = pd.DataFrame()
    df_box = pd.DataFrame()
    imagem = ImageBinary(path=path)
    if labels:
        tess = DetectTesseract(imagem, labels=labels)
        df_txt = tess.get_filters_texts()
    else:
        labels = []
    h, w = imagem.height, imagem.width
    # verificar minino 10% w e h
    size = [(w * 0.02, h * 0.02), (w * 0.7, h * 0.7)]  # Entry
    entry = InputDataBox(imagem, size_rect=size, area_perc=20, max_search=max_search)
    df_box, erro = entry.get_boxes()

    size = [(w * 0.02, h * 0.02), (w * 0.1, h * 0.1)]  # Checkbox
    check = InputDataBox(imagem, size_rect=size, area_perc=20, max_search=max_search)
    df_chk, erro = check.get_boxes()

    df_box = pd.concat([df_box, df_chk], ignore_index=True)
    df_box = df_box.drop_duplicates(ignore_index=True)

    # dataframe de conexão label x 3 caixas entrada, ordem por distância
    link = Text2BoxLinks(df_txt=df_txt, df_box=df_box, labels=labels)
    df3box = link.get_boxes2text_links()

    # Classifica Retângulos
    classe = BoxClass(df3box=df3box)
    df3box = classe.get_classification()
    # proxx = classe.get_box_proximity(label='Telefone')
    # ShowMe(image=imagem.img_gray, row=classe.row_sector, df_sector=proxx.df_sector)

    # Avaliar classe INT para confirmar se há caixa de entrada ao redor do texto
    # avaliar_classe_caixas(imagem, df_class, classe=BoxProximity.INT, show=show)
    crop = CropBox(image=imagem, df3box=df3box, index=0, add_label=len(labels))
    crop.cropped_boxes()
    if len(labels):
        crop_img = crop.get_cropped_image(label='Telefone')
        ShowMe(image=crop_img, scale=0, thickness=1)

    # ShowMe(image=imagem.img_gray, df=classe.df_sector, win_name=classe.label)
    if show:
        if df3box.empty:
            ShowMe(image=imagem.img_gray, df=df_box, scale=0, thickness=1)
        else:
            ShowMe(image=imagem.img_gray, df=df3box, scale=0, thickness=1)
    # print(f'{path}; max_search={max_search}  df_box ={len(df_box)}')

    return len(df_box)


# Função auxiliar para calcular a distância entre ponto_ref com lista de pontos.
def distancia_euclidiana_pontos(ponto_ref: ndarray, pontos: ndarray) -> ndarray:
    """ Return ndarray: lista ordenado das distancias entre pontos """
    # Calcula a diferença entre todos os pontos e o ponto de referência.
    differences = pontos - ponto_ref
    # Calcula a norma 2 (distância euclidiana) para cada diferença.
    # É mais rápido calcular a distância ao quadrado para comparação.
    distancias_quadradas = np.array(np.sqrt(np.sum(differences ** 2, axis=1)), dtype=np.int32)
    # ((ponto2[0] - ponto1[0]) ** 2 + (ponto2[1] - ponto1[1]) ** 2) ** 0.5
    return distancias_quadradas


def localiza_grupo_imagens(qt_img=25):
    df = pd.DataFrame(columns=['imagem', 'l0', 'l1', 'diff'])
    for i in range(qt_img):
        img_path = f'C:/PyCharm/YOLOproject/images/imagem_{i:03}.jpg'
        l0 = localiza_caixas(path=img_path, max_search=0, show=False)
        l1 = localiza_caixas(path=img_path, max_search=1, show=False)
        print(f'{img_path}; df_box, 0 = {l0}   1 = {l1}')
        df.loc[len(df)] = [img_path, l0, l1, (l1 - l0)]
    print('l0,  l1, diff')
    print(df['l0'].sum(), df['l1'].sum(), df['diff'].sum())
    print('fim')


class InputRadio:
    # variáveis de retorno e gerais
    df_box: pd.DataFrame
    rects_found: ndarray
    rects: ndarray
    image: ImageBinary
    metodo: str
    dp: float
    minDist: int
    param1: int
    param2: int
    minRadius: int
    maxRadius: int
    thresh: int
    maxVal: int
    hough_method: int

    all_search = ['gauss', 'mean', 'canny', 'thresh', 'thresh_inv']
    default_search = ['gauss', 'mean']
    search_types: list = default_search

    def __init__(self, im_bin: ImageBinary, **kwargs):
        """
        Detectar Radiobutton por reconhecimento de círculos
        Args:
            :type im_bin: ImageBinary class
            'metodo' (str): 'Hough' usa transformada de Hough ou
            'hough_method' (int):   Metodo de detecção (cv2.HOUGH_GRADIENT)
            'dp' (float): resolução inversa do acumulador (1 significa a mesma resolução da imagem)
            'minDist' (int): distância mínima entre os centros dos círculos detectados
            'param1' (int): limiar superior para o detector de bordas Canny interno
            'param2' (int): limiar acumulador para centro do círculo (quanto menor, mais círculos falsos)
            'minRadius' (int): raio mínimo do círculo a ser detectado
            'maxRadius' (int): raio máximo do círculo a ser detectado
            'ksize' tuple: parametro do gaussianblur (5, 5)
            'thresh' int: parametro image THRESH_BINARY
            'maxVal' int: parametro image THRESH_BINARY
        Returns:
        df_box dataframe: df de retangulos [[x, y, w, h], ...]
        """

        # guarda variável imagem original
        self.image = im_bin
        self.metodo = kwargs.get('metodo', 'HOUGH')
        self.minRadius = kwargs.get('minRadius', 1)
        self.maxRadius = kwargs.get('maxRadius', 20)
        if self.metodo == 'HOUGH':
            self.dp = kwargs.get('dp', 1.2)
            self.minDist = kwargs.get('minDist', 5)
            self.param1 = kwargs.get('param1', 50)
            self.param2 = kwargs.get('param2', 40)
            self.hough_method = kwargs.get('hough_method', cv.HOUGH_GRADIENT)
            self.ksize = kwargs.get('ksize', (5, 5))
            self.detect_circles_hough(self.image)
        else:
            self.thresh = kwargs.get('thresh', 201)
            self.maxVal = kwargs.get('maxVal', 255)
            self.detect_circles_contours(self.image)

    # Detectar círculos usando a Transformada de Hough
    def detect_circles_hough(self, img):

        # Converter para escala de cinza, obrigatorio
        # Aplicar um desfoque (blur) para reduzir ruído, o que ajuda na detecção de círculos
        gray_blurred = img.get_image_thresh(method='GaussianBlur', ksize=self.ksize)
        circles = cv.HoughCircles(gray_blurred, self.hough_method, dp=self.dp, minDist=self.minDist,
                                  param1=self.param1, param2=self.param2,
                                  minRadius=self.minRadius, maxRadius=self.maxRadius)
        if circles is not None:
            # Converter as coordenadas e raios para inteiros
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                # Desenhar o círculo detectado na imagem original
                center = (i[0], i[1])
                radius = i[2]
                # Desenhar o contorno do círculo (verde, espessura 2)
                cv.circle(gray_blurred, center, radius, (0, 255, 0), 2)
                # Desenhar o centro do círculo (vermelho, preenchido)
                cv.circle(gray_blurred, center, 2, (0, 0, 255), 3)

            # Exibir o resultado
            cv.imshow(f'{len(circles)} Circulos Detectados com Hough', gray_blurred)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("Nenhum círculo detectado.")

    def detect_circles_contours(self, img):

        conta = 0
        # Pré-processamento: converter para cinza e binarizar
        thresh = img.get_image_thresh('THRESH_BINARY', thresh=self.thresh, maxVal=self.maxVal)
        # Encontrar contornos
        contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Calcular a área e o perímetro do contorno
            area = cv.contourArea(contour)
            perimeter = cv.arcLength(contour, True)
            # Evitar divisão por zero para contornos muito pequenos
            if perimeter == 0:
                continue
            # Calcular a circularidade: (4 * PI * Área) / Perímetro^2
            # Um círculo perfeito tem circularidade = 1
            circularity = (4 * np.pi * area) / (perimeter ** 2)
            # Definir um limiar para a circularidade (ex: 0.8 a 1.1 pode indicar um círculo)
            if circularity > 0.8 and circularity < 1.2:
                # Encontrar o círculo mínimo de delimitação para obter centro e raio
                ((x, y), radius) = cv.minEnclosingCircle(contour)
                # Ignorar círculos muito pequenos (ruído) ou muito grandes
                if self.minRadius < radius < self.maxRadius:
                    # Desenhar o círculo detectado
                    cv.circle(self.image.image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                    cv.circle(self.image.image, (int(x), int(y)), 2, (0, 0, 255), 3)
                    conta += 1

        # Exibir o resultado
        cv.imshow(f'{conta} Circulos Detectados com Contornos', self.image.image)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    # Execute when the module is not initialized from an import statement.

    nomes_lst = ["Endereco", "Nome", "Telefone", "Data Entrada", "Data Da Entrada",
                 "Data Saida", "Indicacao/Origem", "CEP", "Data Nasc", "Cidade",
                 "Localidade", "CPF", "Cod.", "Respons", "Localiza"]

    # imagem = 'Imagens/Consultoria2.JPG'
    img_path = f'c:/PyCharm/YOLOproject/images/imagem_128.jpg'
    # localiza_grupo_imagens(qt_img=25)
    # localiza_caixas(path=img_path, max_search=1, show=True)

    img = ImageBinary(path=img_path)
    # Detectar RadioButton
    radio1 = InputRadio(im_bin=img, metodo='HOUGH', minRadius=1, maxRadius=25, ksize=(7, 7))
    radio2 = InputRadio(im_bin=img, metodo='CONTOURS', minRadius=1, maxRadius=25, thresh=188, maxVal=255)
