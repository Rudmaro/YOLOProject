import cv2 as cv
import numpy as np
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)))


# custom data structure to hold the state of a Canny edge filter
class EdgeFilter:

    def __init__(self, kernelSize=None, erodeIter=None, dilateIter=None, canny1=None,
                    canny2=None):
        self.kernelSize = kernelSize
        self.erodeIter = erodeIter
        self.dilateIter = dilateIter
        self.canny1 = canny1
        self.canny2 = canny2


# custom data structure to hold the state of an HSV filter
class HsvFilter:

    def __init__(self, hMin=None, sMin=None, vMin=None, hMax=None, sMax=None, vMax=None,
                 sAdd=None, sSub=None, vAdd=None, vSub=None):
        self.hMin = hMin
        self.sMin = sMin
        self.vMin = vMin
        self.hMax = hMax
        self.sMax = sMax
        self.vMax = vMax
        self.sAdd = sAdd
        self.sSub = sSub
        self.vAdd = vAdd
        self.vSub = vSub


# Recortar imagem (palheiro) conforme setor
# 1 | 2  ,   .|12|.   ,     |.|
# __5__  ,   __5__    ,   13|5|24     # 15 --> eh 5 dentro de 1
# 3 | 4  ,   .|34|.   ,     |.|
# retorna imagem recortada e x, y atualizados para calcular ponto real de clique
# usa setores pre definidos se há entrada setor
# recorte palheiro pelo retângulo se há rect, e aplica borda ao rect
def crop_image(img, rect=None, setor=None, borda=None):
    if img is None:
        return [], 0, 0
    setor = setor or 0
    # imagem mantida
    if setor == 0 and rect is None:
        return img, 0, 0
    if type(setor) is str:
        setor = setor.lower()
    elif type(setor) is tuple:
        rect = setor   # define entrada por rect

    # inicia valores da imagem maior (palheiro)
    x, y, alt, lrg = 0, 0, img.shape[0], img.shape[1]

    # se recorte for por entrada de retângulo rect
    if rect is not None:
        x, y, h, w = rect
        if setor == '%':
            x, y, h, w = x/100, y/100, h/100, w/100
            x = int(lrg * x)
            y = int(alt * y)
            w = int(lrg * w)
            h = int(alt * h)
        rect = (x, y, h, w)
        setor = (x, y, h, w)
        # aplica borda e ajusta quadro (rect) para caber dentro da imagem (palheiro)
        if borda is None:
            borda = (0, 0)
        gap_x, gap_y = borda
        x, y, h, w = rect
        x1 = (x - gap_x) if (x - gap_x) > 0 else 0
        y1 = (y - gap_y) if (y - gap_y) > 0 else 0
        x2, y2 = x + w + gap_x, y + h + gap_y
        wr, hr = x2 - x1, y2 - y1
        wr = wr if wr < lrg else lrg
        hr = hr if hr < alt else alt
        rect1 = (x1, y1, hr, wr)  # rect ajustado dentro da imagem
        x, y, h, w = rect1
        crop_scr = img[y:y + h, x:x + w]
        return crop_scr, x, y

    # entrada por setor
    x, y, h, w = 0, 0, alt, lrg
    # imagem recortada com w e h em 50% (25% de area da imagem)
    if setor in [1, 'se']:  # canto superior esquerdo, w e h em 50%
        x, y, w, h = 0, 0, w // 2, h // 2
    elif setor in [2, 'sd']:  # canto superior direito, w e h em 50%
        x, y, w, h = w // 2, 0, w // 2, h // 2
    elif setor in [3, 'ie']:  # canto inferior esquerdo, w e h em 50%
        x, y, w, h = 0, h // 2, w // 2, h // 2
    elif setor in [4, 'id']:  # canto inferior direito, w e h em 50%
        x, y, w, h = w // 2, h // 2, w // 2, h // 2
    elif setor in [5, 'c']:  # recorte no centro, w e h em 50%
        x, y, w, h = w // 4, h // 4, w // 2, h // 2
    elif setor in [12, 'cs']:  # meio entre 1 e 2 (centro superior)
        x, y, w, h = w // 4, 0, w // 2, h // 2
    elif setor in [34, 'ci']:  # meio entre 3 e 4 (centro inferior)
        x, y, w, h = w // 4, h // 2, w // 2, h // 2
    elif setor in [13, 'ce']:  # meio entre 1 e 3 (centro esquerdo)
        x, y, w, h = 0, h // 4, w // 2, h // 2
    elif setor in [24, 'cd']:  # meio entre 2 e 4 (centro direito)
        x, y, w, h = w // 2, h // 4, w // 2, h // 2

    # imagem recortada com w e h em 25% (12.5% de area da imagem)
    elif setor in [15, '1c', 'sec']:  # centro do canto superior esquerdo (5 dentro de 1)
        x, y, w, h = w // 8, h // 8, w // 4, h // 4
    elif setor in [1254, '1d', 'sed']:  # metade direita do setor 1 (direita do canto se)
        x, y, w, h = w // 4, 0, w // 4, h // 2
    elif setor in [1153, '1e', 'see']:  # metade esquerda do setor 1 (esquerda do canto se)
        x, y, w, h = 0, 0, w // 4, h // 2
    elif setor in ['2e', 'sde']:  # metade esquerda do setor 2 (esquerda do canto sd)
        x, y, w, h = w // 2, 0, w // 4, h // 2
    elif setor in ['2c', 'sdc']:  # centro do setor 2 (centro do canto sd)
        x, y, w, h = w // 2 + w // 8, 0 + h // 8, w // 4, h // 4
    elif setor in ['2d', 'sdd']:  # metade direita do setor 2 (direita do canto sd)
        x, y, w, h = (w * 3) // 4, 0, w // 4, h // 2  # w/2+w/4 ou w*3/4
    elif setor in ['2dd', 'sddd']:  # metade direita da direita do setor 2 (d de d canto sd)
        x, y, w, h = (w * 7) // 8, 0, w // 8, h // 2  # w/2+w/4 ou w*3/4
    elif setor in [152, 's']:  # metade lado superior
        x, y, w, h = 0, 0, w, h // 2
    elif setor in [354, 'i']:  # metade lado inferior
        x, y, w, h = 0, h // 2, w, h // 2
    elif setor in [153, 'e']:  # metade lado esquerda
        x, y, w, h = 0, 0, w // 2, h
    elif setor in [254, 'd']:  # metade lado direito
        x, y, w, h = w // 2, 0, w // 2, h
    elif setor in [21, 'sdse']:  # 1 dentro de 2, canto se do canto sd
        x, y, w, h = (w * 2) // 4, 0, w // 4, h // 4
    elif setor in [22, 'sdsd']:  # 2 dentro de 2, canto sd do canto sd
        x, y, w, h = (w * 3) // 4, 0, w // 4, h // 4
    elif setor in [23, 'sdie']:  # 3 dentro de 2, canto ie do canto sd
        x, y, w, h = (w * 2) // 4, (h * 1) // 4, w // 4, h // 4
    elif setor in [24, 'sdid']:  # 4 dentro de 2, canto id do canto sd
        x, y, w, h = (w * 3) // 4, h // 4, w // 4, h // 4
    elif setor in [2324, 'sdiesdid']:  # 24 dentro de 23
        x, y, w, h = int((w * 2.5) / 4), (h * 1) // 4, w // 4, h // 4
    elif setor in [11, 'sese']:  # 1 dentro de 1, canto se do canto se
        x, y, w, h = 0, 0, w // 4, h // 4
    elif setor in [55, 'cc']:  # recorte no centro do centro, w e h em 25%
        x, y, w, h = int(3*(w/8)), int(3*(h/8)), w//8, h//8
    crop_scr = img[y:y + h, x:x + w]
    return crop_scr, x, y


class Vision:
    # Constants
    TRACKBAR_WINDOW = 'Trackbars'

    eps = None
    agulha_img = None
    palheiro_img = None
    agulha_w = 0
    agulha_h = 0
    method = None
    # cores_int = [i for i in range(1, 257)]
    cores = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (125, 125, 125),
             (0, 0, 125), (125, 0, 0), (0, 125, 0), (125, 125, 0), (125, 0, 125), (0, 125, 125), (75, 75, 75)]
    max_results = 10

    grayscale = False

    # Construtor
    def __init__(self, agulha_img_path, method=cv.TM_CCOEFF_NORMED,
                 eps=0.5, max_results=10):

        self.eps = eps
        # Melhor método encontrado foi TM_CCOEFF_NORMED

        self.max_results = max_results

        # Carrega a image a ser localizada
        # https://docs.opencv.org/4.2.0/d4/ds8/group_imgcodecs.html
        # self.agulha_img = cv.imread(agulha_img_path, cv.IMREAD_UNCHANGED)
        if agulha_img_path is not None:
            self.set_img_agulha(agulha_img_path)
            # Salva as dimensões da agulha a ser localizada
            self.agulha_h = self.agulha_img.shape[0]
            self.agulha_w = self.agulha_img.shape[1]

        # Existe 6 métodos para escolher
        # cv.TM_CCOEFF_NORMED, cv.TM_SQDIFF_NORMED ...
        self.method = method

    def set_img_agulha(self, path=None):
        if type(path) is str:
            if path[-4:].lower() in '.jpg, .jpeg, .bmp':
                self.agulha_img = cv.imread(path, cv.IMREAD_UNCHANGED)
        else:
            self.agulha_img = path

    # Verifica se agulha é grayscale
    def isgray(self, img):
        if not hasattr(img, 'shape'):
            return False
        if len(img.shape) < 3 or img.shape[2] == 1:
            return True
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        if (b == g).all() and (b == r).all():
            return True
        return False

    # Converte imagens para escala de cinza
    def bgr2gray(self, img):
        if not self.isgray(img):
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        return img


    # Retorna uma lista [x, y, w, h] de retângulos de imagens correspondentes no palheiro
    def find(self, palheiro_img, threshold=0.7, grayscale=False, setor=None):
        # havendo setor recorta o palheiro
        xs, ys = 0, 0
        if setor is not None:
            palheiro_img, xs, ys = crop_image(img=palheiro_img, setor=setor)
        # Sai fora se agulha é maior que o palheiro
        if self.agulha_h > palheiro_img.shape[0] or self.agulha_w > palheiro_img.shape[1]:
            print('Agulha maior que o palheiro...')
        #  Melhor resultado com threshold 0,685
        if grayscale:
            palheiro_img = self.bgr2gray(palheiro_img)
        # Roda algorítimo openCV
        result = cv.matchTemplate(palheiro_img, self.agulha_img, self.method)

        # min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
        # print(f"Melhor correspondência na posição (topo, esquerdo): {str(max_loc)}")
        # print(f"Maior índice de confiança (0..1) do branco mais brilhante: {str(max_val)}")
        # print(result)

        # limite confiável de correspondência, branco mais brilhante é 1 (0..1)
        locais = np.where(result >= threshold)
        # print("Threshold >= 0.85 ", locais)
        locais = list(zip(*locais[::-1]))
        # print("Locais", locais)

        # Se não encontrar correspondências, locais, retorna agulha
        # array vazio para permitir concatenar resultados em erros
        if not locais:
            return np.array([], dtype=np.int32).reshape(0, 4)

        # Criar a lista de retângulos [x, y, w, h],
        # duplica o retângulo para garantir retorno de pelo menos um em cv.groupRectangles
        rectangles = []
        for loc in locais:
            rect = [int(loc[0]), int(loc[1]), self.agulha_h, self.agulha_w]
            # Gerar duplicidade para garantir retorno de pelo menos um retângulo em cv.groupRectangles
            rectangles.append(rect)
            rectangles.append(rect)

        # Agrupamento de retângulos próximos resultando em um retângulo (central)
        # groupThreshold geralmente é 1, se 0 nenhum agrupamento é feito,
        # se 2 precisa ter pelo menos 3 retângulos para agrupar
        # eps (0.5) é a relativa diferença entre lados dos retângulos para agrupar (mesclar)
        rectangles, pesos = cv.groupRectangles(rectangles, groupThreshold=1, eps=0.5)
        # print(targets)

        # Por razões de desempenho o resultado está limitado a um número máximo
        if len(rectangles) > self.max_results:
            print('Aviso: Muitos resultados, aumente o threshold')
            rectangles = rectangles[:self.max_results]
        if setor is not None:
            return rectangles, xs, ys
        return rectangles


    # Pega a lista [x, y, w, h] de retângulos retornados por find(),
    # e retorna a lista [x, y] da posição central dos retângulos (mira do clique)
    @staticmethod
    def get_click_points(rectangles):

        pontos = []
        # Varrer todos os pontos localizados
        for (x, y, h, w) in rectangles:
            # Determine centro da posição (retângulo)
            x_cen = int(x + (w / 2))
            y_cen = int(y + (h / 2))
            # Guardar os pontos
            pontos.append((x_cen, y_cen))

        return pontos

    # Pega a lista [x, y, w, h] de retângulos e a agulha canvas original (palheiro),
    # e retorna a agulha com retângulos desenhados
    def draw_rectangles(self, palheiro_img, rectangles, line_type=None, thickness=None, color=None):
        if palheiro_img is None:
            return
        # Cores BGR
        tipo = line_type or cv.LINE_4
        esp = thickness or 2
        for i, (x, y, h, w) in enumerate(rectangles):
            x = x if x > 0 else 0
            y = y if y > 0 else 0
            top_left = (x, y)
            hp, wp = palheiro_img.shape[0], palheiro_img.shape[1]
            w = w if w < wp else wp
            h = h if h < hp else hp
            bottom_right = (x+w, y+h)
            # tuple(np.random.choice(range(125), size=3))
            cor = self.cores[i] if i + 1 < len(self.cores) else (25, 0, 25)
            if color:
                cor = color
            cv.rectangle(palheiro_img, top_left, bottom_right,
                         color=cor, thickness=esp, lineType=tipo)
        return palheiro_img

    # Desenha círculos no centro [x, y] com diâmetro = w.
    # Entrada retangulo [x, y, w, y]
    def draw_circles(self, palheiro_img, rectangles, line_type=None, thickness=None, color=None):
        # Cores BGR
        tipo = line_type or cv.LINE_4
        esp = thickness or 2
        for i, (x, y, h, w) in enumerate(rectangles):
            centro = (int(x+w/2), int(y+h/2))
            raio = int((h if h < w else w)/2)
            cor = self.cores[i] if i+1 < len(self.cores) else (25, 0, 25)
            if color:
                cor = color
            cv.circle(palheiro_img, centro, radius=raio, color=cor,
                      thickness=esp, lineType=tipo)
        return palheiro_img


    # Pega a lista [x, y] de pontos centrais dos retângulos e a agulha canvas original (palheiro),
    # e retorna a agulha com marcas de mira '+' desenhadas
    def draw_crosshairs(self, palheiro_img, points):
        tipo_marca = cv.MARKER_CROSS
        for i, (x_cen, y_cen) in enumerate(points):
            pts = (int(x_cen), int(y_cen))
            cor = self.cores[i] if i+1 < len(self.cores) else (25, 0, 25)
            cv.drawMarker(palheiro_img, position=pts, color=cor,
                          markerType=tipo_marca, markerSize=40, thickness=4)
        return palheiro_img

    # Retorna retângulos (x, y, w, h) a partir do ponto central (xc, yc)
    # Agrupamento de retângulos próximos resultando em um retângulo (central)
    def rect_at_center_point(self, points, w=None, h=None, eps=None, max_results=None):
        self.eps = eps or self.eps or 0.5
        self.max_results = max_results or self.max_results or 15
        rectangles = []
        for i, (xc, yc) in enumerate(points):
            w, h = w or self.agulha_w, h or self.agulha_h
            x, y = int(xc - w/2), int(yc - h/2)
            rect = [x, y, h, w]
            # Gerar duplicidade para garantir retorno de pelo menos um retângulo em cv.groupRectangles
            rectangles.append(rect)
            rectangles.append(rect)
        # groupThreshold geralmente é 1, se 0 nenhum agrupamento é feito,
        # se 2 precisa ter pelo menos 3 retângulos para agrupar
        # eps (0.5) é a relativa diferença entre lados dos retângulos para agrupar (mesclar)
        rectangles, pesos = cv.groupRectangles(rectangles, groupThreshold=1, eps=self.eps)
        # print(targets)

        # Por razões de desempenho o resultado está limitado a um número máximo
        if len(rectangles) > self.max_results:
            print('Aviso: Muitos resultados, aumente o threshold')
            rectangles = rectangles[:self.max_results]

        return rectangles

    # Mostra agulha na tela ou grava em arquivo com nome titulo
    def show_image(self, palheiro_img, titulo='Resultado', wait=False, write=False):
        cv.imshow(titulo, palheiro_img)
        if wait:
            cv.waitKey()
        if write:
            cv.imwrite(titulo, palheiro_img)

    # Criar janela guia com controles para ajustar argumentos em tempo real
    def ini_control_gui(self):

        cv.namedWindow(self.TRACKBAR_WINDOW, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.TRACKBAR_WINDOW, 350, 700)

        # Requerido retorno de chamada callback. Usaremos getTrackbarPos() para fazer marcações lookups
        # ao invés de usar a chamada callback
        def nothing(position):
            pass

        # Criar trackbars para bracketing
        # OpenCV scale for HSV is H: 0-170, S:0-255, V:0-255
        cv.createTrackbar('HMin', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMin', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('HMax', self.TRACKBAR_WINDOW, 0, 179, nothing)
        cv.createTrackbar('SMax', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VMax', self.TRACKBAR_WINDOW, 0, 255, nothing)

        # Aplica valor padrão para Max HSV trackbars
        cv.setTrackbarPos('HMax', self.TRACKBAR_WINDOW, 179)
        cv.setTrackbarPos('SMax', self.TRACKBAR_WINDOW, 255)
        cv.setTrackbarPos('VMax', self.TRACKBAR_WINDOW, 255)

        # Criar trackbar para incremento/decremento de Saturação e Valor
        cv.createTrackbar('SAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('SSub', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VAdd', self.TRACKBAR_WINDOW, 0, 255, nothing)
        cv.createTrackbar('VSub', self.TRACKBAR_WINDOW, 0, 255, nothing)

        # Trackbar para criação de bordas
        cv.createTrackbar('KernelSize', self.TRACKBAR_WINDOW, 1, 30, nothing)
        cv.createTrackbar('ErodeIter', self.TRACKBAR_WINDOW, 1, 5, nothing)
        cv.createTrackbar('DilateIter', self.TRACKBAR_WINDOW, 1, 5, nothing)
        cv.createTrackbar('Canny1', self.TRACKBAR_WINDOW, 0, 200, nothing)
        cv.createTrackbar('Canny2', self.TRACKBAR_WINDOW, 0, 500, nothing)
        # Set default value for Canny trackbars
        cv.setTrackbarPos('KernelSize', self.TRACKBAR_WINDOW, 5)
        cv.setTrackbarPos('Canny1', self.TRACKBAR_WINDOW, 100)
        cv.setTrackbarPos('Canny2', self.TRACKBAR_WINDOW, 200)

    # returns an HSV filter object based on the control GUI values
    def get_hsv_filter_from_controls(self):
        # Pega posição corrente de todos os controles (trackbars)
        hsv_filter = HsvFilter()
        hsv_filter.hMin = cv.getTrackbarPos('HMin', self.TRACKBAR_WINDOW)
        hsv_filter.sMin = cv.getTrackbarPos('SMin', self.TRACKBAR_WINDOW)
        hsv_filter.vMin = cv.getTrackbarPos('VMin', self.TRACKBAR_WINDOW)
        hsv_filter.hMax = cv.getTrackbarPos('HMax', self.TRACKBAR_WINDOW)
        hsv_filter.sMax = cv.getTrackbarPos('SMax', self.TRACKBAR_WINDOW)
        hsv_filter.vMax = cv.getTrackbarPos('VMax', self.TRACKBAR_WINDOW)
        hsv_filter.sAdd = cv.getTrackbarPos('SAdd', self.TRACKBAR_WINDOW)
        hsv_filter.sSub = cv.getTrackbarPos('SSub', self.TRACKBAR_WINDOW)
        hsv_filter.vAdd = cv.getTrackbarPos('VAdd', self.TRACKBAR_WINDOW)
        hsv_filter.vSub = cv.getTrackbarPos('VSub', self.TRACKBAR_WINDOW)
        return hsv_filter


    # set HSV filter object to the control GUI values
    def set_hsv_filter_from_controls(self, hsv_filter: HsvFilter):
        # Atualiza a posição de todos os controles (trackbars)
        cv.setTrackbarPos('HMin', self.TRACKBAR_WINDOW, hsv_filter.hMin)
        cv.setTrackbarPos('SMin', self.TRACKBAR_WINDOW, hsv_filter.sMin)
        cv.setTrackbarPos('VMin', self.TRACKBAR_WINDOW, hsv_filter.vMin)
        cv.setTrackbarPos('HMax', self.TRACKBAR_WINDOW, hsv_filter.hMax)
        cv.setTrackbarPos('SMax', self.TRACKBAR_WINDOW, hsv_filter.sMax)
        cv.setTrackbarPos('VMax', self.TRACKBAR_WINDOW, hsv_filter.vMax)
        cv.setTrackbarPos('SAdd', self.TRACKBAR_WINDOW, hsv_filter.sAdd)
        cv.setTrackbarPos('SSub', self.TRACKBAR_WINDOW, hsv_filter.sSub)
        cv.setTrackbarPos('VAdd', self.TRACKBAR_WINDOW, hsv_filter.vAdd)
        cv.setTrackbarPos('VSub', self.TRACKBAR_WINDOW, hsv_filter.vSub)
        return hsv_filter

    # returns a Canny edge filter object based on the control GUI values
    def get_edge_filter_from_controls(self):
        # Get current positions of all trackbars
        edge_filter = EdgeFilter()
        edge_filter.kernelSize = cv.getTrackbarPos('KernelSize', self.TRACKBAR_WINDOW)
        edge_filter.erodeIter = cv.getTrackbarPos('ErodeIter', self.TRACKBAR_WINDOW)
        edge_filter.dilateIter = cv.getTrackbarPos('DilateIter', self.TRACKBAR_WINDOW)
        edge_filter.canny1 = cv.getTrackbarPos('Canny1', self.TRACKBAR_WINDOW)
        edge_filter.canny2 = cv.getTrackbarPos('Canny2', self.TRACKBAR_WINDOW)
        return edge_filter


    # set a Canny edge filter object to the control GUI values
    def set_edge_filter_from_controls(self, edge_filter: EdgeFilter):
        # update current positions of all trackbars
        cv.setTrackbarPos('KernelSize', self.TRACKBAR_WINDOW, edge_filter.kernelSize)
        cv.setTrackbarPos('ErodeIter', self.TRACKBAR_WINDOW, edge_filter.erodeIter)
        cv.setTrackbarPos('DilateIter', self.TRACKBAR_WINDOW, edge_filter.dilateIter)
        cv.setTrackbarPos('Canny1', self.TRACKBAR_WINDOW, edge_filter.canny1)
        cv.setTrackbarPos('Canny2', self.TRACKBAR_WINDOW, edge_filter.canny2)
        return edge_filter

    # Pega agulha e um filtro HSV, aplica o filtro e resulta em uma agulha
    # se o filtro não é informado, o controle GUI trackbars será usado
    def apply_hsv_filter(self, original_image, hsv_filter=None):
        # Converter agulha para HSV
        hsv = cv.cvtColor(original_image, cv.COLOR_BGR2HSV)

        # Se não haver um valor de filtro usa valores de filtro da GUI
        if not hsv_filter:
            hsv_filter = self.get_hsv_filter_from_controls()

        # adiciona ou subtrai Saturação e  (h = hue = matiz)
        h, s, v = cv.split(hsv)
        s = self.shift_channel(s, hsv_filter.sAdd)
        s = self.shift_channel(s, -hsv_filter.sSub)
        v = self.shift_channel(v, hsv_filter.vAdd)
        v = self.shift_channel(v, -hsv_filter.vSub)
        hsv = cv.merge([h, s, v])

        # Aplica valor HSV minimo e máximo para mostrar
        lower = np.array([hsv_filter.hMin, hsv_filter.sMin, hsv_filter.vMin])
        upper = np.array([hsv_filter.hMax, hsv_filter.sMax, hsv_filter.vMax])
        # Aplica the thresholds
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(hsv, hsv, mask=mask)

        # Converter retorno para BRG para imshow() para mostrar propriamente
        img = cv.cvtColor(result, cv.COLOR_HSV2BGR)

        return img

    # Pega uma agulha e o filtro de borda Canny, e retorna a agulha resultante
    # se filtro não informado, será usado o controle GUI trackbars
    def apply_edge_filter(self, original_image, edge_filter=None):
        # if we haven't been given a defined filter, use the filter values from the GUI
        if not edge_filter:
            edge_filter = self.get_edge_filter_from_controls()

        kernel = np.ones((edge_filter.kernelSize, edge_filter.kernelSize), np.uint8)
        eroded_image = cv.erode(original_image, kernel, iterations=edge_filter.erodeIter)
        dilated_image = cv.dilate(eroded_image, kernel, iterations=edge_filter.dilateIter)

        # canny edge detection
        result = cv.Canny(dilated_image, edge_filter.canny1, edge_filter.canny2)

        # convert single channel image back to BGR
        img = cv.cvtColor(result, cv.COLOR_GRAY2BGR)

        return img


    # Aplicar ajustes para o canal HSV
    def shift_channel(self, c, amount):
        if amount > 0:
            lim = 255 - amount
            c[c >= lim] = 255
            c[c < lim] += amount
        elif amount < 0:
            amount = -amount
            lim = amount
            c[c <= lim] = 0
            c[c > lim] -= amount
        return c

    # Método ORB detector de ponto-chave FAST e descritor BRIEF (qt_recs, cantos)
    # Opção de usar detecção de objetos no lugar do find (em escala)
    def match_keypoints(self, original_image, patch_size=32, min_match_count=5):
        # min_match_count = 5

        orb = cv.ORB_create(edgeThreshold=0, patchSize=patch_size)
        keypoints_agulha, descriptors_agulha = orb.detectAndCompute(self.agulha_img, None)
        orb2 = cv.ORB_create(edgeThreshold=0, patchSize=patch_size, nfeatures=2000)
        keypoints_palheiro, descriptors_palheiro = orb2.detectAndCompute(original_image, None)

        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)

        search_params = dict(checks=50)

        try:
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(descriptors_agulha, descriptors_palheiro, k=2)
        except cv.error:
            return None, None, [], [], None

        # store all the good matches as per Lowe's ratio test.
        good = []
        points = []

        for pair in matches:
            if len(pair) == 2:
                if pair[0].distance < 0.7 * pair[1].distance:
                    good.append(pair[0])

        if len(good) > min_match_count:
            print(f'Correspondências {len(good)}, kp {len(keypoints_agulha)}')
            for match in good:
                points.append(keypoints_palheiro[match.trainIdx].pt)
            # print(points)

        return keypoints_agulha, keypoints_palheiro, good, points

    def centroid(self, point_list):
        point_list = np.asarray(point_list, dtype=np.int32)
        length = point_list.shape[0]
        sum_x = np.sum(point_list[:, 0])
        sum_y = np.sum(point_list[:, 1])
        return [np.floor_divide(sum_x, length), np.floor_divide(sum_y, length)]
