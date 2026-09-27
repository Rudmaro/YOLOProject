# Projeto YOLOProject
Projeto para treinamento personalizado de um modelo de reconhecimento de objetos em imagens.
Um modelo especifico para reconhecer objetos de entrada de dados (widgets) em imagens das telas de aplicativos baseados em desktop.
Lista de objetos a serem detectados:
    0 Entry: Caixa de entrada
    1 Combobox: Caixa de entrada com lista suspensa
    2 Checkbox: Caixinha de marcação quadrada
    3 Radiobutton: Botão selector redondo
    4 Listbox: Quadro com lista de textos 
    5 Button: Botão clássico (normalmente com texto)
    6 Shape: Imagem retangular
    7 Picture: Figura
    8 Table: Tabela (varias linhas e colunas)
    9 Iconbutton: Botão com imagem ou com aspecto diferente do clássico  
    10 Underline: Linha horizontal para entrada de texto 
    11 Switch: Botão seletor tipo ligado/desligado
O "dataset" com anotações de imagens previamente selecionadas
será usado para treinar com base em modelo já existente, por exemplo, yolov8n.

Etapas principais:
    1. Preparar os dados para treinamento
        Utiliza projeto 'LabelmeProject'
    2. Carregar um modelo pré-treinado (recomendado para fine-tuning)
        model = YOLO('yolov8n.pt')
    3. Treinar o modelo com seu dataset personalizado
        results = model.train(data=YOLOvar.data_yaml, epochs=30)  # imgsz=640

Preparação e refinamento das imagens para treinamento do modelo

# Funções para anotação formato labelme e edição de arquivos json 
Recria um arquivo json de anotação com uso IA,
aplicando dataset treinado (best.pt) para reconhecer shapes das imagens

Sequencia:
1. pegar arquivo /images/imagem\_00n.jpg
2. joga arquivo imagem .jpg em memória (formato ndarray) 
3. carregar modelo best.pt e aplicar inferência à imagem
4. visualizar imagem e resultados das caixas delimitadores retornadas
5. salvar arquivo de anotação das caixas delimitadoras em json na pasta dataset/json
6. abrir o programa labelme 
7. editar e salvar imagem no labelme


# LabelmeProject
Projeto com funções auxiliares para gerar o "dataset" de anotações de imagens.
Pode utilizar como base o aplicativo 'Labelme', por exemplo.

# Preparar os dados de treinamento

    Funções para Treinamento do Modelo
    Crie uma lista numérica para definir a quantidade de imagens utilizadas
        lista = list(range(0, 501))  # 0 .. 501
    Classificar as imagens listadas por qualidade e tipo
        lista = classifica\_lista(lista)
    Dividir a lista em duas partes para treino e validação, teste opcional
        treino, valida = dividir_treino_valida(lista, 70)
        Converter formato de arquivo json para yolo 
            Grava imagens no conjunto de dados 'dataset' descritos em mydata.yaml
            converter\_json\_para\_yolo(lista\_num=treino, pasta='train')
            converter\_json\_para\_yolo(lista\_num=valida, pasta='val')
    
    # dataset/ <dir yolo>
    # label.txt
    #   images/ <dir yolo images>
    #       train/ <dir yolo images train>
    #           image\_000.jpg
    #       val/
    #           image\_001.jpg
    #   labels/  <dir yolo labels>
    #       train/  <dir yolo train>
    #           image\_000.txt
    #               id  x\_center    y\_center    weight      height
    #               1   0.830368    0.829574    0.080276    0.045113
    #       val/
    #           image\_001.txt
    #               id  x\_center    y\_center    weight      height
    #               2   0.830368    0.829574    0.080276    0.045113
  

