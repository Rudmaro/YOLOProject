import cv2
import numpy as np
import os
import random


def create_synthetic_shapes_dataset(output_dir, num_images_per_shape=10, img_size=64):
    """
    Gera um dataset sintético de círculos e quadrados.
    Args:
        output_dir (str): Diretório para salvar as imagens.
        num_images_per_shape (int): Número de imagens para cada forma.
        img_size (int): Tamanho da imagem (largura e altura).
    """
    # Cria diretórios para as classes se não existirem
    circle_dir = os.path.join(output_dir, 'circles')
    square_dir = os.path.join(output_dir, 'squares')
    os.makedirs(circle_dir, exist_ok=True)
    os.makedirs(square_dir, exist_ok=True)

    for i in range(num_images_per_shape):
        # Gerar imagem para círculo
        img_circle = np.zeros((img_size, img_size, 3), dtype=np.uint8) * 255  # Fundo branco
        center_c = (random.randint(15, img_size - 15), random.randint(15, img_size - 15))
        radius = random.randint(5, 10)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(img_circle, center_c, radius, color, -1)
        cv2.imwrite(os.path.join(circle_dir, f'circle_{i}.png'), img_circle)

        # Gerar imagem para quadrado
        img_square = np.zeros((img_size, img_size, 3), dtype=np.uint8) * 255  # Fundo branco
        # Coordenadas do canto superior esquerdo
        start_point = (random.randint(10, img_size - 20), random.randint(10, img_size - 20))
        # Coordenadas do canto inferior direito
        end_point = (start_point[0] + random.randint(10, 15), start_point[1] + random.randint(10, 15))
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.rectangle(img_square, start_point, end_point, color, -1)
        cv2.imwrite(os.path.join(square_dir, f'square_{i}.png'), img_square)

    print(f"Dataset gerado com {num_images_per_shape} imagens por classe em '{output_dir}'")


# Uso da função
output_dataset_folder = 'output/images/geometric_shapes'
create_synthetic_shapes_dataset(output_dataset_folder, num_images_per_shape=5)
