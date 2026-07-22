"""Generate starter Jupyter notebooks for all 9 labs."""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent


def make_notebook(title: str, sections: list[tuple[str, str]]) -> dict:
    cells = []

    # Title cell
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [f"# {title}\n\nСтартовый блокнот. Заполните ячейки с кодом в соответствии с заданием.\n"],
    })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Подготовка среды\n\nУстановите необходимые библиотеки:"],
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": ["# !pip install -q torch torchvision opencv-python matplotlib numpy\n"],
        "outputs": [],
    })

    for section_title, instructions in sections:
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {section_title}\n\n{instructions}\n"],
        })
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": ["# TODO: реализуйте этот раздел\npass\n"],
            "outputs": [],
        })

    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": ["## Выводы\n\nПроанализируйте полученные результаты, сравните методы, укажите ограничения."],
    })

    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "source": ["# TODO: напишите выводы\n"],
        "outputs": [],
    })

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "cells": cells,
    }


def save_notebook(filename: str, notebook: dict):
    path = HERE / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, ensure_ascii=False, indent=1)
    print(f"Created {path.name}")


save_notebook(
    "lab01_viola_jones.ipynb",
    make_notebook(
        "ЛР-1. Классическое детектирование по Виоле-Джонса",
        [
            ("Загрузка данных", "Загрузите датасет FDDB или Caltech Pedestrians. Визуализируйте примеры."),
            ("Каскад Хаара", "Загрузите предобученный каскад из OpenCV. Примените к тестовым изображениям. Оцените TPR, FPR."),
            ("Обучение каскада", "Подготовьте положительные и отрицательные выборки. Обучите каскад с opencv_traincascade."),
            ("HOG + SVM", "Извлеките HOG-дескрипторы. Обучите LinearSVC. Сравните с каскадом Хаара по mAP."),
            ("Визуализация", "Покажите примеры детекций обоими методами. Постройте Precision-Recall кривые."),
        ],
    ),
)

save_notebook(
    "lab02_cnn_comparison.ipynb",
    make_notebook(
        "ЛР-2. Сравнительный анализ свёрточных архитектур",
        [
            ("Загрузка данных", "Загрузите CIFAR-10 или CSU101. Создайте DataLoader с аугментацией."),
            ("Базовое воспроизведение", "Загрузите предобученный ResNet50. Замените head. Дообучите. Зафиксируйте accuracy."),
            ("SE block", "Реализуйте Squeeze-and-Excitation блок. Встройте в ResNet. Обучите."),
            ("ECA block", "Реализуйте ECA блок. Сравните с SE по точности и числу параметров."),
            ("Inverted bottleneck", "Реализуйте блок MobileNetV2. Постройте график accuracy vs params."),
        ],
    ),
)

save_notebook(
    "lab03_yolo.ipynb",
    make_notebook(
        "ЛР-3. Детекция объектов YOLO",
        [
            ("Подготовка датасета", "Соберите 100-200 изображений. Разметьте в формате YOLO."),
            ("Обучение", "Обучите YOLOv8/v11 на своём датасете. Настройте гиперпараметры."),
            ("Оценка", "Вычислите Precision, Recall, mAP@0.5, mAP@0.5:0.95."),
            ("Инференс", "Проведите детекцию на тестовых изображениях и видео. Визуализируйте bounding boxes."),
            ("Анализ ошибок", "Проанализируйте false positive и false negative."),
        ],
    ),
)

save_notebook(
    "lab04_unet.ipynb",
    make_notebook(
        "ЛР-4. Семантическая сегментация U-Net",
        [
            ("Загрузка данных", "Загрузите CamVid или Cityscapes. Реализуйте pipeline аугментации."),
            ("U-Net реализация", "Реализуйте U-Net с бэкбоном ResNet-34. Настройте функцию потерь."),
            ("FCN реализация", "Реализуйте FCN. Сравните с U-Net по IoU."),
            ("Обучение", "Обучите с Cross-Entropy и Dice Loss. Сравните результаты."),
            ("Визуализация", "Покажите исходное изображение, маску, предсказание для нескольких примеров."),
        ],
    ),
)

save_notebook(
    "lab05_dcgan.ipynb",
    make_notebook(
        "ЛР-5. Генерация изображений DCGAN",
        [
            ("Загрузка данных", "Загрузите CelebA или CIFAR-10. Создайте DataLoader."),
            ("Generator", "Реализуйте архитектуру генератора DCGAN."),
            ("Discriminator", "Реализуйте архитектуру дискриминатора."),
            ("Цикл обучения", "Реализуйте цикл обучения GAN. Фиксируйте loss."),
            ("Визуализация", "Покажите генерируемые изображения каждые N эпох."),
            ("Стабилизация", "Примените label smoothing, LeakyReLU. Оцените FID."),
        ],
    ),
)

save_notebook(
    "lab06_vit.ipynb",
    make_notebook(
        "ЛР-6. Vision Transformer ViT",
        [
            ("Загрузка данных", "Загрузите CSU101 или подмножество ImageNet."),
            ("ViT fine-tuning", "Загрузите предобученный ViT-Base. Замените head. Дообучите."),
            ("Карты внимания", "Визуализируйте self-attention maps для разных слоёв."),
            ("Сравнение с CNN", "Сравните ViT с ResNet-50 по accuracy и FPS."),
            ("Patch size", "Экспериментируйте с patch_size=16 vs 32."),
        ],
    ),
)

save_notebook(
    "lab07_siamese.ipynb",
    make_notebook(
        "ЛР-7. Метрическое обучение (Siamese Networks)",
        [
            ("Загрузка данных", "Загрузите LFW. Сформируйте триплеты (anchor, positive, negative)."),
            ("Архитектура", "Постройте сиамскую сеть на основе ResNet-18 (без head)."),
            ("Triplet Loss", "Реализуйте Triplet Loss с hard mining."),
            ("Обучение", "Обучите модель. Визуализируйте динамику loss."),
            ("Визуализация эмбеддингов", "Примените t-SNE или UMAP. Оцените accuracy, ROC-AUC."),
        ],
    ),
)

save_notebook(
    "lab08_multimodal.ipynb",
    make_notebook(
        "ЛР-8. Мультимодальная модель (Captioning)",
        [
            ("Загрузка данных", "Загрузите COCO Captions или Flickr8k."),
            ("Архитектура", "Реализуйте кодировщик изображений (ViT/ResNet) + декодер (Transformer/LSTM)."),
            ("Проекционный слой", "Реализуйте проекцию image features → text embedding space."),
            ("Обучение", "Обучите с cross-entropy loss (teacher forcing)."),
            ("Генерация", "Генерируйте подписи с beam search. Оцените BLEU, ROUGE."),
        ],
    ),
)

save_notebook(
    "lab09_crnn.ipynb",
    make_notebook(
        "ЛР-9. Распознавание текста CRNN",
        [
            ("Загрузка данных", "Загрузите IIIT5K или SVHN. Синтезируйте дополнительные данные."),
            ("CNN backbone", "Реализуйте VGG-like или ResNet backbone."),
            ("BiLSTM + CTC", "Реализуйте BiLSTM и CTC Loss."),
            ("Обучение", "Обучите CRNN. Фиксируйте CER, WER."),
            ("Тестирование", "Протестируйте на реальных изображениях. Проанализируйте ошибки."),
        ],
    ),
)
