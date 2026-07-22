# Методические указания преподавателям по использованию ресурсов

## Общие рекомендации

Данный раздел содержит рекомендации по подбору, подготовке и использованию информационных ресурсов для проведения лабораторных работ по дисциплине «Современные методы компьютерного зрения».

Ресурсы сгруппированы по типам в папке [resources](../../resources/README.md). Для каждого ресурса рекомендуется оформить карточку с полями: название, тип, аннотация, связанные КИМ, доступ, лицензия, дата проверки.

## Ресурсы по лабораторным работам

### Модуль 1. Фундаментальные архитектуры

#### ЛР №1. Классическое детектирование объектов по Виоле-Джонса

| Тип ресурса | Название | Назначение | Ссылка |
|---|---|---|---|
| Библиотека | OpenCV | Каскады Хаара, HOG, работа с изображениями | [opencv.org](https://opencv.org/) |
| Датасет | FDDB | Детекция лиц (оценка TPR, FPR) | [fddb] |
| Датасет | Caltech Pedestrians | Детекция пешеходов | [caltech-dataset] |
| Документация | OpenCV Cascade Classification | Руководство по обучению каскадов | [docs.opencv.org](https://docs.opencv.org/master/dc/d88/tutorial_traincascade.html) |

**Рекомендации по проведению:**
- Обратите внимание студентов на разницу между каскадами Хаара и HOG+SVM
- Покажите, как собирать положительные и отрицательные выборки
- Объясните метрики TPR, FPR, Precision-Recall на примере детекции

#### ЛР №2. Сравнительный анализ свёрточных архитектур

| Тип ресурса | Название | Назначение | Ссылка |
|---|---|---|---|
| Библиотека | PyTorch | Реализация и обучение CNN | [pytorch.org](https://pytorch.org/) |
| Библиотека | torchvision | Предобученные модели, датасеты | [pytorch.org/vision](https://pytorch.org/vision/) |
| Датасет | CIFAR-10 | Классификация изображений (10 классов) | [cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html) |
| Датасет | CSU101 | Классификация (101 класс) | [csu101] |
| Статья | Squeeze-and-Excitation Networks | SE block | Hu et al., CVPR 2018 |
| Статья | MobileNetV2 | Inverted bottleneck | Sandler et al., CVPR 2018 |
| Инструмент | torchsummary / thop | Подсчёт параметров модели | pip install torchsummary |
| Инструмент | TensorBoard / WandB | Логирование экспериментов | [wandb.ai](https://wandb.ai/) |

**Рекомендации по проведению:**
- Убедитесь, что студенты фиксируют random seed для воспроизводимости
- Обратите внимание на разницу между обучением с нуля и fine-tuning
- Рекомендуйте использовать единую стратегию аугментации и оптимизатор (AdamW + cosine annealing)

### Модуль 2. Детекция, сегментация, генерация

#### ЛР №3. Разработка и обучение детектора объектов (YOLO)

| Тип ресурса | Название | Назначение | Ссылка |
|---|---|---|---|
| Библиотека | Ultralytics YOLO | Одноэтапный детектор | [docs.ultralytics.com](https://docs.ultralytics.com/) |
| Датасет | COCO | Детекция 80 классов | [cocodataset.org](https://cocodataset.org/) |
| Датасет | Roboflow | Разметка и подготовка датасетов | [roboflow.com](https://roboflow.com/) |
| Инструмент | LabelImg / CVAT | Разметка изображений | [github.com/heartexlabs/labelImg](https://github.com/heartexlabs/labelImg) |
| Метрика | mAP | Средняя точность детекции | — |

**Рекомендации по проведению:**
- Студенты должны собрать минимум 100–200 размеченных изображений
- Обратите внимание на баланс классов
- Рекомендуйте Roboflow для аугментации данных
- Научите интерпретировать mAP@0.5 и mAP@0.5:0.95

#### ЛР №4. Семантическая сегментация (U-Net, FCN)

| Тип ресурса | Название | Назначение | Ссылка |
|---|---|---|---|
| Библиотека | segmentation_models_pytorch | Готовые реализации U-Net, FPN, DeepLab | [github.com/qubvel/segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) |
| Датасет | CamVid | Сегментация дорожных сцен | [camvid] |
| Датасет | Cityscapes | Сегментация городских сцен (50 классов) | [cityscapes-dataset.com](https://www.cityscapes-dataset.com/) |
| Функция потерь | Dice Loss | Для несбалансированных классов | — |

**Рекомендации по проведению:**
- Покажите разницу между Cross-Entropy и Dice Loss
- Объясните IoU метрику и её связь с Dice Coefficient
- Рекомендуйте предобученные бэкбоны (ResNet-34, EfficientNet)

#### ЛР №5. Генерация изображений с помощью GAN

| Тип ресурса | Название | Назначение | Ссылка |
|---|---|---|---|
| Датасет | CelebA | 200K изображений лиц | [mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) |
| Датасет | Anime Faces | Синтез аниме-лиц | [kaggle.com] |
| Библиотека | PyTorch | Реализация DCGAN | [pytorch.org/tutorials](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html) |
| Метрика | FID | Fréchet Inception Distance | [github.com/mseitzer/pytorch-fid](https://github.com/mseitzer/pytorch-fid) |

**Рекомендации по проведению:**
- Обратите внимание на стабильность обучения GAN: mode collapse, не сходимость
- Покажите label smoothing, BatchNorm, LeakyReLU как методы стабилизации
- Научите рассчитывать FID для объективной оценки

### Модуль 3. Трансформеры и метрическое обучение

#### ЛР №6. Vision Transformer (ViT)

| Тип ресурса | Название | Назначение | Ссылка |
|---|---|---|---|
| Библиотека | HuggingFace Transformers | Предобученные ViT модели | [huggingface.co/models](https://huggingface.co/models) |
| Библиотека | timm (PyTorch Image Models) | Справочник архитектур | [github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models) |
| Статья | ViT — An Image is Worth 16x16 Words | Оригинальная статья | Dosovitskiy et al., ICLR 2021 |
| Инструмент | captum | Визуализация атрибуций | [captum.ai](https://captum.ai/) |
| Датасет | CSU101 / ImageNet subset | Классификация | — |

**Рекомендации по проведению:**
- Сравните ViT с ResNet: когда ViT выигрывает, а когда проигрывает
- Покажите визуализацию карт внимания на разных слоях
- Объясните влияние размера патча на качество и скорость

#### ЛР №7. Сиамские сети (верификация)

| Тип ресурса | Название | Назначение | Ссылка |
|---|---|---|---|
| Датасет | LFW (Labeled Faces in the Wild) | Верификация лиц | [vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/) |
| Статья | FaceNet | Triplet Loss и сиамские сети | Schroff et al., CVPR 2015 |
| Метод | t-SNE / UMAP | Визуализация эмбеддингов | scikit-learn / umap-learn |
| Потеря | Triplet Loss | Функция потерь для метрического обучения | — |

**Рекомендации по проведению:**
- Покажите важность hard/soft mining при формировании триплетов
- Объясните влияние margin в Triplet Loss
- Продемонстрируйте t-SNE для визуализации эмбеддингов

### Модуль 4. Продвинутые и гибридные методы

#### ЛР №8. Мультимодальная модель (Captioning)

| Тип ресурса | Название | Назначение | Ссылка |
|---|---|---|---|
| Датасет | COCO Captions | 330K изображений с подписями | [cocodataset.org](https://cocodataset.org/#captions-2015) |
| Датасет | Flickr8k | 8K изображений с подписями | [kaggle.com] |
| Библиотека | HuggingFace Transformers | LLM (GPT-2, BERT) и ViT | [huggingface.co](https://huggingface.co/) |
| Метрика | BLEU | Точность n-грамм | nltk |
| Метрика | ROUGE-L | Воспроизведение текста | rouge-score |
| Метрика | CIDEr | Консенсусная метрика | — |

**Рекомендации по проведению:**
- Покажите архитектуру encoder-decoder и проекционный слой
- Объясните teacher forcing и beam search
- Сравните разные декодеры (LSTM vs Transformer)
- Обратите внимание на разницу между BLEU, ROUGE, CIDEr

#### ЛР №9. Распознавание текста (CRNN)

| Тип ресурса | Название | Назначение | Ссылка |
|---|---|---|---|
| Датасет | IIIT5K | Распознавание текста (3000 слов) | [cvit.iiit.ac.in] |
| Датасет | SVHN | Распознавание номеров домов | [ufldl.stanford.edu/housenumbers/](http://ufldl.stanford.edu/housenumbers/) |
| Датасет | MJSynth (MJ) | Синтезированный текст (9M) | [synthtext] |
| Метод | MSER | Детекция текстовых областей | OpenCV |
| Метод | CRAFT | Современный детектор текста | [github.com/clovaai/CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) |
| Потеря | CTC Loss | Connectionist Temporal Classification | torch.nn.CTCLoss |
| Метрика | CER / WER | Character/Word Error Rate | Levenshtein distance |

**Рекомендации по проведению:**
- Объясните архитектуру CNN + BiLSTM + CTC
- Покажите разницу между CER и WER
- Продемонстрируйте синтез данных для увеличения датасета
- Рекомендуйте тестирование на реальных фотографиях (вывески, документы)

## Общие рекомендации по проведению ЛР

### Подготовка среды
- Рекомендуется использовать Google Colab (GPU) или локальную среду с CUDA
- Подготовьте requirements.txt для каждой ЛР
- Проверьте доступность датасетов до начала работы

### Контроль самостоятельности
- Проверяйте, что студенты понимают свой код
- Используйте устную защиту для проверки самостоятельности
- При подозрении на несамостоятельное выполнение — дополнительные вопросы

### Использование LLM
- Разрешите использование ChatGPT/Copilot для пояснения и отладки
- Запретите полное решение задания через LLM
- Студент должен указать, если использовал LLM

### Обратная связь
- Комментируйте код студентов с рекомендациями
- Указывайте на типовые ошибки всей группе
- Ведите журнал частых ошибок для улучшения заданий
