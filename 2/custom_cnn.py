"Пункт 0. Собственная CNN для классификации изображений"

from collections.abc import Callable

from torch import Tensor, nn


class CustomCNN(nn.Module):
    """
    Преобразует батч изображений N × C × H × W в N × num_classes логитов.

    N — число изображений в батче
    C — число каналов (in_channels)

    H — высота
    W — ширина изображения в пикселях

    num_classes — число классов, между которыми выбирает модель

    логит — оценка класса до softmax. Чем выше логит, тем выше вероятность класса после softmax. CrossEntropyLoss принимает логиты.
    """

    def __init__(
        self,
        num_classes: int, # - число классов, между которыми выбирает модель
        in_channels: int = 3, # - число каналов
        *,
        block1_filters: int = 32,  # - число фильтров первой свёртки
        block2_filters: int = 64,  # - число фильтров второй свёртки
        kernel_size: int = 3,  # - размер стороны квадратного ядра
        stride: int = 1,  # - шаг ядра в пикселях
        padding: int = 1,  # padding задаёт число нулевых пикселей с каждой стороны входа свёртки.
        activation: Callable[[], nn.Module] = nn.ReLU,
        pool_kernel_size: int = 2,  # - размер окна максимального пулинга
        pool_stride: int = 2,  # - шаг окна максимального пулинга
    ):
        super().__init__()

       
        # Ядро 3×3 с шагом 1 и padding=1 сохраняет высоту и ширину.
        # activation — класс или функция, создающая отдельный слой для каждого блока.

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, # если изображение трехканальное, то матрица свертки получается 3 * kernel_size ^ 2. и суммируем все что попадет в область трехмерного фильтра
                out_channels=block1_filters, # 32 фильтра, то есть 32 раза применяем свертки разные и получаем 32 карты признаков. Потом складываем этим карты признаков в один тензор
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            activation(),
        )

  
        self.max_pool = nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride) # MaxPool берёт каждую карту отдельно, делит её на квадраты 2 на 2 и из каждого квадрата оставляет самое большое число. Это уменьшает число признаков вдвое. Это оплезно если разрешение у картинки высокое и признаков становится тоже очень много
        
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=block1_filters, # число фильтров на первом слое
                out_channels=block2_filters, # число фильтров на втором слое
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            activation(),
        )

        # Глобальный пулинг усредняет каждую карту признаков до одного числа.
        self.avg_pool = nn.AdaptiveAvgPool2d(1) # nn.AdaptiveAvgPool2d(1) берёт среднее по всей площади каждой карты. Параметр 1 означает, что выход каждой карты имеет размер 1 на 1
        self.classifier = nn.Linear(block2_filters, num_classes) # создаёт последний слой сети, который по извлечённым признакам вычисляет оценку каждого класса

    def forward(self, x: Tensor) -> Tensor:
       
        x = self.block1(x)
        x = self.max_pool(x)
        x = self.block2(x)
        x = self.avg_pool(x)
        x = x.flatten(1)  # Сохраняем ось батча: N×64×1×1 → N×64.
        return self.classifier(x)
