"Пункт 1 - ResNet-18 для классификации изображений"

from torch import Tensor, nn


class ResidualBlock(nn.Module):
    "Остаточный блок с двумя свёртками 3×3"

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels) # BatchNorm вычисляет одно среднее и одно стандартное отклонение для канала. потом: Значение - Среднее по всем картинкам и деленное на стандартное отклонение. То есть центрирует около нуля, нормализует
        # bn1 нормализует карты после первой свёртки 3×3
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)  # bn2 нормализует карты после conv2. Эта свёртка 3 на 3 уже обрабатывает соседние позиции, поэтому значения и их разброс могут измениться
        self.relu = nn.ReLU()

        # Проекция 1×1 нужна при изменении числа каналов или размера карты
        self.downsample = nn.Identity()
        if in_channels != out_channels or stride != 1:  # выбирает, как провести вход x по обходной ветви остаточного блока
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + residual)


class ResNet18(nn.Module):
    "Преобразует батч N × C × H × W в N × num_classes логитов"

    def __init__(self, num_classes: int, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # В каждой группе по два остаточных блока
        self.layer1 = self._make_stage(
            in_channels=64,
            out_channels=64,
            stride=1,
        )
        self.layer2 = self._make_stage(
            in_channels=64,
            out_channels=128,
            stride=2,
        )
        self.layer3 = self._make_stage(
            in_channels=128,
            out_channels=256,
            stride=2,
        )
        self.layer4 = self._make_stage(
            in_channels=256,
            out_channels=512,
            stride=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    @staticmethod # @staticmethod говорит Python, что при вызове метода не нужно автоматически передавать self первым аргументом. Поэтому _make_stage принимает ровно три значения: in_channels, out_channels и stride
    def _make_stage(in_channels: int, out_channels: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)
