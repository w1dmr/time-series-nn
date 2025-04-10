import os

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


class TimeSeriesDataset(data.Dataset):
    def __init__(self, path, transform=None, Q_window=True, Q_lags=4,
                 n_window=False, n_lags=4, filled=False,
                 split='train', train_ratio=0.8):
        """
        Инициализация датасета
        :param path: Путь к директории с файлами временных рядов
        :param transform: Трансформация для данных. Если указано, применяется к данным.
        :param Q_window: Флаг для создания лаговых признаков для 'water_flow'
        :param Q_lags: Количество лагов для 'water_flow'
        :param n_window: Флаг для создания лаговых признаков для уровней воды
        :param n_lags: Количество лагов для уровней воды
        :param filled: Флаг для заполнения пропущенных значений
        :param split: Определяет разделение на обучающую и тестовую выборки
        :param train_ratio: Пропорция данных для обучающей выборки
        """
        self.path = path
        self.transform = transform

        self.Q_window = Q_window
        self.Q_lags = Q_lags
        self.n_window = n_window
        self.n_lags = n_lags
        self.filled = filled
        self.split = split
        self.train_ratio = train_ratio
        self.seed = 42
        self.list_files = None

        # Список файлов в указанной директории
        all_files = os.listdir(self.path)
        np.random.seed(self.seed)
        np.random.shuffle(all_files)
        split_idx = int(len(all_files) * self.train_ratio)

        if self.split == 'train':
            self.list_files = all_files[:split_idx]
        elif self.split == 'test':
            self.list_files = all_files[split_idx:]
        else:
            self.list_files = all_files

        # Загрузка и обработка всех файлов
        self.X = self._load_and_process_files()

    def _load_and_process_files(self):
        """
        Загрузка и предварительная обработка данных из всех файлов в директории
        :return: Обработанный тензор данных, содержащий временные ряды после применения всех лагов
        """
        time_series = []
        for file_name in self.list_files:
            dataset = np.loadtxt(os.path.join(self.path, file_name), dtype=np.float32, delimiter=' ')[:, 5:]
            processed_data = self._get_lagged_data(dataset)
            time_series.append(processed_data)
        combined_data = torch.stack(time_series)
        return combined_data.view(-1, combined_data.size(dim=2))

    def _get_lagged_data(self, ts):
        """
        Обработка временного ряда для создания лаговых признаков
        :param ts: Временной ряд в виде numpy массива
        :return: Преобразованный тензор с лаговыми признаками для модели
        """
        base_columns = ['water_flow', 'water_level_1', 'water_level_2', 'water_level_3',
                        'roughness_coeff', 'alpha_coeff', 'gamma_coeff', 'theta_boundary']
        dataset = pd.DataFrame(ts, columns=base_columns)

        # Добавляем лаговые признаки для water_flow
        if self.Q_window:
            fill_value = dataset['water_flow'].iloc[0] if self.filled else None
            self._add_lag_features(dataset, 'water_flow', self.Q_lags, fill_value)

        # Добавляем лаговые признаки для уровней воды
        if self.n_window:
            for col, fill_value in zip(['water_level_1', 'water_level_2', 'water_level_3'],
                                       [dataset[col].iloc[0] if self.filled else None for col in
                                        ['water_level_1', 'water_level_2', 'water_level_3']]):
                self._add_lag_features(dataset, col, self.n_lags, fill_value)

        # Удаляем строки с пропущенными значениями, если fill_value не задан
        if not self.filled:
            dataset.dropna(inplace=True)

        # Упорядочиваем столбцы в определенном порядке
        dataset = self._reorder_columns(dataset, base_columns)

        return torch.tensor(dataset.to_numpy(), dtype=torch.float32)

    @staticmethod
    def _add_lag_features(dataset, column, lags, fill_value):
        """
        Добавляет лаговые признаки для указанной колонки
        :param dataset: DataFrame с данными
        :param column: Название колонки, для которой создаются лаги
        :param lags: Количество лагов
        :param fill_value: Значение для заполнения пропусков (если указано)
        """
        for lag in range(1, lags + 1):
            dataset[f'{column}_lag_{lag}'] = dataset[column].shift(lag, fill_value=fill_value)

    def _reorder_columns(self, dataset, base_columns):
        """
        Упорядочивает столбцы в соответствии с заданной логикой
        :param dataset: DataFrame с данными
        :param base_columns: Базовые столбцы, которые должны быть на своих местах
        :return: DataFrame с упорядоченными столбцами
        """
        # Формируем список упорядоченных колонок
        ordered_columns = [base_columns[0]]  # Добавляем water_flow

        # Добавляем лаги для water_flow, если они включены
        if self.Q_window:
            ordered_columns += [f'water_flow_lag_{lag}' for lag in range(1, self.Q_lags + 1)]

        # Добавляем water_level_1 и лаги, если n_window включен
        ordered_columns.append(base_columns[1])
        if self.n_window:
            ordered_columns += [f'water_level_1_lag_{lag}' for lag in range(1, self.n_lags + 1)]

        ordered_columns.append(base_columns[2])
        if self.n_window:
            ordered_columns += [f'water_level_2_lag_{lag}' for lag in range(1, self.n_lags + 1)]

        # Добавляем water_level_3 и лаги, если n_window включен
        ordered_columns.append(base_columns[3])
        if self.n_window:
            ordered_columns += [f'water_level_3_lag_{lag}' for lag in range(1, self.n_lags + 1)]

        # Добавляем оставшиеся базовые столбцы (без лагов)
        ordered_columns += base_columns[4:]

        # Возвращаем DataFrame с упорядоченными столбцами
        return dataset[ordered_columns]

    def __getitem__(self, index):
        """
        Возвращает элемент по указанному индексу
        :param index: Индекс элемента для извлечения
        :return: Кортеж, содержащий входные данные (X) и целевые метки (y)
        """
        X, y = self.X[index, :-4], self.X[index, -4:]
        return X, y

    def __len__(self):
        """
        Возвращает количество элементов в датасете
        :return: Длина датасета
        """
        return len(self.X)
