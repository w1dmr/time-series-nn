import os
import shutil


def create_directory(directory):
    """Создает директорию, если она не существует."""
    try:
        os.mkdir(directory)
    except FileExistsError:
        pass


def copy_files(source_dir, target_dir, file_name):
    """Копирует указанный файл из каждой поддиректории source_dir в target_dir с новым именем."""
    list_dirs = os.listdir(source_dir)[:-2]

    for i, sub_dir in enumerate(list_dirs):
        source_path = os.path.join(source_dir, sub_dir, file_name)
        destination_path = os.path.join(target_dir, f'dataset_{i + 1}.dat')
        try:
            shutil.copy(source_path, destination_path)
            print(f"Скопирован: {destination_path}")
        except FileNotFoundError:
            print(f"Файл не найден: {source_path}")
        except PermissionError:
            print(f"Нет доступа к файлу: {source_path}")
        except Exception as e:
            print(f"Ошибка при копировании файла {source_path}: {e}")


if __name__ == "__main__":
    new_path = 'datasets'
    path = 'I_Train_set'
    file = 'AI_Q_eta_par.dat'

    # Создаем целевую директорию
    create_directory(new_path)

    # Копируем файлы
    copy_files(path, new_path, file)
