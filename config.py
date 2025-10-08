"""Configuration for CrypTen Linear Regression"""
import os


class Config:
    """Конфигурация из переменных окружения."""
    
    # Precision для CrypTen encoder
    PRECISION = int(os.getenv('PRECISION', 16))
    
    # Параметры обучения
    EPOCHS = int(os.getenv('EPOCHS', 30))
    BATCH_SIZE = int(os.getenv('BATCH_SIZE', 512))
    
    # Пути к данным (для старых задач)
    X_TRAIN_PATH = os.getenv('X_TRAIN_PATH', '/data/linreg/x_train.pth')
    Y_TRAIN_PATH = os.getenv('Y_TRAIN_PATH', '/data/linreg/y_train.pth')
    X_TEST_PATH = os.getenv('X_TEST_PATH', '/data/linreg/x_test.pth')
    Y_TEST_PATH = os.getenv('Y_TEST_PATH', '/data/linreg/y_test.pth')
    
    # Логирование
    LOGLEVEL = os.getenv('LOGLEVEL', 'INFO')


config = Config()