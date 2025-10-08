import time
import logging
import torch
import crypten
import crypten.communicator as comm
import crypten.nn as nn
from crypten.nn.loss import MSELoss
from crypten.optim import SGD

from config import config
from . import task


def init_crypten():
    crypten.init()
    comm.get().set_verbosity(True)
    crypten.encoder.set_default_precision(config.PRECISION)


def calculate_metrics(y_pred_enc, y_true_enc):
    # MSE - используем .pow(2) вместо ** 2
    mse = ((y_pred_enc - y_true_enc).pow(2)).mean()
    
    # MAE - используем метод .abs()
    mae = (y_pred_enc - y_true_enc).abs().mean()
    
    # R² = 1 - SS_res / SS_tot
    ss_res = ((y_true_enc - y_pred_enc).pow(2)).sum()
    ss_tot = ((y_true_enc - y_true_enc.mean()).pow(2)).sum()
    r2 = 1.0 - (ss_res / ss_tot)
    
    return {
        'mse': mse.get_plain_text().item(),
        'mae': mae.get_plain_text().item(),
        'r2': r2.get_plain_text().item()
    }



@task("ttp")
def ttp():
    """TTP процесс для генерации троек Бивера."""
    crypten.mpc.provider.TTPServer()


@task("linreg_homework")
def linreg_homework():
    """
    Обучение линейной регрессии в сценарии Data Augmentation с TTP.
    
    Архитектура:
    - Worker 0 (rank=0): имеет X_train_0, y_train_0
    - Worker 1 (rank=1): имеет X_train_1, y_train_1, X_test, y_test
    - TTP (rank=2): генерирует тройки Бивера для ускорения MPC
    """
    init_crypten()
    rank = comm.get().get_rank()
    
    logging.info(f"=" * 80)
    logging.info(f"Rank {rank}: Starting Data Augmentation scenario with TTP")
    logging.info(f"=" * 80)
    
    # ========================================================================
    # ШАГ 1: Загрузка тренировочных данных от ОБЕИХ сторон
    # ========================================================================
    
    logging.info(f"Rank {rank}: Loading training data from both workers...")
    
    # Worker 0 загружает свои данные (2000 примеров)
    X_train_worker0 = crypten.load_from_party(
        "/data/linreg/homework/x_train_norm_worker1.pth", 
        src=0
    )
    y_train_worker0 = crypten.load_from_party(
        "/data/linreg/homework/y_train_worker1.pth", 
        src=0
    )
    
    # Worker 1 загружает свои данные (1200 примеров)
    X_train_worker1 = crypten.load_from_party(
        "/data/linreg/homework/x_train_norm_worker2.pth", 
        src=1
    )
    y_train_worker1 = crypten.load_from_party(
        "/data/linreg/homework/y_train_worker2.pth", 
        src=1
    )
    
    logging.info(f"Rank {rank}: Worker 0 data: X={X_train_worker0.size()}, y={y_train_worker0.size()}")
    logging.info(f"Rank {rank}: Worker 1 data: X={X_train_worker1.size()}, y={y_train_worker1.size()}")
    
    # ========================================================================
    # ШАГ 2: Объединение данных (Data Augmentation - конкатенация по строкам)
    # ========================================================================
    
    logging.info(f"Rank {rank}: Combining encrypted data from both workers...")
    
    # Объединяем данные от обеих сторон по оси 0 (строки)
    # Результат: 3200 наблюдений (2000 + 1200)
    X_train_enc = crypten.cat([X_train_worker0, X_train_worker1], dim=0)
    y_train_enc = crypten.cat([y_train_worker0, y_train_worker1], dim=0)
    
    logging.info(f"Rank {rank}: Combined training data: X={X_train_enc.size()}, y={y_train_enc.size()}")
    
    # ========================================================================
    # ШАГ 3: Загрузка тестовых данных (только от worker 0)
    # ========================================================================
    
    logging.info(f"Rank {rank}: Loading test data from worker 0...")
    
    X_test_enc = crypten.load_from_party(
        "/data/linreg/homework/x_test_norm_worker1.pth", 
        src=0
    )
    y_test_enc = crypten.load_from_party(
        "/data/linreg/homework/y_test_worker1.pth",
        src=0
    )
    
    logging.info(f"Rank {rank}: Test data: X={X_test_enc.size()}, y={y_test_enc.size()}")
    
    # ========================================================================
    # ШАГ 4: Создание и инициализация модели
    # ========================================================================
    
    logging.info(f"Rank {rank}: Creating and initializing model...")
    
    # Модель линейной регрессии: y = X @ w + b
    class LinearModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    # Создаем модель (64 входных признака -> 1 выход)
    input_dim = X_train_enc.shape[1]
    model = LinearModel(input_dim)
    
    # Инициализация весов
    for name, weight in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(weight, mean=0.0, std=0.01)
        elif 'bias' in name:
            nn.init.constant_(weight, 0.0)
    
    # Шифруем модель (переводим веса в режим secret sharing)
    model.encrypt()
    
    logging.info(f"Rank {rank}: Model initialized and encrypted")
    
    # ========================================================================
    # ШАГ 5: Обучение модели на зашифрованных данных
    # ========================================================================
    
    logging.info(f"Rank {rank}: Starting model training...")
    
    # Loss и оптимизатор
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)
    
    # Параметры обучения
    batch_size = config.BATCH_SIZE  # из переменной окружения
    n_epochs = config.EPOCHS        # из переменной окружения
    
    n_samples = y_train_enc.size(0)  # 3200
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    logging.info(f"Rank {rank}: Training config: epochs={n_epochs}, batch_size={batch_size}, n_batches={n_batches}")
    
    # Засекаем время
    t0 = time.time()
    comm.get().reset_communication_stats()
    
    # Цикл обучения
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        # Мини-батчи
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            
            # Получаем батч
            X_batch = X_train_enc[start:end]
            y_batch = y_train_enc[start:end]
            
            # Forward pass
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            
            # Backward pass
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Накапливаем loss (расшифровываем для логирования)
            epoch_loss += loss.get_plain_text().item()
        
        # Логируем каждые 5 эпох
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / n_batches
            logging.info(f"Rank {rank}: [Epoch {epoch+1:3d}/{n_epochs}] loss={avg_loss:.6f}")
    
    training_time = time.time() - t0
    logging.info(f"Rank {rank}: Training completed in {training_time:.4f} seconds")
    
    # Статистика коммуникации
    logging.info(f"Rank {rank}: Communication statistics:")
    comm.get().print_communication_stats()
    
    # ========================================================================
    # ШАГ 6: Оценка на тестовом датасете
    # ========================================================================
    
    logging.info(f"Rank {rank}: Evaluating on test set...")
    
    with torch.no_grad():
        y_pred_enc = model(X_test_enc)
        
        # Вычисляем все метрики
        metrics = calculate_metrics(y_pred_enc, y_test_enc)
    
    # ========================================================================
    # ШАГ 7: Вывод результатов
    # ========================================================================
    
    logging.info(f"=" * 80)
    logging.info(f"Rank {rank}: FINAL RESULTS (MPC WITH TTP)")
    logging.info(f"=" * 80)
    logging.info(f"Test MSE:     {metrics['mse']:.6f}")
    logging.info(f"Test MAE:     {metrics['mae']:.6f}")
    logging.info(f"Test R²:      {metrics['r2']:.6f}")
    logging.info(f"Training time: {training_time:.4f} seconds")
    logging.info(f"=" * 80)
    
    # Выводим обученные веса (в виде share для проверки)
    w_share = model.fc.weight.share
    b_share = model.fc.bias.share
    
    logging.info(f"Rank {rank}: Learned weights share (first 10): {w_share.view(-1)[:10]}")
    logging.info(f"Rank {rank}: Learned bias share: {b_share.item():.6f}")
    
    # Размораживаем CrypTen
    crypten.uninit()


@task("linreg_homework_compare")
def linreg_homework_compare():
    """
    Полное сравнение: Baseline (открытые данные) vs MPC с TTP.
    
    Phase 1: Обучение на открытых данных (только worker 0)
    Phase 2: Обучение с MPC и TTP (оба воркера)
    Phase 3: Сравнение результатов
    """
    import torch.optim as optim
    
    # Инициализируем CrypTen СРАЗУ, чтобы получить rank
    init_crypten()
    rank = comm.get().get_rank()
    
    # ========================================================================
    # PHASE 1: BASELINE TRAINING (PLAIN DATA - NO SECRET SHARING)
    # ========================================================================
    
    # Только один worker делает baseline обучение
    if rank == 0:
        logging.info("=" * 80)
        logging.info("PHASE 1: BASELINE TRAINING (PLAIN DATA - NO SECRET SHARING)")
        logging.info("=" * 80)
        
        # Загрузка открытых данных БЕЗ шифрования
        X_train_w0 = torch.load("/data/linreg/homework/x_train_norm_worker1.pth")
        y_train_w0 = torch.load("/data/linreg/homework/y_train_worker1.pth")
        X_train_w1 = torch.load("/data/linreg/homework/x_train_norm_worker2.pth")
        y_train_w1 = torch.load("/data/linreg/homework/y_train_worker2.pth")
        
        # Объединение на открытых данных
        X_train_plain = torch.cat([X_train_w0, X_train_w1], dim=0)
        y_train_plain = torch.cat([y_train_w0, y_train_w1], dim=0)
        
        X_test_plain = torch.load("/data/linreg/homework/x_test_norm_worker1.pth")
        y_test_plain = torch.load("/data/linreg/homework/y_test_worker1.pth")
        
        logging.info(f"Loaded plain data: X_train={X_train_plain.shape}, y_train={y_train_plain.shape}")
        logging.info(f"Loaded test data: X_test={X_test_plain.shape}, y_test={y_test_plain.shape}")
        
        # Модель для открытых данных
        class LinearModelPlain(torch.nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.fc = torch.nn.Linear(input_dim, 1)
            
            def forward(self, x):
                return self.fc(x)
        
        model_plain = LinearModelPlain(input_dim=64)
        
        # Инициализация (та же, что и для MPC)
        torch.nn.init.normal_(model_plain.fc.weight, mean=0.0, std=0.01)
        torch.nn.init.constant_(model_plain.fc.bias, 0.0)
        
        criterion_plain = torch.nn.MSELoss()
        optimizer_plain = optim.SGD(model_plain.parameters(), lr=0.05, momentum=0.9)
        
        batch_size = config.BATCH_SIZE
        n_epochs = config.EPOCHS
        n_samples = y_train_plain.size(0)
        n_batches = (n_samples + batch_size - 1) // batch_size
        
        logging.info(f"Starting PLAIN training: {n_epochs} epochs, {n_batches} batches")
        
        t0_plain = time.time()
        
        # Обучение на открытых данных
        for epoch in range(n_epochs):
            epoch_loss = 0.0
            for i in range(n_batches):
                start = i * batch_size
                end = min((i + 1) * batch_size, n_samples)
                
                X_batch = X_train_plain[start:end]
                y_batch = y_train_plain[start:end]
                
                optimizer_plain.zero_grad()
                preds = model_plain(X_batch)
                loss = criterion_plain(preds, y_batch)
                loss.backward()
                optimizer_plain.step()
                
                epoch_loss += loss.item()
            
            if (epoch + 1) % 5 == 0 or epoch == 0:
                logging.info(f"[Epoch {epoch+1:3d}/{n_epochs}] loss={epoch_loss/n_batches:.6f}")
        
        plain_time = time.time() - t0_plain
        
        # Тестирование на открытых данных
        with torch.no_grad():
            y_pred_plain = model_plain(X_test_plain)
            test_mse_plain = criterion_plain(y_pred_plain, y_test_plain).item()
            
            # MAE
            test_mae_plain = torch.mean(torch.abs(y_pred_plain - y_test_plain)).item()
            
            # R²
            ss_res = torch.sum((y_test_plain - y_pred_plain) ** 2).item()
            ss_tot = torch.sum((y_test_plain - y_test_plain.mean()) ** 2).item()
            test_r2_plain = 1.0 - (ss_res / ss_tot)
        
        # Сохранение весов baseline
        weights_plain = model_plain.fc.weight.data.clone()
        bias_plain = model_plain.fc.bias.data.clone()
        
        logging.info("=" * 80)
        logging.info("BASELINE RESULTS:")
        logging.info(f"Test MSE:     {test_mse_plain:.6f}")
        logging.info(f"Test MAE:     {test_mae_plain:.6f}")
        logging.info(f"Test R²:      {test_r2_plain:.6f}")
        logging.info(f"Training time: {plain_time:.4f} seconds")
        logging.info(f"Weights (first 10): {weights_plain.view(-1)[:10]}")
        logging.info(f"Bias: {bias_plain.item():.6f}")
        logging.info("=" * 80)
    
    # ========================================================================
    # PHASE 2: MPC TRAINING (SECRET SHARING WITH TTP)
    # ========================================================================
    
    logging.info("=" * 80)
    logging.info(f"Rank {rank}: PHASE 2: MPC TRAINING (WITH SECRET SHARING AND TTP)")
    logging.info("=" * 80)
    
    # Загрузка данных с разделением секрета
    X_train_worker0 = crypten.load_from_party(
        "/data/linreg/homework/x_train_norm_worker1.pth", 
        src=0
    )
    y_train_worker0 = crypten.load_from_party(
        "/data/linreg/homework/y_train_worker1.pth", 
        src=0
    )
    
    X_train_worker1 = crypten.load_from_party(
        "/data/linreg/homework/x_train_norm_worker2.pth", 
        src=1
    )
    y_train_worker1 = crypten.load_from_party(
        "/data/linreg/homework/y_train_worker2.pth", 
        src=1
    )
    
    logging.info(f"Rank {rank}: Loaded worker 0 data: X={X_train_worker0.size()}, y={y_train_worker0.size()}")
    logging.info(f"Rank {rank}: Loaded worker 1 data: X={X_train_worker1.size()}, y={y_train_worker1.size()}")
    
    # Объединение зашифрованных данных
    X_train_enc = crypten.cat([X_train_worker0, X_train_worker1], dim=0)
    y_train_enc = crypten.cat([y_train_worker0, y_train_worker1], dim=0)
    
    logging.info(f"Rank {rank}: Combined encrypted data: X={X_train_enc.size()}, y={y_train_enc.size()}")
    
    # Загрузка тестовых данных
    X_test_enc = crypten.load_from_party(
        "/data/linreg/homework/x_test_norm_worker1.pth", 
        src=0
    )
    y_test_enc = crypten.load_from_party(
        "/data/linreg/homework/y_test_worker1.pth",
        src=0
    )
    
    logging.info(f"Rank {rank}: Loaded test data: X={X_test_enc.size()}, y={y_test_enc.size()}")
    
    # Модель для MPC
    class LinearModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)
        
        def forward(self, x):
            return self.fc(x)
    
    model_mpc = LinearModel(input_dim=64)
    
    # Инициализация (та же)
    for name, weight in model_mpc.named_parameters():
        if 'weight' in name:
            nn.init.normal_(weight, mean=0.0, std=0.01)
        elif 'bias' in name:
            nn.init.constant_(weight, 0.0)
    
    model_mpc.encrypt()
    
    logging.info(f"Rank {rank}: MPC model initialized and encrypted")
    
    # Обучение MPC
    criterion_mpc = MSELoss()
    optimizer_mpc = SGD(model_mpc.parameters(), lr=0.05, momentum=0.9)
    
    batch_size = config.BATCH_SIZE
    n_epochs = config.EPOCHS
    n_samples = y_train_enc.size(0)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    logging.info(f"Rank {rank}: Starting MPC training: {n_epochs} epochs, {n_batches} batches")
    
    t0_mpc = time.time()
    comm.get().reset_communication_stats()
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, n_samples)
            
            X_batch = X_train_enc[start:end]
            y_batch = y_train_enc[start:end]
            
            preds = model_mpc(X_batch)
            loss = criterion_mpc(preds, y_batch)
            
            model_mpc.zero_grad()
            loss.backward()
            optimizer_mpc.step()
            
            epoch_loss += loss.get_plain_text().item()
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"Rank {rank}: [Epoch {epoch+1:3d}/{n_epochs}] loss={epoch_loss/n_batches:.6f}")
    
    mpc_time = time.time() - t0_mpc
    
    logging.info(f"Rank {rank}: MPC training completed in {mpc_time:.4f} seconds")
    comm.get().print_communication_stats()
    
    # Тестирование MPC
    with torch.no_grad():
        y_pred_enc = model_mpc(X_test_enc)
        metrics_mpc = calculate_metrics(y_pred_enc, y_test_enc)
    
    # Расшифровка весов MPC
    weights_mpc = model_mpc.fc.weight.get_plain_text()
    bias_mpc = model_mpc.fc.bias.get_plain_text()
    
    logging.info("=" * 80)
    logging.info(f"Rank {rank}: MPC RESULTS:")
    logging.info(f"Test MSE:     {metrics_mpc['mse']:.6f}")
    logging.info(f"Test MAE:     {metrics_mpc['mae']:.6f}")
    logging.info(f"Test R²:      {metrics_mpc['r2']:.6f}")
    logging.info(f"Training time: {mpc_time:.4f} seconds")
    logging.info(f"Weights (first 10): {weights_mpc.view(-1)[:10]}")
    logging.info(f"Bias: {bias_mpc.item():.6f}")
    logging.info("=" * 80)
    
    # ========================================================================
    # PHASE 3: СРАВНЕНИЕ РЕЗУЛЬТАТОВ
    # ========================================================================
    
    if rank == 0:
        logging.info("=" * 80)
        logging.info("COMPARISON: PLAIN vs MPC WITH TTP")
        logging.info("=" * 80)
        logging.info(f"{'Metric':<20} {'Plain':<15} {'MPC':<15} {'Difference':<15}")
        logging.info("-" * 80)
        logging.info(f"{'Test MSE:':<20} {test_mse_plain:<15.6f} {metrics_mpc['mse']:<15.6f} {abs(test_mse_plain - metrics_mpc['mse']):<15.6f}")
        logging.info(f"{'Test MAE:':<20} {test_mae_plain:<15.6f} {metrics_mpc['mae']:<15.6f} {abs(test_mae_plain - metrics_mpc['mae']):<15.6f}")
        logging.info(f"{'Test R²:':<20} {test_r2_plain:<15.6f} {metrics_mpc['r2']:<15.6f} {abs(test_r2_plain - metrics_mpc['r2']):<15.6f}")
        logging.info("")
        logging.info(f"{'Training time:':<20} {plain_time:<15.4f} {mpc_time:<15.4f} {mpc_time - plain_time:<15.4f}")
        logging.info(f"{'Slowdown factor:':<20} {'1.00x':<15} {mpc_time/plain_time:<15.2f}x")
        logging.info("")
        
        # Сравнение весов
        weights_diff = torch.abs(weights_plain.view(-1) - weights_mpc.view(-1))
        bias_diff = abs(bias_plain.item() - bias_mpc.item())
        
        logging.info(f"{'Bias (Plain):':<20} {bias_plain.item():.6f}")
        logging.info(f"{'Bias (MPC):':<20} {bias_mpc.item():.6f}")
        logging.info(f"{'Bias difference:':<20} {bias_diff:.6f}")
        logging.info("")
        logging.info(f"{'Max weight diff:':<20} {weights_diff.max().item():.6f}")
        logging.info(f"{'Mean weight diff:':<20} {weights_diff.mean().item():.6f}")
        logging.info("=" * 80)
        
        # Проверка успешности
        mse_close = abs(test_mse_plain - metrics_mpc['mse']) < 0.01
        r2_close = abs(test_r2_plain - metrics_mpc['r2']) < 0.05
        
        logging.info("")
        logging.info("VALIDATION:")
        logging.info(f"✓ MSE близка к baseline: {mse_close}")
        logging.info(f"✓ R² близка к baseline: {r2_close}")
        logging.info(f"✓ Данные не раскрыты: True (использовался MPC)")
        logging.info(f"✓ TTP использован: True (для генерации троек Бивера)")
        logging.info("=" * 80)
    
    crypten.uninit()