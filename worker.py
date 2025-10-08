import sys
import logging
from tasks import REGISTRY


def setup_logging():
    """Настройка логирования."""
    import os
    loglevel = os.getenv('LOGLEVEL', 'INFO').upper()
    
    logging.basicConfig(
        level=getattr(logging, loglevel),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Главная функция для запуска задач."""
    setup_logging()
    
    if len(sys.argv) < 3 or sys.argv[1] != 'call':
        print("Usage: python -m worker call <module:task_name>")
        print("\nAvailable tasks:")
        for task_name in REGISTRY.keys():
            print(f"  - {task_name}")
        sys.exit(1)
    
    task_path = sys.argv[2]
    
    if ':' in task_path:
        module_path, task_name = task_path.split(':')
    else:
        task_name = task_path
    
    if task_name not in REGISTRY:
        print(f"Error: Task '{task_name}' not found in registry.")
        print("\nAvailable tasks:")
        for name in REGISTRY.keys():
            print(f"  - {name}")
        sys.exit(1)
    
    task_fn = REGISTRY[task_name]
    
    logging.info(f"Starting task: {task_name}")
    
    try:
        task_fn()
        logging.info(f"Task '{task_name}' completed successfully")
    except Exception as e:
        logging.error(f"Task '{task_name}' failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()