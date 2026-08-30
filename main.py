import logging
from src.dataset import EDA,test
from src.utils import print_experiment_configs

from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

def perform_EDA():
    eda = EDA()
    eda.get_dataset_stats()

def main():
    print("Hello from 2026900001-assignment1!")


if __name__ == "__main__":
    perform_EDA()
    print_experiment_configs()
    test()
