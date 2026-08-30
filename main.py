import logging
from src.dataset import EDA,test_dataset_preparation
from src.utils import print_experiment_configs
from src.models.positional import test_positional_encoding
from src.models.attention import test_scaled_dotproduct
from src.models.feedforward import test_ffn
from src.models.encoder import test_encoder_layer

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
    test_dataset_preparation()
    test_positional_encoding()
    test_scaled_dotproduct()
    test_ffn()
    test_encoder_layer()
