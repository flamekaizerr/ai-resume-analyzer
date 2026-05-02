from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
import logging

logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        self.embedding_model = None
        self.ner_pipeline = None
        self.t5_model = None
        self.t5_tokenizer = None

    def load_models(self):
        """Lazy load models on startup"""
        try:
            logger.info("Loading SentenceTransformer...")
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Loading NER pipeline...")
            self.ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
            
            logger.info("Loading T5 pipeline...")
            self.t5_tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
            self.t5_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
            self.t5_model.eval()
            logger.info("All models loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

ml_models = ModelManager()
