

from crag_batch_iterator import CRAGTurnBatchIterator
from cragmm_search.search import UnifiedSearchPipeline
from utils import display_results, ensure_crag_cache_dir_is_configured
from tokenizers import Tokenizer

search_api_text_model_name = "BAAI/bge-large-en-v1.5"
search_api_image_model_name = "openai/clip-vit-large-patch14-336"
search_api_web_hf_dataset_id = "crag-mm-2025/web-search-index-validation"
search_api_image_hf_dataset_id = "crag-mm-2025/image-search-index-validation"

search_pipeline = UnifiedSearchPipeline(
        text_model_name=search_api_text_model_name,
        image_model_name=search_api_image_model_name,
        web_hf_dataset_id=search_api_web_hf_dataset_id,
        image_hf_dataset_id=search_api_image_hf_dataset_id,
  )

results = search_pipeline("What is the capital of France?",k=3)

print(results)