import os
import base64
import io
import hashlib
from typing import List, Dict, Any
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from tenacity import retry, stop_after_attempt, wait_exponential

from openai import OpenAI
from cragmm_search.search import UnifiedSearchPipeline
from agents.base_agent import BaseAgent


class QwenAPIAgent(BaseAgent):
    def __init__(self, search_pipeline: UnifiedSearchPipeline):
        super().__init__(search_pipeline)
        self.search_pipeline = search_pipeline
        self.client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        self.model_name = "qwen-vl-max"
        self.ocrmodel_name = "qwen-vl-ocr"
        self.system_prompt = (
            "You are a multimodal AI assistant specialized in image-based question answering. Follow these guidelines:"
            "1. Analyze both the image content and provided text context"
            "2. Provide factual, concise answers (1-2 sentences max)"
            "3.For unanswerable cases:"
            "- If image/text is unclear → 'i don't know"
            "- If beyond knowledge → 'This is beyond my current knowledge,i don't know'"
            "4. Never speculate or hallucinate details"
        )
        self.image_cache = {}

    def get_batch_size(self) -> int:
        return 4  # 根据API限制调整

    def batch_generate_response(
            self,
            queries: List[str],
            images: List[Image.Image],
            message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        with ThreadPoolExecutor(max_workers=self.get_batch_size()) as executor:
            futures = []
            for query, image, history in zip(queries, images, message_histories):
                futures.append(
                    executor.submit(
                        self._generate_single_response,
                        query,
                        image,
                        history
                    )
                )
            return [future.result() for future in futures]

    @retry(stop=stop_after_attempt(3),
           wait=wait_exponential(multiplier=1, min=2, max=10))
    def _safe_api_call(self, messages: List[Dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            timeout=15  # 设置超时
        )
        return response.choices[0].message.content.strip()

    def _generate_single_response(
            self,
            query: str,
            image: Image.Image,
            history: List[Dict[str, Any]]
    ) -> str:
        get_ocr_result = self._ocr_image(image)
        search_results = self._retrieve_knowledge(query, image, ocr_result=get_ocr_result)
        messages = self._build_messages(query, image, history, search_results)
        try:
            return self._safe_api_call(messages)
        except Exception as e:
            return self._handle_api_error(e)

    def _ocr_image(self, image: Image.Image) -> str:
        """OCR image using Qwen OCR API.

        Args:
            image: Input image

        Returns:
            OCR result as string
        """
        b64_img = self._get_image_url(image)
        prompt = "Please provide text context for the image."
        client = OpenAI(
            api_key=os.getenv("QWEN_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        completion = client.chat.completions.create(
            model="qwen-vl-ocr-latest",  # 可按需替换模型
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": b64_img,
                        },
                        {"type": "text", "text": prompt},
                    ]
                }
            ])
        return completion.choices[0].message.content

    def _handle_api_error(self, error: Exception) -> str:
        error_msg = str(error).lower()
        if "rate limit" in error_msg:
            return "Error: API rate limit exceeded"
        elif "timeout" in error_msg:
            return "Error: Request timeout"
        elif "invalid" in error_msg:
            return "Error: Invalid request"
        else:
            return f"Error: {str(error)}"

    def _retrieve_knowledge(
            self,
            query: str,
            image: Image.Image,
            ocr_result: str = None,
    ) -> Dict:
        """Retrieve relevant knowledge with enhanced filtering and sorting.

        Args:
            query: Text query for retrieval
            image: Image for visual retrieval

        Returns:
            Dict containing filtered and sorted search results with:
            - text: List of text results (score > 0.5)
            - image: List of image results (score > 0.6)
        """
        # Configuration
        TEXT_SCORE_THRESHOLD = 0.5
        IMAGE_SCORE_THRESHOLD = 0.6
        MAX_RESULTS = 3

        results = {}
        try:
            # Text retrieval with filtering and sorting
            combined_text = f"{query} {ocr_result}" if ocr_result else query
            if query and self.search_pipeline.web_search:
                raw_text_results = self.search_pipeline(combined_text, k=MAX_RESULTS * 3)  # Get more for filtering
                # Filter by score and sort descending
                results['text'] = sorted(
                    [r for r in raw_text_results if r['score'] > TEXT_SCORE_THRESHOLD],
                    key=lambda x: x['score'],
                    reverse=True
                )[:MAX_RESULTS]  # Take top N after filtering

            # Image retrieval with filtering and sorting
            if image and self.search_pipeline.image_collection:
                raw_image_results = self.search_pipeline(image, k=MAX_RESULTS * 3)
                # Filter by score and sort descending
                results['image'] = sorted(
                    [r for r in raw_image_results if r['score'] > IMAGE_SCORE_THRESHOLD],
                    key=lambda x: x['score'],
                    reverse=True
                )[:MAX_RESULTS]

            # Add quality metrics
            if results:
                results['metrics'] = {
                    'text_results': len(results.get('text', [])),
                    'image_results': len(results.get('image', [])),
                    'min_text_score': min([r['score'] for r in results.get('text', [])], default=0),
                    'min_image_score': min([r['score'] for r in results.get('image', [])], default=0)
                }

        except Exception as e:
            print(f"Error in knowledge retrieval: {e}")
            results['error'] = str(e)

        return results

    def _build_messages(
            self,
            query: str,
            image: Image.Image,
            history: List[Dict[str, Any]],
            search_results: Dict
    ) -> List[Dict[str, Any]]:
        """Build messages with enhanced context from filtered search results.

        Args:
            query: User query
            image: Input image
            history: Conversation history
            search_results: Filtered search results from _retrieve_knowledge

        Returns:
            List of message dicts for API call
        """
        messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]

        # Build high-quality context from filtered results
        context_parts = []

        # Add text results context if available and high quality
        if search_results.get('text'):
            for i, result in enumerate(search_results['text'], 1):
                context_parts.append(
                    f"[Text Reference {i} | Confidence: {result['score']:.2f}]\n"
                    f"{result['page_snippet'][:300]}"
                )

        # Add image results context if available and high quality
        if search_results.get('image'):
            for i, result in enumerate(search_results['image'], 1):
                entities = "\n".join(
                    f"- {e['entity_name']}: {e['entity_attributes']}"
                    for e in result["entities"]
                )
                context_parts.append(
                    f"[Image Reference {i} | Confidence: {result['score']:.2f}]\n"
                    f"Detected entities:\n{entities}"
                )

        # Add quality metrics if available
        if search_results.get('metrics'):
            metrics = search_results['metrics']
            context_parts.append(
                f"\n[Retrieval Quality Metrics]\n"
                f"Text results: {metrics['text_results']} (min score: {metrics['min_text_score']:.2f})\n"
                f"Image results: {metrics['image_results']} (min score: {metrics['min_image_score']:.2f})"
            )

        # Add context message if we have any relevant information
        if context_parts:
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "Relevant Context:\n" + "\n\n".join(context_parts)
                }]
            })
        # Add conversation history
        for turn in history:
            messages.append({
                "role": turn["role"],
                "content": [{"type": "text", "text": turn["content"]}]
            })

        # Build final user message with image and query
        user_content = []
        if image:
            user_content.append({
                "type": "image_url",
                "image_url": {"url": self._get_image_url(image)}
            })

        user_content.append({
            "type": "text",
            "text": query + (
                "\n\n[Instruction]\n"
                "Please consider the provided context carefully when answering. "
                "Focus on the highest confidence references when they are available."
            )
        })

        messages.append({
            "role": "user",
            "content": user_content
        })

        return messages

    def _get_image_url(self, image: Image.Image) -> str:
        # 生成图像哈希
        buffered = io.BytesIO()
        image.convert("RGB")
        image.save(buffered, format="JPEG")
        image_data = buffered.getvalue()
        image_hash = hashlib.md5(image_data).hexdigest()

        # 缓存检查
        if image_hash in self.image_cache:
            return self.image_cache[image_hash]

        # 新图像处理
        b64_img = base64.b64encode(image_data).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{b64_img}"
        self.image_cache[image_hash] = image_url

        return image_url