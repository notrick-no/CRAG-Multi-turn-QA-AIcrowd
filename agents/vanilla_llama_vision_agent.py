from typing import Dict, List, Any
import os
import numpy as np
from collections import defaultdict

import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from agents.base_agent import BaseAgent
from cragmm_search.search import UnifiedSearchPipeline

import vllm
#你好
# Configuration constants
AICROWD_SUBMISSION_BATCH_SIZE = 8

# GPU utilization settings
# Change VLLM_TENSOR_PARALLEL_SIZE during local runs based on your available GPUs
# For example, if you have 2 GPUs on the server, set VLLM_TENSOR_PARALLEL_SIZE=2.
# You may need to uncomment the following line to perform local evaluation with VLLM_TENSOR_PARALLEL_SIZE>1.
# os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'

#### Please ensure that when you submit, VLLM_TENSOR_PARALLEL_SIZE=1.
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.85

# These are model specific parameters to get the model to run on a single NVIDIA L40s GPU
MAX_MODEL_LEN = 8192
MAX_NUM_SEQS = 2

MAX_GENERATION_TOKENS = 75

# RRF (Reciprocal Rank Fusion) parameters
RRF_K = 60  # RRF parameter for rank fusion
TEXT_SEARCH_K = 20  # Number of text search results before RRF
IMAGE_SEARCH_K = 8  # Number of image search results
RRF_WEIGHT = 0.7  # Weight for RRF score vs original similarity score
MIN_SIMILARITY_THRESHOLD = 0.3  # Minimum similarity threshold for filtering

# System prompt for vision-language tasks
VISION_SYSTEM_PROMPT = (
    "You are an expert visual assistant that provides accurate, detailed, and helpful answers about images. "
    "Your role is to analyze the visual content and answer questions truthfully and completely.\n\n"
    "Guidelines:\n"
    "• Provide comprehensive answers that address all aspects of the question\n"
    "• Include specific details about objects, people, actions, relationships, and context\n"
    "• If asked about ownership, relationships, or specific details, provide that information\n"
    "• Be precise and descriptive in your observations\n"
    "• Keep responses concise but informative\n"
    "• If you cannot determine something from the image, clearly state  'I don't know'\n"
    "• Maintain a helpful and professional tone\n\n"
    "Focus on being thorough and accurate in your visual analysis."
)

# Confidence assessment parameters
CONFIDENCE_THRESHOLD = 0.6  # Lowered threshold for better balance
HALLUCINATION_PENALTY = 0.3  # Penalty for potential hallucinations
UNCERTAINTY_PENALTY = 0.2   # Penalty for uncertainty indicators
DETAIL_BONUS = 0.1          # Bonus for specific details
COMPLETENESS_BONUS = 0.15   # Bonus for complete answers

class LlamaVisionModel(BaseAgent):
    """
    LlamaVisionModel is an implementation of BaseAgent using Meta's Llama 3.2 Vision models.

    This agent processes image-based queries using the Llama-3.2-11B-Vision-Instruct model
    and generates responses based on the visual content. It leverages vLLM for efficient,
    batched inference and supports multi-turn conversations.

    The model handles formatting of prompts and processes both single-turn and multi-turn
    conversations in a standardized way.

    Attributes:
        search_pipeline (UnifiedSearchPipeline): Pipeline for searching relevant information.
            Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
        model_name (str): Name of the Hugging Face model to use.
        max_gen_len (int): Maximum generation length for responses.
        llm (vllm.LLM): The vLLM model instance for inference.
        tokenizer: The tokenizer associated with the model.
    """

    def __init__(
            self, search_pipeline: UnifiedSearchPipeline, model_name="meta-llama/Llama-3.2-11B-Vision-Instruct",
            max_gen_len=64
    ):
        """
        Initialize the agent

        Args:
            search_pipeline (UnifiedSearchPipeline): A pipeline for searching web and image content.
                Note: The web-search will be disabled in case of Task 1 (Single-source Augmentation) - so only image-search can be used in that case.
            model_name (str): Hugging Face model name to use for vision-language processing.
            max_gen_len (int): Maximum generation length for model outputs.
        """
        super().__init__(search_pipeline)
        self.model_name = model_name
        self.max_gen_len = max_gen_len

        self.initialize_models()

    def initialize_models(self):
        """
        Initialize the HuggingFace model and tokenizer directly to avoid vLLM compatibility issues.

        This configures the model for vision-language tasks with optimized
        GPU memory usage. This is the only approach that works on older GPUs like NVIDIA T1000.
        """
        print("Loading model with HuggingFace transformers...")

        # Initialize the model and processor with HuggingFace transformers
        self.model = MllamaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        self.processor = AutoProcessor.from_pretrained(self.model_name)

        print("Loaded models")

    def _apply_rrf_reranking(self, search_results: List[Dict], query: str) -> List[Dict]:
        """
        Apply Reciprocal Rank Fusion (RRF) to rerank search results.
        
        Args:
            search_results (List[Dict]): List of search results with scores
            query (str): Original query for keyword extraction
            
        Returns:
            List[Dict]: Reranked search results
        """
        if not search_results:
            return []
        
        # Extract keywords from query for keyword-based ranking
        query_keywords = [word.lower() for word in query.split() if len(word) > 2]
        
        # Create two ranking lists: similarity-based and keyword-based
        similarity_rankings = []
        keyword_rankings = []
        
        for i, result in enumerate(search_results):
            # Similarity-based ranking (original order)
            similarity_rankings.append((i, result))
            
            # Keyword-based ranking
            snippet = result.get('page_snippet', '').lower()
            keyword_score = sum(1 for keyword in query_keywords if keyword in snippet)
            keyword_rankings.append((i, result, keyword_score))
        
        # Sort keyword rankings by keyword score (descending)
        keyword_rankings.sort(key=lambda x: x[2], reverse=True)
        
        # Apply RRF fusion
        rrf_scores = defaultdict(float)
        
        # Add similarity ranking scores
        for rank, (idx, result) in enumerate(similarity_rankings):
            rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)
        
        # Add keyword ranking scores
        for rank, (idx, result, _) in enumerate(keyword_rankings):
            rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)
        
        # Create reranked results
        reranked_results = []
        for idx, result in enumerate(search_results):
            rrf_score = rrf_scores[idx]
            original_score = result.get('score', 0.0)
            
            # Combine RRF score with original similarity score
            combined_score = RRF_WEIGHT * rrf_score + (1 - RRF_WEIGHT) * original_score
            
            # Create new result with combined score
            new_result = result.copy()
            new_result['rrf_score'] = rrf_score
            new_result['combined_score'] = combined_score
            reranked_results.append(new_result)
        
        # Sort by combined score
        reranked_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return reranked_results

    def _retrieve_enhanced_knowledge(self, query: str, image: Image.Image) -> Dict[str, Any]:
        """
        Retrieve knowledge using enhanced search with RRF reranking.
        
        Args:
            query (str): User query
            image (Image.Image): Input image
            
        Returns:
            Dict[str, Any]: Enhanced knowledge with text and image information
        """
        knowledge = {
            "text_context": [],
            "image_context": [],
            "confidence": 0.0
        }
        
        # Text search with RRF reranking
        if query and self.search_pipeline.web_search:
            try:
                # Get initial text search results
                text_results = self.search_pipeline(query, k=TEXT_SEARCH_K)
                
                if text_results:
                    # Apply RRF reranking
                    reranked_text_results = self._apply_rrf_reranking(text_results, query)
                    
                    # Filter by similarity threshold and take top results
                    filtered_results = []
                    for result in reranked_text_results:
                        combined_score = result.get('combined_score', 0.0)
                        if combined_score > MIN_SIMILARITY_THRESHOLD:
                            filtered_results.append(result)
                    
                    # Take top results after filtering
                    top_results = filtered_results[:5]  # Take top 5 after RRF
                    
                    for i, result in enumerate(top_results):
                        snippet = result.get('page_snippet', '')[:200]
                        source = result.get('source', 'Unknown')
                        combined_score = result.get('combined_score', 0.0)
                        
                        knowledge["text_context"].append({
                            "content": snippet,
                            "source": source,
                            "score": combined_score,
                            "index": i + 1
                        })
                        
                        # Update overall confidence
                        knowledge["confidence"] = max(knowledge["confidence"], combined_score)
                        
            except Exception as e:
                print(f"Enhanced text search error: {e}")
        
        # Image search
        if image and self.search_pipeline.image_collection:
            try:
                image_results = self.search_pipeline(image, k=IMAGE_SEARCH_K)
                
                for i, result in enumerate(image_results):
                    # Convert distance to similarity score
                    similarity_score = max(0, 1 - (result.get('dist', 2.0) / 2.0))
                    
                    if similarity_score > MIN_SIMILARITY_THRESHOLD:
                        entities = result.get("entities", [])
                        entity_names = [e.get('entity_name', '') for e in entities[:3] if e.get('entity_name')]
                        
                        knowledge["image_context"].append({
                            "entities": entity_names,
                            "score": similarity_score,
                            "index": i + 1
                        })
                        
                        # Update overall confidence
                        knowledge["confidence"] = max(knowledge["confidence"], similarity_score)
                        
            except Exception as e:
                print(f"Image search error: {e}")
        
        return knowledge

    def _format_enhanced_knowledge(self, knowledge: Dict[str, Any]) -> str:
        """
        Format enhanced knowledge for prompt construction.
        
        Args:
            knowledge (Dict[str, Any]): Enhanced knowledge dictionary
            
        Returns:
            str: Formatted knowledge text
        """
        context_parts = []
        
        # Format text context
        if knowledge["text_context"]:
            context_parts.append("## Web Search Results (RRF Enhanced):")
            for item in knowledge["text_context"]:
                context_parts.append(
                    f"[Ref {item['index']} | Score: {item['score']:.3f} | Source: {item['source']}]\n"
                    f"{item['content']}"
                )
        
        # Format image context
        if knowledge["image_context"]:
            context_parts.append("## Image Knowledge Graph:")
            for item in knowledge["image_context"]:
                entities_str = ", ".join(item["entities"])
                context_parts.append(
                    f"[Image {item['index']} | Score: {item['score']:.3f}]\n"
                    f"Detected entities: {entities_str}"
                )
        
        # Add confidence information
        if context_parts:
            context_parts.append(f"\nOverall Confidence: {knowledge['confidence']:.3f}")
        
        return "\n\n".join(context_parts) if context_parts else "No relevant information found."

    def _detect_hallucinations(self, response: str, query: str) -> float:
        """
        Detect potential hallucinations in the response.
        
        Args:
            response (str): The generated response
            query (str): The original query
            
        Returns:
            float: Hallucination penalty score (0.0 to 1.0)
        """
        response_lower = response.lower()
        query_lower = query.lower()
        
        # High-confidence hallucination indicators
        hallucination_indicators = [
            "definitely", "certainly", "absolutely", "without a doubt",
            "clearly shows", "obviously", "evidently", "undoubtedly",
            "i can see that", "the image shows", "it's clear that"
        ]
        
        # Check for overconfident statements about unclear things
        penalty = 0.0
        
        # Penalize overconfident statements
        for indicator in hallucination_indicators:
            if indicator in response_lower:
                penalty += 0.1
        
        # Check for contradictions with uncertainty
        uncertainty_words = ["i don't know", "cannot see", "not visible", "unclear", "hard to tell"]
        has_uncertainty = any(word in response_lower for word in uncertainty_words)
        
        # If response has both uncertainty and overconfidence, it's suspicious
        if has_uncertainty and penalty > 0:
            penalty += 0.2
        
        # Check for specific claims without evidence
        specific_claims = [
            "wearing a", "holding a", "sitting on", "standing next to",
            "color is", "size is", "position is"
        ]
        
        claim_count = sum(1 for claim in specific_claims if claim in response_lower)
        if claim_count > 3 and has_uncertainty:
            penalty += 0.15
        
        return min(penalty, 1.0)

    def _assess_response_quality(self, response: str, query: str) -> float:
        """
        Assess the overall quality and completeness of the response.
        
        Args:
            response (str): The generated response
            query (str): The original query
            
        Returns:
            float: Quality score (0.0 to 1.0)
        """
        response_lower = response.lower()
        query_lower = query.lower()
        
        score = 0.5  # Base score
        
        # Check response length adequacy
        response_words = len(response.split())
        query_words = len(query.split())
        
        # Short responses to complex queries get penalized
        if query_words > 5 and response_words < 8:
            score -= 0.2
        elif response_words > 15:  # Detailed responses get bonus
            score += 0.1
        
        # Check for specific details that indicate good analysis
        detail_indicators = [
            "wearing", "holding", "sitting", "standing", "color", "size",
            "position", "background", "foreground", "left", "right", "center",
            "next to", "behind", "in front of", "on the", "at the"
        ]
        
        detail_count = sum(1 for indicator in detail_indicators if indicator in response_lower)
        score += min(detail_count * 0.03, 0.2)  # Max 0.2 bonus for details
        
        # Check if response addresses the query
        query_keywords = [word for word in query_lower.split() if len(word) > 3]
        addressed_keywords = sum(1 for keyword in query_keywords if keyword in response_lower)
        
        if query_keywords:
            keyword_coverage = addressed_keywords / len(query_keywords)
            score += keyword_coverage * 0.2
        
        # Penalize very generic responses
        generic_phrases = [
            "i can see", "there is", "the image shows", "i notice",
            "appears to be", "looks like", "seems to be"
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase in response_lower)
        if generic_count > 2:
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def _assess_response_confidence(self, response: str, query: str, image: Image.Image) -> float:
        """
        Assess the confidence level of a generated response with focus on hallucination detection.
        
        Args:
            response (str): The generated response
            query (str): The original query
            image (Image.Image): The input image
            
        Returns:
            float: Confidence score between 0.0 and 1.0
        """
        # Start with base confidence
        confidence = 0.6
        
        # Detect hallucinations
        hallucination_penalty = self._detect_hallucinations(response, query)
        confidence -= hallucination_penalty * HALLUCINATION_PENALTY
        
        # Assess response quality
        quality_score = self._assess_response_quality(response, query)
        confidence += quality_score * 0.3
        
        # Check for uncertainty indicators (but with lower penalty)
        uncertainty_indicators = [
            "i don't know", "i cannot see", "i'm not sure", "unclear", 
            "cannot determine", "not visible", "hard to tell", "unable to"
        ]
        
        response_lower = response.lower()
        has_uncertainty = any(indicator in response_lower for indicator in uncertainty_indicators)
        
        if has_uncertainty:
            confidence -= UNCERTAINTY_PENALTY
        
        # Bonus for balanced responses (not too confident, not too uncertain)
        if 0.4 <= confidence <= 0.8:
            confidence += 0.05
        
        # Ensure confidence is within bounds
        confidence = max(0.0, min(1.0, confidence))
        
        return confidence

    def _postprocess_response(self, response: str, confidence: float) -> str:
        """
        Post-process the response based on confidence level.
        
        Args:
            response (str): The original response
            confidence (float): The confidence score
            
        Returns:
            str: The processed response
        """
        # Clean up the response
        response = response.strip()
        
        # If confidence is very low, return "I don't know"
        if confidence < CONFIDENCE_THRESHOLD:
            return "I don't know"
        
        # Ensure proper punctuation
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        return response

    def get_batch_size(self) -> int:
        """
        Determines the batch size used by the evaluator when calling batch_generate_response.

        The evaluator uses this value to determine how many queries to send in each batch.
        Valid values are integers between 1 and 16.

        Returns:
            int: The batch size, indicating how many queries should be processed together
                 in a single batch.
        """
        return AICROWD_SUBMISSION_BATCH_SIZE

    def prepare_formatted_prompts(self, queries: List[str], images: List[Image.Image],
                                  message_histories: List[List[Dict[str, Any]]]) -> List[str]:
        """
        Prepare formatted prompts for the model by applying the chat template.

        This method formats the prompts according to Llama's chat template,
        including system prompts, images, conversation history, and the current query.

        Args:
            queries (List[str]): List of user questions or prompts.
            images (List[Image.Image]): List of PIL Image objects to analyze.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories.
                Each conversation history is a list of message dictionaries with the
                following structure:
                - For user messages: {"role": "user", "content": "user message text"}
                - For assistant messages: {"role": "assistant", "content": "assistant response"}

                For multi-turn conversations, the history contains all previous turns.
                For single-turn queries, the history will be an empty list.

        Returns:
            List[str]: List of formatted prompts ready for the model.
        """
        formatted_prompts = []

        for query_idx, (query, image) in enumerate(zip(queries, images)):
            message_history = message_histories[query_idx]

            # Build messages list with system prompt and image
            messages = [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "image"}]}
            ]

            # Add conversation history if exists
            if message_history:
                messages.extend(message_history)

            # Add current query
            messages.append({"role": "user", "content": query})

            # Apply chat template
            formatted_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            formatted_prompts.append(formatted_prompt)

        return formatted_prompts

    def batch_generate_response(
            self,
            queries: List[str],
            images: List[Image.Image],
            message_histories: List[List[Dict[str, Any]]],
    ) -> List[str]:
        """
        Generate responses for a batch of queries with associated images using RRF-enhanced knowledge retrieval.

        This method is the main entry point called by the evaluator. It handles
        preparing the prompts, combining them with images, and generating responses
        through the HuggingFace transformers model with enhanced RRF-based knowledge retrieval.

        Args:
            queries (List[str]): List of user questions or prompts.
            images (List[Image.Image]): List of PIL Image objects, one per query.
                The evaluator will ensure that the dataset rows which have just
                image_url are populated with the associated image.
            message_histories (List[List[Dict[str, Any]]]): List of conversation histories,
                one per query. Each history is a list of message dictionaries with
                'role' and 'content' keys in the following format:

                - For single-turn conversations: Empty list []
                - For multi-turn conversations: List of previous message turns in the format:
                  [
                    {"role": "user", "content": "first user message"},
                    {"role": "assistant", "content": "first assistant response"},
                    {"role": "user", "content": "follow-up question"},
                    {"role": "assistant", "content": "follow-up response"},
                    ...
                  ]

        Returns:
            List[str]: List of generated responses, one per input query.
        """
        responses = []

        for query, image, message_history in zip(queries, images, message_histories):
            # Retrieve enhanced knowledge using RRF
            enhanced_knowledge = self._retrieve_enhanced_knowledge(query, image)
            knowledge_text = self._format_enhanced_knowledge(enhanced_knowledge)
            
            # Build enhanced system prompt with knowledge context
            enhanced_system_prompt = VISION_SYSTEM_PROMPT + "\n\n" + knowledge_text
            
            # Build messages list with enhanced system prompt and image
            messages = [
                {"role": "system", "content": enhanced_system_prompt},
                {"role": "user", "content": [{"type": "image"}]}
            ]

            # Add conversation history if exists
            if message_history:
                messages.extend(message_history)

            # Add current query
            messages.append({"role": "user", "content": query})

            # Apply chat template to format the messages into a string
            formatted_text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            # Process inputs
            inputs = self.processor(
                text=formatted_text,
                images=image,
                return_tensors="pt"
            ).to(self.model.device)

            # Generate response
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_GENERATION_TOKENS,
                    temperature=0.1,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )

            # Decode response
            response = self.processor.decode(
                output_ids[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            # Assess confidence with enhanced knowledge context
            confidence = self._assess_response_confidence(response, query, image)
            
            # Post-process response
            processed_response = self._postprocess_response(response, confidence)
            
            responses.append(processed_response)

        return responses
