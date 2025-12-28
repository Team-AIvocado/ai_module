from llm_module.app.services.multimodal_mock import mock_food_classifier
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TODO: 이부분 적정값으로 수정 - MLOps를 위해서 임시 0 으로 설정
CONFIDENCE_THRESHOLD = 0.0

async def process_fallback(result, image_bytes=None, image_url=None):
    """
    Checks the confidence of the inference result.
    If it's below the threshold, calls the LLM service and updates the result.
    The new candidates list will contain:
    1. LLM Result
    2. Original Top 1
    3. Original Top 2
    """
    top_conf = 0.0
    if result.get("candidates") and len(result["candidates"]) > 0:
        top_conf = result["candidates"][0]["confidence"]
    
    # Dynamic Threshold Check
    try:
        from inference_module.app.services.model_loader import effnet_cls
        threshold = getattr(effnet_cls, "threshold", 0.0)
    except ImportError:
        threshold = 0.0 # Safety default

    # Fallback Trigger: If V4 top confidence is low
    if top_conf < threshold:
        logger.info(f"Low Confidence Detected ({top_conf:.2f} < {threshold}). Calling LLM Fallback...")
        try:
            # Call LLM
            llm_food_name = mock_food_classifier(image_bytes=image_bytes, image_url=image_url)
            
            # Clean Label (Remove dots and spaces)
            llm_cleaned = llm_food_name.replace(".", "").strip()
            logger.info(f"LLM Result: {llm_cleaned} (Original: {llm_food_name})")
            
            # Construct new candidates list
            original_candidates = result.get("candidates", [])
            new_candidates = []
            
            # Check for duplication
            is_new = True
            for cand in original_candidates:
                if cand["label"] == llm_cleaned:
                    is_new = False
                    # If duplicate: Add 1.1 to EXISTING confidence
                    # This preserves original score (val - 1.1) and signals fallback (>1.0)
                    cand["confidence"] += 1.1
                    new_candidates.append(cand)
                else:
                    new_candidates.append(cand)
            
            # If new label from LLM: Insert at top with confidence 1.1
            if is_new:
                new_candidates.insert(0, {"label": llm_cleaned, "confidence": 1.1})
            else:
                # If duplicate and modified, we need to resort to ensure it's on top
                new_candidates.sort(key=lambda x: x["confidence"], reverse=True)
                
            # Formatting result
            return {
                "image_id": result["image_id"],
                "food_name": new_candidates[0]["label"],
                "candidates": new_candidates
            }
        except Exception as e:
            logger.error(f"LLM Fallback failed: {e}")
            return result # Return original result if fallback fails
            
    return result
