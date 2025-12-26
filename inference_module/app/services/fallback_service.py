from llm_module.app.services.multimodal_mock import mock_food_classifier

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
    
    if top_conf < CONFIDENCE_THRESHOLD:
        print(f"Low confidence ({top_conf:.4f} < {CONFIDENCE_THRESHOLD}), triggering LLM fallback...")
        try:
            # Call LLM
            llm_food_name = mock_food_classifier(image_bytes=image_bytes, image_url=image_url)
            print(f"LLM Result: {llm_food_name}")
            
            # Construct new candidates list
            original_candidates = result.get("candidates", [])
            new_candidates = [
                {"label": llm_food_name, "confidence": 0.0}  # Rank 1: LLM
            ]
            
            # Add top 2 original candidates (if available)
            if len(original_candidates) > 0:
                new_candidates.append(original_candidates[0]) # Rank 2
            if len(original_candidates) > 1:
                new_candidates.append(original_candidates[1]) # Rank 3
                
            # Formatting result
            return {
                "image_id": result["image_id"],
                "food_name": llm_food_name,
                "candidates": new_candidates
            }
        except Exception as e:
            print(f"LLM Fallback failed: {e}")
            return result # Return original result if fallback fails
            
    return result
