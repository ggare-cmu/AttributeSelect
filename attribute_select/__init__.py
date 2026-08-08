"""AttributeSelect: Distribution-Conditioned Attribute Selection for vision-language models.

Pipeline modules, each runnable as a CLI:
  python -m attribute_select.extract_features     images -> CLIP image embeddings
  python -m attribute_select.encode_attributes    attribute pool txt -> CLIP text embeddings
  python -m attribute_select.attribute_features   image embeddings x attribute embeddings -> similarity features
  python -m attribute_select.select_attributes    similarity features + labels -> per-class attribute lists
  python -m attribute_select.evaluate             attribute lists -> zero-shot top-1 accuracy
  python -m attribute_select.describe_shift       two datasets' similarity features -> shift described in words
"""

__version__ = "0.1.0"

TEMPERATURE = 100.0  # CLIP logit scale used for similarity features
