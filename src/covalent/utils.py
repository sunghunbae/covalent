import json
import base64
import zlib
import binascii
import numpy as np

from typing import Any


def recursive_round(data:Any, decimals:int=2) -> Any:
    """Recursively round float values to a given decimal places.

    Args:
    data: The input data, which can be a list, dictionary, or any
            other data type. It can contain nested lists and dictionaries.
    decimals: number of decimal places.
    """
    if not isinstance(decimals, int) or decimals < 0:
        raise ValueError("decimals must be a non-negative integer.")

    def _recursive_round(current_item):
        if isinstance(current_item, float):
            return round(current_item, decimals)
        elif isinstance(current_item, np.float64):
            return round(float(current_item), decimals)
        elif isinstance(current_item, list):
            return [_recursive_round(item) for item in current_item]
        elif isinstance(current_item, dict):
            return {key: _recursive_round(value) for key, value in current_item.items()}
        else:
            return current_item

    return _recursive_round(data)



def serialize(data: Any) -> str:
    """
    Serialize, compress, and encode data to a base64 string.

    Notes:
        The JSON specification only supports string keys in objects.
        For example, after JSON-serialization/deserialization, keys of integer type are changed to string.
        {1: 'a', 2: 'b', 3: 'c'} --> {'1': 'a', '2': 'b', '3': 'c'}
        Unfortunately, this is a fundamental limitation of JSON itself.
        Integer keys are not valid JSON.

    Args:
        data: Any JSON-serializable Python object

    Returns:
        Base64-encoded string
    """
    # 1. Serialize to JSON string
    json_str = json.dumps(data, separators=(',', ':'))  # Compact format

    # 2. Encode to bytes
    json_bytes = json_str.encode('utf-8')

    # 3. Compress
    compressed = zlib.compress(json_bytes)

    # 4. Base64 encode (no need to decode to str, keep as bytes if storing in binary)
    # Base64 output only contains: A-Z, a-z, 0-9, +, /, =
    encoded = base64.b64encode(compressed)

    # 5. Convert to string for text storage/transmission
    return encoded.decode('utf-8')



def deserialize(encoded_str: str) -> Any:
    """
    Decode, decompress, and deserialize a base64 string back to Python object.

    Args:
        encoded_str: Base64-encoded compressed JSON string

    Returns:
        Deserialized Python object
    """
    try:
        # 1. Convert string to bytes
        encoded_bytes = encoded_str.encode('utf-8')

        # 2. Base64 decode
        # Base64 output only contains: A-Z, a-z, 0-9, +, /, =
        compressed = base64.b64decode(encoded_bytes)

        # 3. Decompress
        json_bytes = zlib.decompress(compressed)

        # 4. Decode bytes to string
        json_str = json_bytes.decode('utf-8')

        # 5. Parse JSON
        return json.loads(json_str)

    except (zlib.error, binascii.Error, json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"Failed to deserialize data: {e}")