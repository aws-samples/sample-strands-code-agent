import base64


def image_to_base64(image_path):
    """Read an image file and return its contents as a base64-encoded string."""
    with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
