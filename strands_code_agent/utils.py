import base64


def image_to_base64(image_path):
    """Read an image file and return its contents as a base64-encoded string."""
    with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')


def get_response_metrics(
          response,
          price_1M_input_tokens=None,
          price_1M_output_tokens=None):
    """
    For the latest pricing see: https://aws.amazon.com/bedrock/pricing
    """
    summary = response.metrics.get_summary()
    inputTokens = summary['accumulated_usage']['inputTokens']
    outputTokens = summary['accumulated_usage']['outputTokens']
    metrics = {
        'total_cycles': summary['total_cycles'],
        'total_duration': summary['total_duration'],
        'input_tokens': inputTokens,
        'output_tokens': outputTokens,
    }

    if price_1M_input_tokens and price_1M_output_tokens:
        metrics['cost'] = (inputTokens * price_1M_input_tokens / 1_000_000) + (outputTokens * price_1M_output_tokens / 1_000_000)

    return metrics
