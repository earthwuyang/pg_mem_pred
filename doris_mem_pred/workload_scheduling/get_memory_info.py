

def parse_metrics(metrics_text):
    metrics = {}
    for line in metrics_text.splitlines():
        if line and not line.startswith('#'):
            try:
                key, value = line.split()
                metrics[key] = float(value)
            except ValueError:
                continue
    return metrics

def get_available_memory():

    import requests

    # Replace with your BE host and port
    BE_HOST = '101.6.5.215'
    BE_PORT = 8040

    metrics_url = f'http://{BE_HOST}:{BE_PORT}/metrics'

    try:
        response = requests.get(metrics_url)
        response.raise_for_status()
        metrics_data = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching metrics: {e}")
        metrics_data = None
    if metrics_data:
        print(f"metrics_data: {metrics_data}")
        metrics = parse_metrics(metrics_data)

        available_memory = metrics.get('doris_be_memory_jemalloc_retained_bytes', 0)
        # print(f"available_memory: {available_memory}")
        return available_memory / 1024**2


if __name__ == '__main__':
    get_available_memory()