# mcp_server.py
import httpx
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Iris Classifier", log_level="WARNING")


@mcp.tool()
def classify_iris(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> str:
    """Classify an iris flower from its four measurements, in centimeters.
    Returns the predicted species: setosa, versicolor, or virginica.
    """
    response = httpx.post(
        "http://127.0.0.1:8000/predict",
        json={
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        },
    )
    return f"Predicted species: {response.json()['species']}"


@mcp.tool()
def iris_probabilities(sepal_length: float, sepal_width: float, petal_length: float, petal_width: float) -> str:
    """Get the model's confidence for each iris species from four measurements, in centimeters.
    Returns a probability between 0 and 1 for setosa, versicolor, and virginica (they sum to about 1).
    Use this instead of classify_iris when the user asks how sure the model is, wants to compare
    species, or gives measurements that may sit between two species (common for versicolor vs.
    virginica). If only the single most likely species is needed, use classify_iris.
    """
    response = httpx.post(
        "http://127.0.0.1:8000/probabilities",
        json={
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width,
        },
    )
    probs = response.json()
    return ", ".join(f"{species}: {p:.2f}" for species, p in probs.items())


if __name__ == "__main__":
    mcp.run()
