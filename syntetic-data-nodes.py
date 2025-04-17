import random
from pymongo import MongoClient
import networkx as nx


def create_graph():
    """
    Genera un grafo conectado usando el modelo de Watts-Strogatz.
    Se generan 100 nodos (numerados de 0 a 99) y se conecta cada nodo con 4 vecinos.
    A cada arista se le asigna un peso aleatorio entre 1 y 20.
    """
    n = 100
    # El parámetro k (número de vecinos conectados en forma circular) debe ser par.
    G = nx.connected_watts_strogatz_graph(n, 4, 0.1, tries=100)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(1, 20)
    return G


def generate_random_question(node):
    """
    Genera una pregunta para un nodo dado usando diferentes plantillas.
    La pregunta tiene un máximo de 200 caracteres.
    """
    templates = [
        "El nodo {node} ¿con quién está conectado?",
        "¿A qué nodos se conecta el nodo {node}?",
        "Lista las conexiones del nodo {node}.",
        "¿Qué vecinos tiene el nodo {node}?",
        "Muestra los enlaces del nodo {node}."
    ]
    template = random.choice(templates)
    question = template.format(node=node)
    return question[:200]


def generate_random_answer(G, node):
    """
    Genera una respuesta para un nodo dado.
    La respuesta indica las conexiones del nodo y la distancia (peso) de cada arista,
    utilizando diversas plantillas en cada conexión. Se trunca a 200 caracteres.
    """
    neighbors = list(G.neighbors(node))
    if not neighbors:
        # Dado que el grafo es conectado, esto no debería ocurrir.
        return f"El nodo {node} no tiene conexiones."

    prefix_options = [
        "El nodo {node} está conectado con ",
        "El nodo {node} se conecta a ",
        "Las conexiones del nodo {node} incluyen "
    ]
    prefix = random.choice(prefix_options).format(node=node)

    connection_phrases = []
    # Ordenar los vecinos para asegurar consistencia en la respuesta
    for neighbor in sorted(neighbors):
        weight = G[node][neighbor]['weight']
        phrase_options = [
            "el nodo {neighbor} (distancia {distance})",
            "{neighbor} [distancia {distance}]",
            "con {neighbor} mediante distancia {distance}"
        ]
        phrase = random.choice(phrase_options).format(neighbor=neighbor, distance=weight)
        connection_phrases.append(phrase)
    answer = prefix + ", ".join(connection_phrases) + "."
    return answer[:200]


def generate_random_tags():
    """
    Selecciona aleatoriamente entre 2 y 4 etiquetas de una lista predefinida.
    """
    possible_tags = ["Graph", "Network", "Distance", "Connectivity", "Nodes"]
    num_tags = random.randint(2, 4)
    return random.sample(possible_tags, num_tags)


def generate_random_category():
    """
    Selecciona aleatoriamente una categoría.
    """
    categories = ["Graph Theory", "Networking", "Data Structures"]
    return random.choice(categories)


def generate_random_difficulty():
    """
    Selecciona aleatoriamente el nivel de dificultad.
    """
    difficulties = ["Beginner", "Intermediate", "Advanced"]
    return random.choice(difficulties)


def main():
    # 1. Generar el grafo con estructura real y pesos válidos para calcular distancias.
    G = create_graph()

    # 2. Conectar a MongoDB en localhost
    client = MongoClient("mongodb://localhost:27017/")
    db = client["datasets"]
    collection = db["smoltest"]

    # 3. Eliminar documentos existentes en la colección
    result = collection.delete_many({})
    print(f"Se eliminaron {result.deleted_count} documentos de la colección 'smoltest'.")

    documents = []
    # 4. Generar 10,000 documentos basados en el grafo
    for _ in range(1000):
        # Seleccionar un nodo aleatorio del grafo
        node = random.choice(list(G.nodes()))
        question = generate_random_question(node)
        answer = generate_random_answer(G, node)
        document = {
            "prompt": question,
            "response": answer,
            "etiquetas": generate_random_tags(),
            "categoria": generate_random_category(),
            "dificultad": generate_random_difficulty()
        }
        documents.append(document)

    # Inserción masiva de documentos
    collection.insert_many(documents)
    print(f"Se insertaron {len(documents)} documentos en la colección 'smoltest'.")


if __name__ == "__main__":
    main()
