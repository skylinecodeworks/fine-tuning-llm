import random
import requests
import time
from datetime import datetime, timedelta
from pymongo import MongoClient


def get_random_date():
    """Devuelve una fecha aleatoria de los últimos 10 años."""
    start_date = datetime.now() - timedelta(days=365 * 10)
    random_days = random.randint(0, 365 * 10)
    date = start_date + timedelta(days=random_days)
    return date.strftime("%Y-%m-%d")


def generate_question(date):
    """Genera una pregunta aleatoria sobre el clima en Madrid para una fecha dada."""
    templates = [
        "¿Cómo fue el clima en Madrid el {date}?",
        "¿Qué temperatura hizo en Madrid el {date} y llovió?",
        "Dame información del clima en Madrid en la fecha {date}.",
        "¿Hubo precipitaciones en Madrid el día {date}?",
        "¿Cuál fue el clima en Madrid el {date}?"
    ]
    template = random.choice(templates)
    return template.format(date=date)


def get_weather_data(date):
    """Consulta la API de Open-Meteo para obtener datos históricos del clima en Madrid."""
    
    time.sleep(3)
    
    latitude = 40.4168
    longitude = -3.7038
    url = (
        f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}"
        f"&start_date={date}&end_date={date}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum"
        f"&timezone=Europe%2FMadrid"
    )
    print(f"Consultando API para {date}: {url}...")
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Error al consultar API para {date}: {response.status_code}")
        return None

    data = response.json()
    if "daily" not in data or not data["daily"]["temperature_2m_max"]:
        return None

    temp_max = data["daily"]["temperature_2m_max"][0]
    temp_min = data["daily"]["temperature_2m_min"][0]
    precipitation = data["daily"]["precipitation_sum"][0]

    print(f" {data}")
    return {
        "temp_max": temp_max,
        "temp_min": temp_min,
        "precipitation": precipitation
    }


def generate_answer(date, weather):
    """Genera una respuesta basada en los datos del clima."""
    if not weather:
        return "No hay datos disponibles para esa fecha."

    response_templates = [
        "El {date}, Madrid registró una temperatura entre {min}°C y {max}°C, con {precip} mm de precipitaciones.",
        "En Madrid el {date}, la temperatura osciló entre {min} y {max} grados. Se registraron {precip} milímetros de lluvia.",
        "Madrid tuvo el {date} una mínima de {min}°C y una máxima de {max}°C. Cayeron {precip} mm de precipitaciones.",
    ]
    template = random.choice(response_templates)
    return template.format(
        date=date,
        min=weather["temp_min"],
        max=weather["temp_max"],
        precip=weather["precipitation"]
    )


def main():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["datasets"]
    collection = db["smoltest2"]
    collection.delete_many({})

    print("Generando datos sintéticos del clima en Madrid...")

    for _ in range(200):  # ajustá este número según tu necesidad
        date = get_random_date()
        question = generate_question(date)
        weather = get_weather_data(date)
        answer = generate_answer(date, weather)

        document = {
            "prompt": question,
            "response": answer,
            "location": "Madrid, España",
            "date": date,
            "categoria": "Clima",
            "etiquetas": ["Clima", "Madrid", "Histórico", "Temperatura", "Precipitaciones"],
            "dificultad": "Beginner"
        }
        collection.insert_one(document)
        print("Insertado en MongoDb.")

    print(f"Inserción completada.")


if __name__ == "__main__":
    main()
