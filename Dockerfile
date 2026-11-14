FROM python:3.11-slim

# Evita la creazione di file .pyc, forza l'output immediato nei log
# e imposta la root del progetto nel PYTHONPATH per gli import tra moduli
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Dipendenze di sistema necessarie per compilazione pacchetti scientifici e git
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Aggiornamento pip e installazione delle librerie Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia dei file del progetto
COPY . .

# Mantiene il container pronto per l'esecuzione interattiva
CMD ["bash"]