FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /API

# Install necessary packages and configure timezone
RUN apt-get update && apt-get install -y procps build-essential && pip install -U pip \
    && rm /etc/localtime \
    && ln -s /usr/share/zoneinfo/America/Mexico_City /etc/localtime  \
    && apt-get install -y libgomp1

# Create and activate a virtual environment
RUN python -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Copy the requirements file into the container and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt || cat /API/requirements.txt

COPY main.py /API

# Expose port 8501
EXPOSE 8000

# Set the command to run Streamlit
CMD ["streamlit", "run", "main.py", "--server.address", "0.0.0.0", "--server.port", "8000"]


