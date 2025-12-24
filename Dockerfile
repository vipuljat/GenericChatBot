FROM python:3.10.8

# Create the app directory
RUN mkdir /app
WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY . /app/


EXPOSE 8001

# Copy the entrypoint script
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint
CMD ["/app/entrypoint.sh"]
