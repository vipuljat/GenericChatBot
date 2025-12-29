FROM python:3.11.8

WORKDIR /app

COPY requirements2.txt .
RUN pip install --no-cache-dir -r requirements2.txt

COPY . .

RUN chmod +x /app/entrypoint.sh

EXPOSE 8001

CMD ["/app/entrypoint.sh"]

