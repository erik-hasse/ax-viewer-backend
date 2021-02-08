all: clean local

local: clean
	docker build -t backendserver .
	docker run --name backend -p 80:80 -v /app/static:/app/static -d backendserver

clean:
	docker ps | grep backend | awk '{print $$1}' | xargs docker stop || true
	docker ps -a | grep backend | awk '{print $$1}' | xargs docker rm || true

.PHONY:
