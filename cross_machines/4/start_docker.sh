docker run -d --network=host --name my_postgres_4  -e POSTGRES_USER=wuy -e POSTGRES_PASSWORD=wuy -e POSTGRES_DB=wuy --memory=4g --shm-size=4g --memory-swap=8g --cpus 5  -v /home/wuy/DB/memory_prediction/cross_machines/4:/var/lib/postgresql/config  docker.1ms.run/postgres -c 'config_file=/var/lib/postgresql/config/postgresql.conf'
