docker run -d --network=host --name my_postgres_3  -e POSTGRES_USER=wuy -e POSTGRES_PASSWORD=wuy -e POSTGRES_DB=wuy --memory=3g --shm-size=1g --memory-swap=4g --cpus 5  -v /home/wuy/DB/memory_prediction/cross_machines/3:/var/lib/postgresql/config  docker.1ms.run/postgres -c 'config_file=/var/lib/postgresql/config/postgresql.conf'
