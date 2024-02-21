docker run \
    --env UID="$(id -u)" --env GID="$(id -g)" \
    --volume $(pwd)/models:/home/app/models --volume $(pwd)/data:/home/app/data --volume $(pwd)/logs:/home/app/logs \
    shakespeare-babbler:latest