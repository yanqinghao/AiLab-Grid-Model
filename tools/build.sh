docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/grid-docker:$1 -f docker/docker_grid/Dockerfile .
docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/grid-stream:$1 -f docker/docker_grid_stream/Dockerfile .

docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/grid-docker:$1
docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/grid-stream:$1
