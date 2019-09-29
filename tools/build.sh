docker build -t registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/grid-docker:$1 -f docker/docker_grid/Dockerfile .

docker push registry-vpc.cn-shanghai.aliyuncs.com/shuzhi/grid-docker:$1
