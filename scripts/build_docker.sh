#!/usr/bin/env bash

REGISTRY=mybigpai-public-registry.cn-beijing.cr.aliyuncs.com/easyrec
DOCKER_TAG=0.7

rm -rf docker/requirements
cp -r requirements/ docker/requirements
cd docker

for DEVICE in cu124 cpu
do
    case ${DEVICE} in
        "cu124") BASE_IMAGE=nvidia/cuda:12.4.1-devel-ubuntu22.04 ;;
        * )      BASE_IMAGE=ubuntu:22.04 ;;
    esac
    docker build --network host -t ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-${DEVICE} --build-arg DEVICE=${DEVICE} --build-arg BASE_IMAGE=${BASE_IMAGE} .
    docker push ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-${DEVICE}
done

docker images -q ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-cu124 | xargs -I {} docker tag {} ${REGISTRY}/tzrec-devel:${DOCKER_TAG}
docker images -q ${REGISTRY}/tzrec-devel:${DOCKER_TAG}-cu124 | xargs -I {} docker tag {} ${REGISTRY}/tzrec-devel:latest
docker push ${REGISTRY}/tzrec-devel:${DOCKER_TAG}
docker push ${REGISTRY}/tzrec-devel:latest
