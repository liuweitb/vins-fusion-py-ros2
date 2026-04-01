all: help

help:
	@echo ""
	@echo "-- Help Menu"
	@echo ""
	@echo "   1. make build   - build docker image"
	@echo "   2. make clean   - remove docker image"
	@echo ""

build:
	@docker build --tag vins-ros2-humble -f ./Dockerfile ..

clean:
	@docker rmi -f vins-ros2-humble
