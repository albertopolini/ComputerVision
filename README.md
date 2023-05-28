# ComputerVision


## Steps to initialize the Docker container


1. cd into the docker folder
2. Run the command

```

docker build -t <image_name> --build-arg token_name=<token> .
    
```
    
To build the container run:
    
```
docker run -it ^
--name <instance_name> ^
-p 8888:8888 ^
-v <Path/to/notebook/folder>:/home/Notebooks ^
-v <Path/to/data/folder>:/home/Data ^
-v <Path/to/scripts/folder>:/home/Scripts ^
<image_name>
```


If you are using Rancher Desktop you should map the folder paths with 'mnt/c' 