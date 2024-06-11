# Stable Diffusion 1.5 with public volume
In the quickstart, you created a simple application. In this example, we use the `Stable Diffusion 1.5` model files in public volume and implement an AIGC(AI generated content) online text-to-image and image-to-image service.  

## Create an app
Create a directory for your app firstly. In your app directory, you should login by token you got in [EverAI](https://everai.expvent.com). After login successfully, run command `everai app create` to create your app.  

```bash
everai login --token <your token>

everai app create <your app name>
```
## Create secrets
If you have already created a secret for your registry, you can skip this step.

In this case, we will create one secret for [quay.io](https://quay.io/).  

```bash
everai secret create your-quay-io-secret-name \
  --from-literal username=<your username> \
  --from-literal password=<your password>
```
>**NOTE**
>
>[quay.io](https://quay.io/) is a well-known public image registry. Well-known image registry similar to [quay.io](https://quay.io/) include [Docker Hub](https://hub.docker.com/), [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry), [Google Container Registry](https://cloud.google.com/artifact-registry), etc.  

## Write your app code in python
### Basic setup
There is an example code in [app.py](https://github.com/everai-example/stable-diffusion-v1-5-with-public-volume/blob/main/app.py).  

First, import the required EverAI Python class library. Then define the variable names that need to be used, including the volume, the secret that accesses the image registry, and the file stored in the volume. Use the `Image.from_registry` static method to create a image instance. Create and define an app instance through the App class.  

What needs to be noted here is that you need to configure GPU resources for your application. The GPU model configured here is "A100 40G", and the number of GPU cards is 1.  

```python
from everai.app import App, context, VolumeRequest
from everai.autoscaling import SimpleAutoScalingPolicy
from everai.image import Image, BasicAuth
from everai.resource_requests import ResourceRequests
from everai.placeholder import Placeholder
from image_builder import IMAGE

APP_NAME = '<your app name>'
VOLUME_NAME = 'expvent/models--runwayml--stable-diffusion-v1-5'
QUAY_IO_SECRET_NAME = 'your-quay-io-secret-name'
HUGGINGFACE_SECRET_NAME = 'your-huggingface-secret-name'
MODEL_NAME = 'runwayml/stable-diffusion-v1-5'
CONFIGMAP_NAME = 'sd15-configmap'

image = Image.from_registry(IMAGE, auth=BasicAuth(
        username=Placeholder(QUAY_IO_SECRET_NAME, 'username', kind='Secret'),
        password=Placeholder(QUAY_IO_SECRET_NAME, 'password', kind='Secret'),
    ))
    
app = App(
    APP_NAME,
    image=image,
    volume_requests=[
        VolumeRequest(name=VOLUME_NAME, create_if_not_exists=True),
    ],
    secret_requests=[
        HUGGINGFACE_SECRET_NAME,
        QUAY_IO_SECRET_NAME
    ],
    configmap_requests=[CONFIGMAP_NAME],
    autoscaling_policy=SimpleAutoScalingPolicy(
        # keep running workers even no any requests, that make reaction immediately for new request
        min_workers=Placeholder(kind='ConfigMap', name=CONFIGMAP_NAME, key='min_workers'),
        # the maximum works setting, protect your application avoid to pay a lot of money
        # when an attack or sudden traffic
        max_workers=Placeholder(kind='ConfigMap', name=CONFIGMAP_NAME, key='max_workers'),
        # this factor controls autoscaler how to scale up your app
        max_queue_size=Placeholder(kind='ConfigMap', name=CONFIGMAP_NAME, key='max_queue_size'),
        # this factor controls autoscaler how to scale down your app
        max_idle_time=Placeholder(kind='ConfigMap', name=CONFIGMAP_NAME, key='max_idle_time'),
        # this factor controls autoscaler how many steps to scale up your app from queue 
        scale_up_step=Placeholder(kind='ConfigMap', name=CONFIGMAP_NAME, key='scale_up_step'),
    ),
    resource_requests=ResourceRequests(
        cpu_num=2,
        memory_mb=20480,
        gpu_num=1,
        gpu_constraints=[
            "A100 40G"
        ],
    ),
)
```

### Load model

You can load the model using the model file in the public volume `expvent/models--runwayml--stable-diffusion-v1-5` we provide.  

```python
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import torch

txt2img_pipe = None
img2img_pipe = None

@app.prepare()
def prepare_model():
    volume = context.get_volume(VOLUME_NAME)
    assert volume is not None and volume.ready

    secret = context.get_secret(HUGGINGFACE_SECRET_NAME)
    assert secret is not None
    huggingface_token = secret.get('token-key-as-your-wish')

    model_dir = volume.path

    global txt2img_pipe
    global img2img_pipe

    txt2img_pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME,
                                                        token=huggingface_token,
                                                        cache_dir=model_dir,
                                                        revision="fp16", 
                                                        torch_dtype=torch.float16, 
                                                        low_cpu_mem_usage=False,
                                                        local_files_only=True
                                                        )
    
    # The self.components property can be useful to run different pipelines with the same weights and configurations without reallocating additional memory.
    # If you just want to use img2img pipeline, you should use StableDiffusionImg2ImgPipeline.from_pretrained below.
    img2img_pipe = StableDiffusionImg2ImgPipeline(**txt2img_pipe.components)
    #image_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(MODEL_NAME,
    #                                                            token=huggingface_token,
    #                                                            cache_dir=model_dir,
    #                                                            revision="fp16",
    #                                                            torch_dtype=torch.float16, 
    #                                                            low_cpu_mem_usage=False,
    #                                                            local_files_only=True
    #                                                            )
      
    if torch.cuda.is_available():
            txt2img_pipe.to("cuda")
            img2img_pipe.to("cuda")
    elif torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        txt2img_pipe.to(mps_device)
        img2img_pipe.to(mps_device)
```
If you want to use `everai app run` to debug this example locally, your local debugging environment needs to have GPU resources(Nvidia GPU or GPU for MacOS devices with Metal programming framework), and use `everai volume pull` command to pull the model file from the cloud to the local environment before debugging the code.  

```bash
everai volume pull expvent/models--runwayml--stable-diffusion-v1-5
```

### Generate inference service
#### Text generates image
Aftering loading `Stable Diffusion 1.5` model, now you can write your Python code that uses `flask` to implement the inference online service of AIGC(AI generated content).  

```python
import flask
from flask import Response

import io

# service entrypoint
# api service url looks https://everai.expvent.com/api/routes/v1/stable-diffusion-v1-5/txt2img
# for test local url is http://127.0.0.1:8866/txt2img
@app.service.route('/txt2img', methods=['GET','POST'])
def txt2img():    
    if flask.request.method == 'POST':
        data = flask.request.json
        prompt = data['prompt']
    else:
        prompt = flask.request.args["prompt"]

    pipe_out = txt2img_pipe(prompt)

    image_obj = pipe_out.images[0]

    byte_stream = io.BytesIO()
    image_obj.save(byte_stream, format="PNG")

    return Response(byte_stream.getvalue(), mimetype="image/png")
```

#### Image generates image
Now you can write your Python code that uses `flask` to implement the inference online image-to-image service of AIGC(AI generated content).  

```python
import PIL
from io import BytesIO

image_pipe = None

# service entrypoint
# api service url looks https://everai.expvent.com/api/routes/v1/stable-diffusion-v1-5/img2img
# for test local url is http://127.0.0.1:8866/img2img
@app.service.route('/img2img', methods=['POST'])
def img2img():        
    f = flask.request.files['file']
    img = f.read()

    prompt = flask.request.form['prompt']
    
    init_image = PIL.Image.open(BytesIO(img)).convert("RGB")
    init_image = init_image.resize((768, 512))

    pipe_out = img2img_pipe(prompt=prompt, image=init_image, strength=0.75, guidance_scale=7.5)

    image_obj = pipe_out.images[0]

    byte_stream = io.BytesIO()
    image_obj.save(byte_stream, format="JPEG")

    return Response(byte_stream.getvalue(), mimetype="image/jpg")
```

## Build image
This step will build the container image, using two very simple files `Dockerfile` and `image_builder.py`.  

There is an example code in [image_builder.py](https://github.com/everai-example/stable-diffusion-v1-5-with-public-volume/blob/main/image_builder.py).

In `image_builder.py`, you should set your image repo.  

In this example, we choose to use [quay.io](https://quay.io/) as the public image registry to store application images. You can also use well-known image registry similar to [quay.io](https://quay.io/), such as [Docker Hub](https://hub.docker.com/), [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry), [Google Container Registry](https://cloud.google.com/artifact-registry), etc. If you have a self-built image registry and the image can be accessed on the Internet, you can also use it.  

```python
from everai.image import Builder

IMAGE = 'quay.io/<username>/<repo>:<tag>'
```
The dependence of this step is docker and buildx installed on your machine. Otherwise we will have further prompts to help you install them.  

```bash
docker login quay.io
docker buildx version
```

Then call the following command will compile the image and push them to your specified registry.  

```bash
everai image build
```

## Deploy image
The final step is to deploy your app to everai and keep it running.  

```bash
everai app deploy
```
After running `everai app list`, you can see the result similar to the following. If your app's status is `DEPLOYED`, it means that your app is deployed successfully.  

```bash
NAME                         STATUS     CREATED_AT                ROUTE_NAME
---------------------------  ---------  ------------------------  ---------------------------
stable-diffusion-v1-5        DEPLOYED   2024-05-19 18:47:32+0800  stable-diffusion-v1-5
```
When your app is deployed, you can use `curl` to execute the following request to test your deployed text-to-image code, A picture generated by the `Stable Diffusion 1.5` model will be downloaded in the current directory on the console.  

```bash
curl -X POST -d '{"prompt": "a photo of a cat on the boat"}' -H 'Content-Type: application/json' -H'Authorization: Bearer <your_token>' -o test.png https://everai.expvent.com/api/routes/v1/<your app route name>/txt2img
```

Open the picture and you can see the following effect.  

<img src="https://expvent.com.cn:1111/evfiles/v1/expvent/public/everai-documentation/demo-cat.png" width = "512" />  

You also can use `curl` to execute the following request to test your deployed image-to-image code.  

Before using `curl` to request, you need to download `sketch-mountains-input.jpg` to your local directory, execute `curl` in the directory of the console terminal, and a new image based on this original image of `sketch-mountains-input.jpg` and `prompt` will be generated by the large model `Stable Diffusion 1.5`.  


```bash
curl -X POST -F 'file=@sketch-mountains-input.jpg' -F "prompt=A fantasy landscape, trending on artstation" -H'Authorization: Bearer <your_token>' -o test.jpg https://everai.expvent.com/api/routes/v1/<your app route name>/img2img
```

An example of the original image is shown below.  

<img src="https://expvent.com.cn:1111/evfiles/v1/expvent/public/everai-documentation/sketch-mountains-input.jpg" width = "512" />

Open the new picture and you can see the following effect.  

<img src="https://expvent.com.cn:1111/evfiles/v1/expvent/public/everai-documentation/img2img-output.jpg" width = "512" />
