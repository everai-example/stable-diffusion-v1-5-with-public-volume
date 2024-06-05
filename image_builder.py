from everai.image import Builder

IMAGE = 'quay.io/mc_jones/stable-diffusion-v1-5:v0.0.1'

image_builder = Builder.from_dockerfile(
    'Dockerfile',
    labels={
        "any-your-key": "value",
    },
    repository=IMAGE,
    platform=['linux/arm64', 'linux/x86_64'],
)
