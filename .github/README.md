# Fine Tuning PaliGemma

PaliGemma (PG) is a family of Vision Language Models from Google. It uses
SigLIP as the vision encoder, and the Gemma family of models as it language counterpart.

> [!Note]
I and Ritwik Raha have covered SigLIP in depth in our blog
[Choosing Between SigLIP and CLIP for Language Image Pretraining](https://blog.ritwikraha.dev/choosing-between-siglip-and-clip-for-language-image-pretraining) if you wanted to read
about it.

PaliGemma is great for fine tuning purposes. In this repository we collect some
examples of fine tuning the PG family of models.

Models hosted on Hugging Face Hub:

1. [PaliGemma](https://huggingface.co/collections/google/paligemma-release-6643a9ffbf57de2ae0448dda)
2. [PaliGemma 2](https://huggingface.co/collections/google/paligemma-2-release-67500e1e1dbfdd4dee27ba48)


## Image Captioning

In the script provided we have used the [`tuxemon`](https://huggingface.co/datasets/diffusers/tuxemon)
dataset, from the diffusers team. The dataset comprises of images of tuxemons (a spin off of pokemons)
and their captions.

| Before Fine Tuning | After Fine Tuning |
|---|---|
| ![image](./assets/image_caption/before.png) | ![image](./assets/image_caption/after.png) |


## Object Detection

While I could not find a document that provides pointers to train the model
on a detection dataset, diving in the official
[big vision space](https://huggingface.co/spaces/big-vision/paligemma) made it
really clear. Taking inspiration from the space, I have create a script to format
any object detection dataset (here the dataset is based on the coco format)
to the format PaliGemma is trained on.

You can find the dataset creation script here: `create_od_dataset.py`.

After the dataset is created run the fine tuning script `object_detection_ft.py`
and run the model.

| Before Fine Tuning | After Fine Tuning |
|---|---|
| ![image](./assets/object_detection/before.png) | ![image](./assets/object_detection/after.png) |


## Count Intersection

Find more information in the [count intersection readme](../count_intersection/README.md).

## Citation

If you like our work and would use it please cite us
```
@misc{github_repository,
  author = {Aritra Roy Gosthipaty, Ritwik Raha}, 
  title = {ft-pali-gemma}, 
  publisher = {{GitHub}(https://github.com)},
  howpublished = {\url{https://github.com/ariG23498/ft-pali-gemma/edit/main/README.md}},
  year = {2024}  
}
```
