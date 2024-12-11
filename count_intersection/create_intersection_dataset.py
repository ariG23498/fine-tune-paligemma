# Reference: https://lucasb.eyer.be/articles/bv_tuto.html

# Example usage:
# python create_intersection_dataset.py \
# --image_size=224 \
# --dpi=100 \
# --num_images=100 \
# --dataset_folder="dataset" \
# --dataset_split="train" \
# --push_to_hub=False

from matplotlib import pyplot as plt
import random
import os
from fire import Fire
from datasets import load_dataset


def are_lines_intersecting(p1, p2, p3, p4):
    # Created by ChatGPT
    # Unpack points
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    # Calculate the determinant
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        # Lines are parallel
        return False

    # Calculate the intersection parameters
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denominator

    # Check if the intersection point lies within the segments
    return 0 <= t <= 1 and 0 <= u <= 1


def create_image(
    x, y1, y2, figure_size, dpi, dataset_folder, dataset_split, idx, count_intersection
):
    plt.figure(figsize=(figure_size, figure_size), dpi=dpi)
    plt.plot(x, y1, color="red")
    plt.plot(x, y2, color="blue")
    plt.axis("off")
    file_name = f"{dataset_folder}/{dataset_split}/{count_intersection}/{idx:04}.png"
    plt.savefig(file_name, dpi=dpi)
    plt.close()


def main(
    image_size: int = 224,
    dpi: int = 100,
    num_images: int = 100,
    dataset_folder: str = "dataset",
    dataset_split: str = "train",
    push_to_hub: bool = False,
    dataset_id: str | None = None,
):
    if push_to_hub and dataset_id is None:
        raise ValueError("`dataset_id` cannot be `None` when you want to push the dataset to Hub")

    if not os.path.exists(f"{dataset_folder}/{dataset_split}"):
        os.makedirs(f"{dataset_folder}/{dataset_split}/0/")
        os.makedirs(f"{dataset_folder}/{dataset_split}/1/")
        os.makedirs(f"{dataset_folder}/{dataset_split}/2/")

    border = 10
    figure_size = image_size / dpi

    # x axis is fixed
    x = [border, (image_size - 2 * border) // 2, image_size - border]

    for number_of_intersection in range(3):
        idx = 0
        while idx < num_images:
            count_intersection = 0

            # y axis is random
            y1 = [random.randint(a=10, b=214) for _ in range(3)]
            y2 = [random.randint(a=10, b=214) for _ in range(3)]

            # check for intersection
            line1 = list(zip(x, y1))
            line2 = list(zip(x, y2))
            p1, p2 = line1[:2]
            p3, p4 = line2[:2]
            count_intersection = count_intersection + are_lines_intersecting(
                p1, p2, p3, p4
            )
            p1, p2 = line1[1:]
            p3, p4 = line2[1:]
            count_intersection = count_intersection + are_lines_intersecting(
                p1, p2, p3, p4
            )

            if count_intersection == number_of_intersection:
                create_image(
                    x,
                    y1,
                    y2,
                    figure_size,
                    dpi,
                    dataset_folder,
                    dataset_split,
                    idx,
                    count_intersection,
                )
                idx = idx + 1

    # push to hub
    if push_to_hub:
        ds = load_dataset(dataset_folder)
        ds.push_to_hub(dataset_id)


if __name__ == "__main__":
    Fire(main)
