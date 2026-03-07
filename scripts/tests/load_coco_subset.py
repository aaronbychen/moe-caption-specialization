from datasets import load_dataset


def main():
    dataset = load_dataset("phiyodr/coco2017", split="train[:20]")
    print(dataset)
    print(dataset[0].keys())
    print(dataset[0]["captions"])
    print(dataset[0]["captions"][0])


if __name__ == "__main__":
    main()