import time
import torch

def main():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_properties(i).name)

    wait_seconds = 20
    time.sleep(wait_seconds)

    return


if __name__ == '__main__':
    main()
